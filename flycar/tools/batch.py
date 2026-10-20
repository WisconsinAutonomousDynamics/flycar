import argparse
import math
from pathlib import Path
import sys

from flycar.env.core import FlyCarEnv
from flycar.drivers import make_driver
import pandas as pd
from multiprocessing import Pool
from multiprocessing import get_context


def run_one(env, driver, seed):
    observation, info = env.reset(seed=seed)
    driver.reset()

    path_length = 0.0
    contact_steps = 0
    x, y, _ = info["pose"]

    while True:
        action = driver.act(observation)
        observation, reward, terminated, truncated, info = env.step(action)

        path_length += math.hypot(info["pose"][0] - x, info["pose"][1] - y)
        if observation["contact"].any():
            contact_steps += 1

        x, y, _ = info["pose"]

        if terminated or truncated:
            break

    if terminated:
        reason = "success"
    elif truncated:
        reason = "timeout"
    else:
        reason = "stopped"

    return {
        "seed": seed,
        "success": info["success"],
        "time_to_target": info["time"] if info["success"] else float("nan"),
        "target_distance": info["target_distance"],
        "contact_steps": contact_steps,
        "path_length": path_length,
        "duration_s": info["time"],
        "termination_reason": reason,
    }

def run_batch(scenarios, drivers, seeds, workers=1):
    jobs = [
        (scenario, driver, seed)
        for scenario in scenarios
        for driver in drivers
        for seed in seeds
    ]
    if workers == 1:
        results = [one_job(job) for job in jobs]
    else:
        with get_context("spawn").Pool(processes=workers) as pool:
            results = pool.map(one_job, jobs, chunksize=1)
    return pd.DataFrame(results)
  

def write_table(table, out_dir):
  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  path = out_dir / "results.csv"
  table.to_csv(path, index=False)
  return path

def summarize(table):
    rows = []
    for (scenario, driver), group in table.groupby(["scenario", "driver"], sort=True):
        n = len(group)
        has_target = group["target_distance"].notna().any()

        row = {
            "scenario": scenario,
            "driver": driver,
            "episodes": n,
            "has_target": has_target,
            "mean_contact_steps": group["contact_steps"].mean(),
            "mean_path_length": group["path_length"].mean(),
        }
        if has_target:
            successes = int(group["success"].sum())
            low, high = wilson_interval(successes, n)
            row.update(
                successes=successes,
                success_rate=successes / n,
                ci_low=low,
                ci_high=high,
                median_time_to_target=group["time_to_target"].median(),
            )
        else:
            nan = float("nan")
            row.update(successes=nan, success_rate=nan, ci_low=nan, ci_high=nan,
                       median_time_to_target=nan)
        rows.append(row)
    return pd.DataFrame(rows)


def print_summary(summary):
    for _, r in summary.iterrows():
        label = f"{r['scenario']} / {r['driver']}"
        if not r["has_target"]:
            print(f"{label}: no target in this scenario, success rate not applicable "
                  f"({int(r['episodes'])} episodes)")
            continue
        median = r["median_time_to_target"]
        median_text = "n/a" if pd.isna(median) else f"{median:.2f}s"
        print(f"{label}: {int(r['successes'])}/{int(r['episodes'])} reached the target "
              f"({r['success_rate']:.2%}, 95% CI: [{r['ci_low']:.2%}, {r['ci_high']:.2%}]), "
              f"median time to target: {median_text}")


def comparison_table(summary):
    return summary.pivot(index="driver", columns="scenario", values="success_rate")

def wilson_interval(succeses, total):
  if total == 0:
    return float("nan"), float("nan")

  p = succeses / total
  z = 1.96
  denominator = 1 + (z**2 / total)
  centre_adjusted_probability = p + (z**2 / (2 * total))
  adjusted_standard_deviation = math.sqrt((p * (1 - p) + (z**2 / (4 * total))) / total)
  lower_bound = (centre_adjusted_probability - z * adjusted_standard_deviation) / denominator
  upper_bound = (centre_adjusted_probability + z * adjusted_standard_deviation) / denominator
  return lower_bound, upper_bound

def one_job(job_specs):
  try:
    scenario_path, driver_name, seed = job_specs
    env = FlyCarEnv(scenario_path, eye_backend="raycast")
    driver = make_driver(driver_name)

    res = run_one(env, driver, seed)
  finally:
    env.close()

  return {"scenario": Path(scenario_path).stem, "driver": driver_name, **res}

def parse_seeds(seeds_str):
  seeds = []
  for part in seeds_str.split(","):
    if "-" in part:
      start, end = map(int, part.split("-"))
      seeds.extend(range(start, end + 1))
    else:
      seeds.append(int(part))
  return seeds

def main():
  parser = argparse.ArgumentParser(description="Run a batch of FlyCar episodes.")
  parser.add_argument("--scenario", required=True, help="Scenario YAML file, e.g. scenarios/light_seek.yaml")
  parser.add_argument("--driver", default="braitenberg",
                          help="random, braitenberg, keyboard, or package.module:ClassName (default: braitenberg)")
  parser.add_argument("--seeds", default="0-9", help="Seeds to run")
  parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers (default: 1)")
  parser.add_argument("--out", help="Directory to write results")

  args = parser.parse_args()
  seeds = parse_seeds(args.seeds)
  scenarios = [s.strip() for s in args.scenario.split(",") if s.strip()]
  drivers = [d.strip() for d in args.driver.split(",") if d.strip()]

  table = run_batch(scenarios, drivers, seeds, workers=args.workers)

  if args.out:
    path = write_table(table, args.out)
    print(f"Results written to {path}")

  print()

  summary = summarize(table)
  print_summary(summary)
  
  print()

  if len(scenarios) > 1 or len(drivers) > 1:
    print(comparison_table(summary).to_string(float_format=lambda v: f"{v:.2f}"))
    

if __name__ == "__main__":
  sys.exit(main())



