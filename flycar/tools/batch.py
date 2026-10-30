# Batch runner for FlyCar: run many episodes (seeds) of one or more scenarios
# with one or more drivers, score them, and print a summary.
#
# Example:
#   python -m flycar.tools.batch --scenario scenarios/light_seek.yaml \
#       --driver braitenberg --seeds 0-49 --workers 4 --out outputs/
#
# How the pieces fit together:
#   main()       reads the command line and calls the functions below
#   run_batch()  builds the list of jobs and runs them (in a loop or a process pool)
#   one_job()    runs inside a worker: builds its own env + driver for ONE episode
#   run_one()    the episode itself: the reset / act / step loop, plus measurements
#   summarize(), print_summary(), comparison_table()  turn the table into results
#
# Because every job builds its own environment, an episode depends only on
# (scenario, driver, seed), so results do not depend on the number of workers.
#
# Standard library imports.
import argparse
import math
from pathlib import Path
import sys

# FlyCar imports: the environment class and the driver factory.
from flycar.env.core import FlyCarEnv
from flycar.drivers import make_driver
# pandas holds the results table; get_context picks how worker processes start.
import pandas as pd
from multiprocessing import get_context


# Run ONE episode and return its measurements as a dict (one table row).
# env and driver are passed in already built, so this can be tested with any env.
def run_one(env, driver, seed):
    # Start a fresh episode. The same seed always gives the same arena and spawn.
    observation, info = env.reset(seed=seed)
    # Clear any state the driver kept from a previous episode.
    driver.reset()

    # Running totals: distance driven so far and the number of control steps
    # in which any contact flag was set. x, y is where the car was one step ago.
    path_length = 0.0
    contact_steps = 0
    x, y, _ = info["pose"]

    # The standard loop: act, then step, until the episode ends.
    while True:
        # The driver sees only the observation (never info or the env).
        action = driver.act(observation)
        # terminated: the car reached the target. truncated: time ran out.
        # info holds ground truth (pose, time, success), which tools may read.
        observation, _, terminated, truncated, info = env.step(action)

        # Add the straight-line distance moved this step to the path length.
        path_length += math.hypot(info["pose"][0] - x, info["pose"][1] - y)
        # observation['contact'] is [left, front, right]; count steps with any touch.
        if observation["contact"].any():
            contact_steps += 1

        # Remember this position for the next step's distance.
        x, y, _ = info["pose"]

        # Stop at success or timeout.
        if terminated or truncated:
            break

    # Why the episode ended, from the two flags (not from info['extras'], which is
    # unstable debug data). 'stopped' should never happen; it is a safety net.
    if terminated:
        reason = "success"
    elif truncated:
        reason = "timeout"
    else:
        reason = "stopped"

    # One table row. time_to_target is NaN for failures so that a median over the
    # column ignores them; duration_s is how long the episode lasted either way.
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

# Run every combination of scenario, driver and seed and return one table.
# workers=1 runs in this process; more than 1 uses a pool of worker processes.
def run_batch(scenarios, drivers, seeds, workers=1) -> pd.DataFrame:
    # One job per (scenario, driver, seed). The last loop varies fastest, so rows
    # come out grouped by scenario, then driver, then seed.
    jobs = [
        (scenario, driver, seed)
        for scenario in scenarios
        for driver in drivers
        for seed in seeds
    ]
    # Serial path: a plain loop, so errors show ordinary tracebacks.
    if workers == 1:
        results = [one_job(job) for job in jobs]
    # Parallel path.
    else:
        # 'spawn' starts clean processes. Chrono objects cannot be shared between
        # processes, so each worker builds its own environment in one_job().
        with get_context("spawn").Pool(processes=workers) as pool:
        # map keeps results in job order. chunksize=1 hands out one job at a time,
        # which balances the load when some episodes (timeouts) are much slower.
            results = pool.map(one_job, jobs, chunksize=1)
    # A list of row dicts becomes a table: one row per episode.
    return pd.DataFrame(results)
  

# Write the episode table to <out_dir>/results.csv and return that path.
def write_table(table, out_dir):
  # Accept a string or a Path.
  out_dir = Path(out_dir)
  # Create the folder (and parents) if needed; fine if it already exists.
  out_dir.mkdir(parents=True, exist_ok=True)
  # The file name is fixed.
  path = out_dir / "results.csv"
  # index=False keeps pandas' row numbers out of the file.
  table.to_csv(path, index=False)
  # Return the path so the caller can say where the file went.
  return path

# One row per (scenario, driver): episode count, success rate with a 95 %
# confidence interval, median time to target, and a few averages.
def summarize(table):
    # One summary row is collected per group.
    rows = []
    # Split the table into one sub-table per (scenario, driver) pair.
    for (scenario, driver), group in table.groupby(["scenario", "driver"], sort=True):
        # Number of episodes in this group.
        n = len(group)
        # A scenario with no target lamp has NaN target_distance in every episode;
        # 'success' means nothing there, so such groups get no success rate.
        has_target = group["target_distance"].notna().any()

        # Numbers that make sense for every group, with or without a target.
        row = {
            "scenario": scenario,
            "driver": driver,
            "episodes": n,
            "has_target": has_target,
            "mean_contact_steps": group["contact_steps"].mean(),
            "mean_path_length": group["path_length"].mean(),
        }
        # Scenario with a target: success statistics.
        if has_target:
            # True counts as 1, so the sum of the column is the number of successes.
            successes = int(group["success"].sum())
            # 95 % Wilson interval for the success rate.
            low, high = wilson_interval(successes, n)
            # Median time to target over successful episodes only (NaNs are skipped).
            row.update(
                successes=successes,
                success_rate=successes / n,
                ci_low=low,
                ci_high=high,
                median_time_to_target=group["time_to_target"].median(),
            )
        # No target: success statistics do not apply, so they are NaN.
        else:
            # NaN means 'not a number', used here for 'no value'.
            nan = float("nan")
            row.update(successes=nan, success_rate=nan, ci_low=nan, ci_high=nan,
                       median_time_to_target=nan)
        # Keep this group's row.
        rows.append(row)
    # Turn the list of row dicts into a table.
    return pd.DataFrame(rows)


# Print one line per (scenario, driver) from the table made by summarize().
def print_summary(summary):
    # iterrows() yields (row number, row); the number is not needed.
    for _, r in summary.iterrows():
    # Prefix for the line, e.g. 'light_seek / braitenberg'.
        label = f"{r['scenario']} / {r['driver']}"
        # No target: say so, instead of printing a misleading 0 %.
        if not r["has_target"]:
            print(f"{label}: no target in this scenario, success rate not applicable "
                  f"({int(r['episodes'])} episodes)")
            continue
        # The median is NaN when nobody reached the target.
        median = r["median_time_to_target"]
        median_text = "n/a" if pd.isna(median) else f"{median:.2f}s"
        # e.g. '45/50 reached the target (90.00%, 95% CI: [78.64%, 95.65%]), ...'
        print(f"{label}: {int(r['successes'])}/{int(r['episodes'])} reached the target "
              f"({r['success_rate']:.2%}, 95% CI: [{r['ci_low']:.2%}, {r['ci_high']:.2%}]), "
              f"median time to target: {median_text}")


# Success rates as a grid: one row per driver, one column per scenario.
def comparison_table(summary):
    return summary.pivot(index="driver", columns="scenario", values="success_rate")

# 95 % Wilson confidence interval for a success rate (successes out of total).
# Unlike the textbook p +/- 1.96*sqrt(p(1-p)/n), it stays sensible near 0 % and
# 100 % and with few episodes. Returns (lower, upper) as fractions in [0, 1].
def wilson_interval(succeses, total):
  # No episodes, no interval.
  if total == 0:
    return float("nan"), float("nan")

  # Observed success rate.
  p = succeses / total
  # 1.96 is the z-value for 95 % confidence.
  z = 1.96
  # Shared denominator of both bounds: 1 + z^2/n.
  denominator = 1 + (z**2 / total)
  # The interval is centred on the rate pulled slightly towards 1/2.
  centre_adjusted_probability = p + (z**2 / (2 * total))
  # Spread of the interval (the adjusted standard deviation).
  adjusted_standard_deviation = math.sqrt((p * (1 - p) + (z**2 / (4 * total))) / total)
  # Lower and upper bounds.
  lower_bound = (centre_adjusted_probability - z * adjusted_standard_deviation) / denominator
  upper_bound = (centre_adjusted_probability + z * adjusted_standard_deviation) / denominator
  return lower_bound, upper_bound

# What one worker runs: ONE episode, described by plain data so that it can be
# sent to another process. Builds its own env and driver (Chrono objects cannot
# be shared between processes), runs run_one, and adds scenario and driver labels.
def one_job(job_specs):
  # Defined up front so the finally block can tell whether the env was created.
  env = None
  # try/finally guarantees the env is closed even if the episode crashes.
  try:
    # A job is a (scenario file path, driver name, seed) tuple.
    scenario_path, driver_name, seed = job_specs
    # Build a fresh environment for this episode (CPU 'raycast' eyes).
    env = FlyCarEnv(scenario_path, eye_backend="raycast")
    # Build the driver from its name (e.g. 'braitenberg').
    driver = make_driver(driver_name)

    # The episode itself.
    res = run_one(env, driver, seed)
  # Runs whether or not the episode succeeded.
  finally:
    # Only close the env if it was actually created.
    if env:
      env.close()

  # Labels first, then the measurements. The scenario column is the file name
  # without folder or extension, e.g. 'light_seek'.
  return {"scenario": Path(scenario_path).stem, "driver": driver_name, **res}

# Turn text like '0-49', '7' or '0,2-4,6' into a list of seeds.
# Both ends of a range are included, so '0-49' is 50 seeds.
def parse_seeds(seeds_str):
  # The seeds collected so far.
  seeds = []
  # Comma-separated parts are handled one at a time.
  for part in seeds_str.split(","):
    # A dash means a range such as '2-4'.
    if "-" in part:
      # Split at the dash and convert both ends to integers.
      start, end = map(int, part.split("-"))
      # A backwards range such as '5-3' is a mistake in the input.
      if (start > end):
        raise ValueError(f"Invalid seed range: {part}")
      # range() excludes its end, hence + 1.
      seeds.extend(range(start, end + 1))
    # Otherwise the part is a single seed.
    else:
      seeds.append(int(part))
  return seeds

# Command-line entry point: read the options, run the batch, report.
def main():
  # Describe the options for --help.
  parser = argparse.ArgumentParser(description="Run a batch of FlyCar episodes.")
  # --scenario may name several files, separated by commas.
  parser.add_argument("--scenario", required=True, help="Scenario YAML file, e.g. scenarios/light_seek.yaml")
  # --driver may name several drivers, separated by commas.
  parser.add_argument("--driver", default="braitenberg",
                          help="random, braitenberg, keyboard, or package.module:ClassName (default: braitenberg)")
  # Seeds as text, parsed by parse_seeds(), e.g. 0-49.
  parser.add_argument("--seeds", default="0-9", help="Seeds to run")
  # How many processes to use; 1 runs everything in this process.
  parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers (default: 1)")
  # Optional folder for results.csv. Nothing is written without it.
  parser.add_argument("--out", help="Directory to write results")
  parser.add_argument("--print", help="Print the results table to stdout", action="store_true")

  # Read the command line.
  args = parser.parse_args()
  # Text like '0-49' becomes a list of integers.
  seeds = parse_seeds(args.seeds)
  # Split the comma-separated lists, trimming spaces and ignoring empty pieces.
  scenarios = [s.strip() for s in args.scenario.split(",") if s.strip()]
  drivers = [d.strip() for d in args.driver.split(",") if d.strip()]

  # Run all (scenario, driver, seed) combinations and collect the table.
  table = run_batch(scenarios, drivers, seeds, workers=args.workers)

  # Only write the CSV if --out was given.
  if args.out:
    path = write_table(table, args.out)
    print(f"\nResults written to {path}")

  # Blank line to separate the sections of output.
  print()

  # One row per (scenario, driver).
  summary = summarize(table)
  # One printed line per row.
  print_summary(summary)

  # Print the results table to stdout if --print is specified.
  if args.print:
    print("\nResults table:")
    print(table.to_string(float_format=lambda v: f"{v:.2f}"))

  print()

  # With several drivers or scenarios, also show the success-rate grid.
  if len(scenarios) > 1 or len(drivers) > 1:
    print(comparison_table(summary).to_string(float_format=lambda v: f"{v:.2f}"))
    

# Run main() only when this file is run directly (python -m flycar.tools.batch).
# This guard is required: worker processes started with 'spawn' re-import this
# file, and without the guard each worker would try to start a batch of its own.
if __name__ == "__main__":
  sys.exit(main())