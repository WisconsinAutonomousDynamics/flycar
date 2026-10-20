import math
from pathlib import Path

from flycar.env.core import FlyCarEnv
from flycar.drivers import make_driver
import pandas as pd


def run_one(env, driver, seed):
  observation, info = env.reset(seed=seed)
  driver.reset()

  path_length = 0
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

  return {"seed": seed, "success": info["success"], "path_length": path_length, "contact_steps": contact_steps, "time": info["time"], "time_to_target": info["time"] if info["success"] else float("nan"), "target_distance": info["target_distance"]}

def run_batch(env, driver, seeds):
  results = []
  for seed in seeds:
    res = run_one(env, driver, seed)
    results.append(res)
  return pd.DataFrame(results)

def write_table(table, out_dir):
  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  path = out_dir / "results.csv"
  table.to_csv(path, index=False)
  return path

def summary(table: pd.DataFrame):
  succsesses = table["success"].sum()
  n = len(table)
  success_rate = table["success"].mean()

  lower_bound, upper_bound = wilson_interval(succsesses, n)

  median_time = table["time_to_target"].median()
  print(f"{succsesses}/{n} reached the target ({success_rate:.2%}, 95% CI: [{lower_bound:.2%}, {upper_bound:.2%}]), median time to target: {median_time:.2f}s")

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

if __name__ == "__main__":
  env = FlyCarEnv("scenarios/light_seek.yaml", eye_backend="raycast")
  driver = make_driver("braitenberg")

  seeds = list(range(10))
  res = run_batch(env, driver, seeds)
  path = write_table(res, Path("outputs"))

  summary(res)
  print(pd.read_csv(path))



