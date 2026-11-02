"""Benchmark fixed-pose CPU eyes: python -m scripts.benchmark_eyes."""

import platform
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pychrono as chrono
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from flycar.env.core import FlyCarEnv
from scripts.check_env import chrono_version


def main():
    scenario = REPO_ROOT / "scenarios" / "light_seek.yaml"
    warmup_renders = 20
    measured_renders = 200
    results = []

    for seed in (0, 1, 2):
        env = FlyCarEnv(scenario, eye_backend="raycast")
        try:
            env.reset(seed=seed)
            assert (env.eyes.rows, env.eyes.cols) == (20, 20)
            for _ in range(warmup_renders):
                env.eyes.render()

            samples = []
            for _ in range(measured_renders):
                started = time.perf_counter()
                env.eyes.render()
                wall_ms = (time.perf_counter() - started) * 1000
                sample = env.eyes.extra_info()
                sample["wall_ms"] = wall_ms
                samples.append(sample)

            metrics = {}
            for key in samples[0]:
                values = [sample[key] for sample in samples]
                if key.endswith("_ms"):
                    metrics[key] = {
                        "median": float(np.median(values)),
                        "p95": float(np.percentile(values, 95)),
                    }
                else:
                    assert all(value == values[0] for value in values), key
                    metrics[key] = values[0]
            results.append({"seed": seed, "metrics": metrics})
        finally:
            env.close()

    return {"eyes_benchmark": {
        "scenario": "scenarios/light_seek.yaml",
        "backend": "raycast",
        "resolution": [20, 20],
        "measurement": "Repeated renders at each seed's reset pose.",
        "warmup_renders": warmup_renders,
        "measured_renders": measured_renders,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "chrono": chrono_version(chrono),
        },
        "results": results,
    }}


if __name__ == "__main__":
    with redirect_stdout(sys.stderr):
        report = main()
    print(yaml.safe_dump(report, sort_keys=False), end="")
