
"""Benchmark FlyCar core performance with 20x20 CPU raycast eyes."""

import json
import time
from collections import defaultdict
from pathlib import Path

from flycar.env.scenario import load_scenario
from flycar.env.core import FlyCarEnv

EPISODES = 10
SCENARIO = "scenarios/light_seek.yaml"
OUTPUT = Path("benchmark_core_results.json")

scenario = load_scenario(SCENARIO)
scenario["timing"]["profile"] = True
scenario["eyes"]["resolution"] = [20, 20]

env = FlyCarEnv(scenario, eye_backend="raycast")

totals = defaultdict(float)
total_wall = 0.0
total_simulated = 0.0
episode_results = []

max_steps = round(
    env.scenario["episode"]["max_time_s"] / env.control_period
)

print(f"Eye backend: {env.eye_backend}")
print(f"Eye resolution: {env.scenario['eyes']['resolution']}")
print(f"Episodes: {EPISODES}")
print(f"Maximum steps per episode: {max_steps}")

try:
    for episode in range(EPISODES):
        env.reset(seed=episode)

        start = time.perf_counter()
        steps = 0

        for _ in range(max_steps):
            _, _, terminated, truncated, info = env.step(
                [0.3, 0.0, 0.0]
            )
            steps += 1

            if terminated or truncated:
                break

        wall = time.perf_counter() - start
        simulated = float(info["time"])
        profile = info["extras"]["core"]["env"]

        for phase, seconds in profile["timings_s"].items():
            totals[phase] += seconds

        total_wall += wall
        total_simulated += simulated

        result = {
            "episode": episode + 1,
            "seed": episode,
            "steps": steps,
            "simulated_s": simulated,
            "wall_s": wall,
            "rtf": simulated / wall,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "timings_s": dict(profile["timings_s"]),
            "calls": dict(profile["calls"]),
        }
        episode_results.append(result)

        print(
            f"Episode {episode + 1:2d}: "
            f"sim={simulated:.2f}s, "
            f"wall={wall:.2f}s, "
            f"RTF={simulated / wall:.2f}x",
            flush=True,
        )

finally:
    env.close()

summary = {
    "scenario": SCENARIO,
    "eye_backend": "raycast",
    "eye_resolution": [20, 20],
    "episodes": EPISODES,
    "action": [0.3, 0.0, 0.0],
    "simulated_s": total_simulated,
    "wall_s": total_wall,
    "rtf": total_simulated / total_wall,
    "timings_s": dict(totals),
    "episode_results": episode_results,
}

OUTPUT.write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8",
)

print("\n=== A1 PROFILING RESULTS ===")
print(f"Episodes: {EPISODES}")
print(f"Simulated time: {total_simulated:.3f}s")
print(f"Wall time: {total_wall:.3f}s")
print(f"Real-time factor: {total_simulated / total_wall:.3f}x")

print("\nPhase breakdown:")
for phase, seconds in sorted(
    totals.items(), key=lambda item: item[1], reverse=True
):
    print(
        f"{phase:24s} "
        f"{seconds:9.4f}s "
        f"{100 * seconds / total_wall:6.2f}%"
    )

print("\nNote: Nested timing categories may overlap.")
print(f"Results saved to: {OUTPUT}")
