"""Run a driver in a scenario.

    python run.py --scenario scenarios/light_seek.yaml --driver braitenberg

Useful extras:

    --episodes 20        run 20 episodes (seeds 0..19) and print the success rate
    --seed 7             start from seed 7 instead of 0
    --log                write one Parquet log per episode into logs/
    --dashboard          live matplotlib view (map, eyes, control traces)
    --render             3D Irrlicht window that follows the car
    --eyes raycast       force an eye backend (auto, sensor or raycast)
    --realtime           do not run faster than real time

Drivers: random, braitenberg, keyboard, or your own as package.module:ClassName.

This script is also the reference for how to use the environment: it is the
standard reset / act / step loop and nothing more.
"""

import argparse
import sys
import time
from pathlib import Path

from flycar.drivers import make_driver
from flycar.env.core import FlyCarEnv


def run_episode(env, driver, seed, args):
    """Run one episode. Returns the final ``info`` dict, or None if a window was closed."""
    observation, info = env.reset(seed=seed)
    driver.reset()

    logger = None
    if args.log is not None:
        from flycar.tools.logger import EpisodeLogger

        name = f"{env.scenario['name']}_{args.driver.split(':')[-1]}_seed{seed}.parquet"
        details = {"scenario": env.scenario["name"], "driver": args.driver, "seed": seed}
        logger = EpisodeLogger(Path(args.log) / name, info["world"], details)

    dashboard = None
    if args.dashboard:
        from flycar.tools.dashboard import Dashboard

        dashboard = Dashboard(
            info["world"],
            observation["eye_left"].shape,
            show_cameras=(info["eye_backend"] == "sensor"),
            driver=driver,
        )

    wall_start = time.perf_counter()
    stopped_by_user = False

    while True:
        # ---- The whole public API in three lines ----------------------------
        action = driver.act(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        # ---------------------------------------------------------------------

        if logger is not None:
            logger.log(action, observation, info)
        if dashboard is not None and not dashboard.update(observation, info):
            stopped_by_user = True
        if env.window_closed:
            stopped_by_user = True

        if args.realtime:
            # Sleep until the wall clock catches up with the simulation clock.
            ahead = info["time"] - (time.perf_counter() - wall_start)
            if ahead > 0:
                time.sleep(ahead)

        if done or stopped_by_user:
            break

    if logger is not None:
        path = logger.close()
        print(f"  log written to {path}")
    if dashboard is not None:
        dashboard.close()

    return None if stopped_by_user else info


def main():
    parser = argparse.ArgumentParser(description="Run a Fly Car driver in a scenario.")
    parser.add_argument("--scenario", required=True, help="Scenario YAML file, e.g. scenarios/light_seek.yaml")
    parser.add_argument("--driver", default="braitenberg",
                        help="random, braitenberg, keyboard, or package.module:ClassName (default: braitenberg)")
    parser.add_argument("--seed", type=int, default=0, help="Seed of the first episode (default: 0)")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes; seeds count up from --seed")
    parser.add_argument("--log", nargs="?", const="logs", default=None, metavar="DIR",
                        help="Write a Parquet log per episode into DIR (default DIR: logs)")
    parser.add_argument("--dashboard", action="store_true", help="Show the live matplotlib dashboard")
    parser.add_argument("--render", action="store_true", help="Show the 3D Irrlicht window")
    parser.add_argument("--eyes", choices=["auto", "sensor", "raycast"], default=None,
                        help="Override the scenario's eye backend")
    parser.add_argument("--realtime", action="store_true", help="Slow the run down to real time")
    args = parser.parse_args()

    driver = make_driver(args.driver)

    # A person at the keyboard needs a window to type into, and real-time speed.
    if getattr(driver, "needs_key_events", False):
        if not args.dashboard:
            print("The keyboard driver reads keys from the dashboard window: turning on --dashboard.")
            args.dashboard = True
        args.realtime = True

    env = FlyCarEnv(args.scenario, render_mode="human" if args.render else None, eye_backend=args.eyes)
    has_target = len([light for light in env.scenario["lights"] if light["target"]]) > 0

    successes = 0
    finished = 0
    for episode in range(args.episodes):
        seed = args.seed + episode
        info = run_episode(env, driver, seed, args)
        if info is None:
            print("Window closed, stopping.")
            break

        finished += 1
        successes += int(info["success"])
        if has_target:
            outcome = "reached the target" if info["success"] else "did NOT reach the target"
            print(f"seed {seed}: {outcome} after {info['time']:.1f} s "
                  f"(final distance {info['target_distance']:.1f} m)")
        else:
            x, y, _ = info["pose"]
            print(f"seed {seed}: ran {info['time']:.1f} s, ended at x = {x:.1f} m, y = {y:.1f} m")

    env.close()

    if has_target and finished > 1:
        print(f"Reached the target in {successes} of {finished} episodes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
