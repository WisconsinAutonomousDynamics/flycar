"""Record the behaviour baseline in tests/baseline/.

    python scripts/record_baseline.py

The baseline is a handful of reference episodes, stored as small ``.npz``
files. ``tests/test_baseline.py`` reruns the same episodes and checks that
nothing changed. It is how we enforce "new features are opt-in, defaults are
unchanged": if a pull request changes the trajectory of a default scenario, the
baseline test fails.

**Do not re-record to make a failing baseline test pass.** A failure means the
code now behaves differently. Find out why. Only the project leads re-record,
and only when a behaviour change was agreed on in an ``api-change`` issue.

What is stored, every ``RECORD_EVERY``-th control step (step 0 is the state
right after ``reset()``):

    step              control step number
    pose              info["pose"]            [x, y, yaw]
    applied_action    info["applied_action"]  [throttle, steering, brake]
    imu               obs["imu"]
    contact           obs["contact"]
    eye_left_sum      checksums of each eye image (see eye_checksums below)
    eye_left_wsum
    eye_right_sum
    eye_right_wsum

and, once per episode: the number of steps, and whether it ended terminated,
truncated or with success.
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))   # so "import flycar" works from anywhere

BASELINE_DIR = REPO_ROOT / "tests" / "baseline"

# Store one sample every this many control steps.
RECORD_EVERY = 10

# (file name, scenario file, driver name, seed). Always headless and raycast.
BASELINE_RUNS = [
    ("light_seek_braitenberg_seed0", "scenarios/light_seek.yaml", "braitenberg", 0),
    ("light_seek_braitenberg_seed1", "scenarios/light_seek.yaml", "braitenberg", 1),
    ("light_seek_braitenberg_seed2", "scenarios/light_seek.yaml", "braitenberg", 2),
    ("empty_arena_random_seed0", "scenarios/empty_arena.yaml", "random", 0),
]


def eye_checksums(image):
    """Two numbers that change if any pixel of an eye image changes.

    ``sum`` is the total brightness. ``wsum`` weights every pixel by its
    position (1, 2, 3, ... row by row), so that two pixels swapping values also
    shows up. Floats rather than a hash, so the test can allow for the last
    digit of rounding.
    """
    pixels = np.asarray(image, dtype=np.float64).ravel()
    weights = np.arange(1, len(pixels) + 1, dtype=np.float64)
    return float(pixels.sum()), float(np.dot(pixels, weights))


def run_episode(scenario_path, driver_name, seed):
    """Run one full episode headless with raycast eyes. Returns a dict of arrays."""
    from flycar.drivers import make_driver
    from flycar.env.core import FlyCarEnv

    env = FlyCarEnv(REPO_ROOT / scenario_path, render_mode=None, eye_backend="raycast")
    driver = make_driver(driver_name)

    samples = {
        "step": [], "pose": [], "applied_action": [], "imu": [], "contact": [],
        "eye_left_sum": [], "eye_left_wsum": [], "eye_right_sum": [], "eye_right_wsum": [],
    }

    def record(observation, info):
        samples["step"].append(int(info["step"]))
        samples["pose"].append(np.array(info["pose"], dtype=np.float64))
        samples["applied_action"].append(np.array(info["applied_action"], dtype=np.float64))
        samples["imu"].append(np.array(observation["imu"], dtype=np.float64))
        samples["contact"].append(np.array(observation["contact"], dtype=np.float64))
        left_sum, left_wsum = eye_checksums(observation["eye_left"])
        right_sum, right_wsum = eye_checksums(observation["eye_right"])
        samples["eye_left_sum"].append(left_sum)
        samples["eye_left_wsum"].append(left_wsum)
        samples["eye_right_sum"].append(right_sum)
        samples["eye_right_wsum"].append(right_wsum)

    observation, info = env.reset(seed=seed)
    driver.reset()
    record(observation, info)

    while True:
        action = driver.act(observation)
        observation, _, terminated, truncated, info = env.step(action)
        if info["step"] % RECORD_EVERY == 0:
            record(observation, info)
        if terminated or truncated:
            break
    env.close()

    result = {name: np.array(values) for name, values in samples.items()}
    result["n_steps"] = np.array(int(info["step"]))
    result["terminated"] = np.array(bool(terminated))
    result["truncated"] = np.array(bool(truncated))
    result["success"] = np.array(bool(info["success"]))
    return result


def main():
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    for name, scenario_path, driver_name, seed in BASELINE_RUNS:
        result = run_episode(scenario_path, driver_name, seed)
        path = BASELINE_DIR / f"{name}.npz"
        np.savez_compressed(path, **result)
        print(f"{path.relative_to(REPO_ROOT)}: {int(result['n_steps'])} steps, "
              f"success={bool(result['success'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
