"""Tools & drivers team: logger, replay, dashboard, run.py, and the drivers."""

import ast
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from conftest import REPO_ROOT, small_scenario
from flycar.drivers import BUILTIN_DRIVERS, make_driver
from flycar.drivers.base import Driver
from flycar.env import spaces
from flycar.tools.logger import EpisodeLogger, load_log


def _run_and_log(make_env, path, steps=60):
    scenario = small_scenario(lights=[{"x": 8.0, "y": 3.0, "target": True}])
    env = make_env(scenario)
    driver = make_driver("braitenberg")
    observation, info = env.reset(seed=0)
    driver.reset()
    logger = EpisodeLogger(path, info["world"], {"scenario": "test", "driver": "braitenberg", "seed": 0})
    for _ in range(steps):
        action = driver.act(observation)
        observation, _, _, _, info = env.step(action)
        logger.log(action, observation, info)
    return logger.close(), info


def test_logger_round_trip(make_env, tmp_path):
    path, last_info = _run_and_log(make_env, tmp_path / "logs" / "run.parquet")
    assert path.exists()

    table, world, details = load_log(path)
    assert len(table) == 60
    np.testing.assert_allclose(table["time"], np.arange(1, 61) * 0.01)

    # Action, observation summary and ground-truth pose are all there.
    for column in ("action_throttle", "action_steering", "action_brake",
                   "applied_throttle", "applied_steering", "applied_brake",
                   "x", "y", "yaw", "speed", "yaw_rate",
                   "contact_left", "contact_front", "contact_right",
                   "eye_left_mean", "eye_right_mean", "target_distance", "success"):
        assert column in table.columns, column
    assert table["x"].iloc[-1] == last_info["pose"][0]

    # Full eye frames are stored every 10th step, and can be un-flattened.
    stored = table[table["eye_left"].notna()]
    assert list(stored["step"]) == [10, 20, 30, 40, 50, 60]
    assert np.asarray(stored["eye_left"].iloc[0]).reshape(details["eye_shape"]).shape == (20, 20)

    # The metadata is enough to redraw the arena.
    assert world["target"] == [8.0, 3.0]
    assert details["driver"] == "braitenberg"
    assert details["eye_backend"] == "raycast"

    # And it is an ordinary Parquet file that plain pandas can read.
    assert len(pd.read_parquet(path)) == 60


def test_replay_writes_a_figure(make_env, tmp_path):
    from flycar.tools.replay import make_figure

    path, _ = _run_and_log(make_env, tmp_path / "run.parquet")
    table, world, details = load_log(path)
    figure = make_figure(table, world, details, eye_times=[0.1, 0.5])
    output = tmp_path / "replay.png"
    figure.savefig(output)
    assert output.stat().st_size > 10_000


def test_dashboard_runs_headless_and_forwards_keys(make_env):
    from flycar.tools.dashboard import Dashboard

    env = make_env(small_scenario())
    driver = make_driver("keyboard")
    observation, info = env.reset(seed=0)
    dashboard = Dashboard(info["world"], observation["eye_left"].shape, driver=driver, refresh_every=2)

    # A key press in the window reaches the driver.
    from matplotlib.backend_bases import KeyEvent

    canvas = dashboard.figure.canvas
    canvas.callbacks.process("key_press_event", KeyEvent("key_press_event", canvas, "w"))
    assert "w" in driver.keys_down

    for _ in range(10):
        observation, _, _, _, info = env.step(driver.act(observation))
        assert dashboard.update(observation, info)
    assert info["pose"][0] > 0.0          # holding "w" moved the car forwards

    dashboard.close()
    assert dashboard.update(observation, info) is False


def test_logger_leaves_out_extras_unless_asked(make_env, tmp_path):
    env = make_env(small_scenario())
    observation, info = env.reset(seed=0)
    plain = EpisodeLogger(tmp_path / "plain.parquet", info["world"])
    with_extras = EpisodeLogger(tmp_path / "extras.parquet", info["world"], log_extras=True)
    for _ in range(5):
        action = np.array([0.5, 0.0, 0.0], dtype=np.float32)
        observation, _, _, _, info = env.step(action)
        info["extras"] = {"core": {"contact": {"force": np.float32(1.5), "points": np.zeros(2)}}}
        plain.log(action, observation, info)
        with_extras.log(action, observation, info)

    table, _, _ = load_log(plain.close())
    assert "extras" not in table.columns
    table, _, _ = load_log(with_extras.close())
    import json
    assert json.loads(table["extras"].iloc[0]) == {"core": {"contact": {"force": 1.5, "points": [0.0, 0.0]}}}


# ----------------------------------------------------------------------------
# run.py
# ----------------------------------------------------------------------------


def _run(arguments, timeout=300):
    return subprocess.run(
        [sys.executable, *[str(a) for a in arguments]],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout,
    )


@pytest.mark.slow
def test_run_py_with_random_driver_and_logging(tmp_path):
    result = _run([
        REPO_ROOT / "run.py",
        "--scenario", REPO_ROOT / "scenarios" / "empty_arena.yaml",
        "--driver", "random", "--eyes", "raycast", "--log", tmp_path,
    ])
    assert result.returncode == 0, result.stdout + result.stderr
    assert "eye backend: raycast" in result.stdout
    assert (tmp_path / "empty_arena_random_seed0.parquet").exists()


@pytest.mark.slow
def test_braitenberg_reaches_the_light_in_most_seeds():
    """Acceptance criterion 3, on the first five seeds."""
    result = _run([
        REPO_ROOT / "run.py",
        "--scenario", REPO_ROOT / "scenarios" / "light_seek.yaml",
        "--driver", "braitenberg", "--eyes", "raycast", "--episodes", "5",
    ], timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    successes = result.stdout.count("reached the target after")
    assert successes >= 4, result.stdout


# ----------------------------------------------------------------------------
# Drivers: they must stay independent of the simulator
# ----------------------------------------------------------------------------


DRIVER_DIR = REPO_ROOT / "flycar" / "drivers"


DRIVER_FILES = sorted(DRIVER_DIR.glob("*.py"))


ALLOWED_FLYCAR_IMPORTS = ("flycar.drivers", "flycar.env.spaces")


def _imported_modules(path):
    """Every module name imported anywhere in a Python file (found by parsing it)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


@pytest.mark.parametrize("path", DRIVER_FILES, ids=lambda p: p.name)
def test_drivers_do_not_import_pychrono(path):
    """Static check: read the source, look at the import statements."""
    for name in _imported_modules(path):
        assert not name.startswith("pychrono"), f"{path.name} imports {name}"
        if name.startswith("flycar"):
            assert name.startswith(ALLOWED_FLYCAR_IMPORTS), (
                f"{path.name} imports {name}; drivers may only use {ALLOWED_FLYCAR_IMPORTS}"
            )


def test_loading_every_driver_does_not_load_pychrono():
    """Run-time check, in a fresh Python process so earlier tests cannot interfere."""
    code = (
        "import sys\n"
        "from flycar.drivers import BUILTIN_DRIVERS, make_driver\n"
        "for name in BUILTIN_DRIVERS:\n"
        "    make_driver(name)\n"
        "loaded = [m for m in sys.modules if m.startswith('pychrono')]\n"
        "assert not loaded, loaded\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("name", sorted(BUILTIN_DRIVERS))
def test_builtin_drivers_follow_the_protocol(name):
    driver = make_driver(name)
    assert isinstance(driver, Driver)

    observation_space = spaces.make_observation_space()
    action_space = spaces.make_action_space()
    observation_space.seed(0)

    driver.reset()
    for _ in range(20):
        observation = observation_space.sample()
        observation["time"] = np.array([0.5], dtype=np.float32)
        action = np.asarray(driver.act(observation), dtype=np.float32)
        assert action.shape == (3,)
        assert action_space.contains(action)


def test_make_driver_accepts_a_module_path_and_rejects_nonsense():
    driver = make_driver("flycar.drivers.braitenberg:BraitenbergDriver")
    assert isinstance(driver, Driver)
    with pytest.raises(ValueError, match="Unknown driver"):
        make_driver("no_such_driver")


def _observation_with_eyes(left, right):
    return {
        "eye_left": left.astype(np.float32),
        "eye_right": right.astype(np.float32),
        "imu": np.zeros(5, dtype=np.float32),
        "contact": np.zeros(3, dtype=np.float32),
        "time": np.zeros(1, dtype=np.float32),
    }


def test_braitenberg_steers_towards_the_brighter_eye():
    driver = make_driver("braitenberg")
    dark = np.full((20, 20), 0.05)
    lamp = dark.copy()
    lamp[9:11, 9:11] = 0.9   # a small bright spot

    steer_left = driver.act(_observation_with_eyes(lamp, dark))[spaces.STEERING]
    steer_right = driver.act(_observation_with_eyes(dark, lamp))[spaces.STEERING]
    straight = driver.act(_observation_with_eyes(lamp, lamp))[spaces.STEERING]
    assert steer_left > 0.2      # positive = left
    assert steer_right < -0.2
    assert straight == pytest.approx(0.0)


def test_braitenberg_prefers_a_small_bright_lamp_to_a_large_dim_wall():
    driver = make_driver("braitenberg")
    wall = np.full((20, 20), 0.05)
    wall[5:15, :] = 0.25                 # half the left eye filled with a dimly lit wall
    lamp = np.full((20, 20), 0.05)
    lamp[9:12, 9:12] = 0.9               # a small lamp in the right eye
    steering = driver.act(_observation_with_eyes(wall, lamp))[spaces.STEERING]
    assert steering < 0.0                # towards the lamp, not the wall


def test_keyboard_driver_maps_keys_to_actions():
    driver = make_driver("keyboard")
    observation = _observation_with_eyes(np.zeros((20, 20)), np.zeros((20, 20)))
    np.testing.assert_array_equal(driver.act(observation), [0.0, 0.0, 0.0])

    driver.on_key_press("w")
    driver.on_key_press("left")
    np.testing.assert_array_equal(driver.act(observation), [1.0, 1.0, 0.0])

    driver.on_key_release("w")
    driver.on_key_release("left")
    driver.on_key_press("d")
    driver.on_key_press("down")
    np.testing.assert_array_equal(driver.act(observation), [0.0, -1.0, 1.0])

    driver.reset()
    np.testing.assert_array_equal(driver.act(observation), [0.0, 0.0, 0.0])


def test_random_driver_is_reproducible():
    driver = make_driver("random")
    observation = _observation_with_eyes(np.zeros((20, 20)), np.zeros((20, 20)))

    def episode():
        driver.reset()
        actions = []
        for step in range(200):
            observation["time"] = np.array([step * 0.01], dtype=np.float32)
            actions.append(driver.act(observation).copy())
        return np.array(actions)

    first = episode()
    second = episode()
    np.testing.assert_array_equal(first, second)
    assert len(np.unique(first[:, spaces.STEERING])) > 1   # it does change its mind
