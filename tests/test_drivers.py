"""Drivers: they must stay independent of the simulator."""

import ast
import subprocess
import sys

import numpy as np
import pytest

from conftest import REPO_ROOT
from flycar.drivers import BUILTIN_DRIVERS, make_driver
from flycar.drivers.base import Driver
from flycar.env import spaces

DRIVER_DIR = REPO_ROOT / "flycar" / "drivers"
DRIVER_FILES = sorted(DRIVER_DIR.glob("*.py"))

# The only parts of flycar a driver may import: other drivers, and the spaces.
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
