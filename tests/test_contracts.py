"""The frozen contracts (docs/CONTRACTS.md). Owned by the project leads.

Changing anything tested here needs a lead-approved ``api-change`` issue.
"""

import re
import subprocess
import sys

import numpy as np
import pytest

from conftest import REPO_ROOT, small_scenario
from flycar.env import spaces
from flycar.env.scenario import ScenarioError, load_scenario, sample


# ----------------------------------------------------------------------------
# spaces.py and docs/API.md
# ----------------------------------------------------------------------------


API_DOC = REPO_ROOT / "docs" / "API.md"


def test_action_space_matches_the_brief():
    space = spaces.make_action_space()
    assert space.shape == (3,)
    assert space.dtype == np.float32
    np.testing.assert_array_equal(space.low, [0.0, -1.0, 0.0])
    np.testing.assert_array_equal(space.high, [1.0, 1.0, 1.0])
    assert [c.name for c in spaces.ACTION_CHANNELS] == ["throttle", "steering", "brake"]


def test_observation_space_keys_shapes_and_dtypes():
    space = spaces.make_observation_space()
    assert set(space.spaces) == {"eye_left", "eye_right", "imu", "contact", "time"}
    assert space["eye_left"].shape == (20, 20)
    assert space["eye_right"].shape == (20, 20)
    assert space["imu"].shape == (5,)
    assert space["contact"].shape == (3,)
    assert space["time"].shape == (1,)
    for key, box in space.spaces.items():
        assert box.dtype == np.float32, key


def test_eye_shape_is_configurable():
    space = spaces.make_observation_space((8, 12))
    assert space["eye_left"].shape == (8, 12)


def test_api_doc_tables_are_generated_from_spaces():
    """If this fails, run: python scripts/gen_api_docs.py"""
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "gen_api_docs.py"), "--check"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_api_doc_mentions_every_field_with_units():
    text = API_DOC.read_text(encoding="utf-8")
    channels = spaces.ACTION_CHANNELS + spaces.IMU_CHANNELS + spaces.CONTACT_CHANNELS
    for channel in channels:
        assert f"`{channel.name}`" in text, f"{channel.name} is not documented"
        assert channel.units, f"{channel.name} has no units"
    for key in ("eye_left", "eye_right", "imu", "contact", "time"):
        assert f"`{key}`" in text
    for key, _, _, _ in spaces.INFO_FIELDS:
        assert f"`{key}`" in text


def _driver_example():
    """The code of the example driver in docs/API.md."""
    text = API_DOC.read_text(encoding="utf-8")
    section = text.split("## Writing a new driver")[1]
    return re.search(r"```python\n(.*?)```", section, flags=re.DOTALL).group(1)


def test_driver_example_is_twenty_lines():
    lines = _driver_example().rstrip("\n").split("\n")
    assert len(lines) <= 20, f"the example driver in docs/API.md has {len(lines)} lines"


def test_driver_example_actually_runs(make_env):
    """The documentation must not rot: run the example driver in the simulator."""
    namespace = {}
    exec(_driver_example(), namespace)
    driver = namespace["SlowSeeker"]()

    env = make_env(small_scenario())
    observation, info = env.reset(seed=0)
    driver.reset()
    for _ in range(50):
        action = driver.act(observation)
        assert env.action_space.contains(np.asarray(action, dtype=np.float32))
        observation, _, _, _, info = env.step(action)
    assert info["pose"][0] > 0.1   # it drove forwards


# ----------------------------------------------------------------------------
# The scenario loader and the config registry
# ----------------------------------------------------------------------------


def test_defaults_are_filled_in():
    scenario = load_scenario({"name": "tiny"})
    assert scenario["eyes"]["resolution"] == [20, 20]
    assert scenario["arena"]["walls"] is True
    assert scenario["lights"] == []


def test_misspelled_setting_is_rejected():
    with pytest.raises(ScenarioError, match="max_tim_s"):
        load_scenario({"episode": {"max_tim_s": 10}})


def test_sample_number_and_range():
    rng = np.random.default_rng(0)
    assert sample(3.5, rng) == 3.5
    for _ in range(100):
        assert 2.0 <= sample([2.0, 4.0], rng) <= 4.0


def test_scenario_error_is_still_importable_from_scenario_py():
    import flycar.config
    import flycar.env.scenario
    assert flycar.env.scenario.ScenarioError is flycar.config.ScenarioError


def test_two_teams_cannot_declare_the_same_section():
    import types

    from flycar import config

    def validate(scenario, where):
        pass

    clash = types.SimpleNamespace(__name__="flycar.config.clash", SECTIONS={"eyes": {}}, validate=validate)
    with pytest.raises(ImportError, match="eyes"):
        config._collect(config.TEAM_MODULES + [clash])


def test_every_section_comes_from_exactly_one_team_module():
    from flycar import config

    seen = list(config.TOP_LEVEL_DEFAULTS)
    for module in config.TEAM_MODULES:
        seen.extend(module.SECTIONS)
    assert sorted(seen) == sorted(config.DEFAULTS)
    assert len(seen) == len(set(seen))


# ----------------------------------------------------------------------------
# Names and signatures from docs/CONTRACTS.md (contracts 2 to 8).
# A pull request that renames one of these fails here.
# ----------------------------------------------------------------------------

def _parameters(function):
    """Parameter names of a function or method, without ``self``."""
    import inspect

    return [name for name in inspect.signature(function).parameters if name != "self"]


# Contract 2: the layout dict, key by key.
LAYOUT_TYPES = {
    "arena": dict,
    "walls": list,
    "obstacles": list,
    "lights": list,
    "target": (list, type(None)),
    "spawn": list,
    "lighting": dict,
}
ARENA_KEYS = {"size_m", "walls", "wall_height_m", "wall_thickness_m"}
SOLID_KEYS = {"shape", "x", "y", "yaw", "size_m"}
LIGHT_KEYS = {"x", "y", "z", "intensity", "falloff_m", "target"}


def test_contract_2_layout_keys_and_types():
    from flycar.env.world import sample_layout

    assert _parameters(sample_layout) == ["scenario", "rng"]
    scenario = load_scenario(REPO_ROOT / "scenarios" / "light_seek.yaml")
    layout = sample_layout(scenario, np.random.default_rng(0))

    assert set(layout) == set(LAYOUT_TYPES)
    for key, expected in LAYOUT_TYPES.items():
        assert isinstance(layout[key], expected), key
    assert set(layout["arena"]) == ARENA_KEYS
    assert len(layout["arena"]["size_m"]) == 2
    assert isinstance(layout["arena"]["walls"], bool)
    for solid in layout["walls"] + layout["obstacles"]:
        assert set(solid) == SOLID_KEYS
        assert solid["shape"] in ("box", "cylinder")
    for light in layout["lights"]:
        assert set(light) == LIGHT_KEYS
        assert isinstance(light["target"], bool)
    assert len(layout["target"]) == 2
    assert len(layout["spawn"]) == 3
    assert set(layout["lighting"]) == {"ambient", "sky"}


VEHICLE_METHODS = {
    "pose": [],
    "position": [],
    "velocity": [],
    "rotation_matrix": [],
    "forward_speed": [],
    "yaw_rate": [],
    "set_action": ["action", "control_period"],
    "hold_brake": [],
    "release_controls": [],
}


def test_contract_3_vehicle_accessors(make_env):
    from flycar.env.vehicle import FlyCarVehicle

    for name, parameters in VEHICLE_METHODS.items():
        assert _parameters(getattr(FlyCarVehicle, name)) == parameters, name

    env = make_env(small_scenario())
    env.reset(seed=0)
    for name in ("applied_action", "inputs", "vehicle"):
        assert hasattr(env.vehicle, name), name
    assert env.vehicle.pose().shape == (3,)
    assert env.vehicle.rotation_matrix().shape == (3, 3)


def test_contract_4_world_and_tags(make_env):
    from flycar.env import vehicle, world

    assert _parameters(world.World.current_lights) == []
    assert world.WORLD_TAG_BASE == 1000
    assert vehicle.CHASSIS_TAG == 500

    env = make_env(small_scenario(lights=[{"x": 5.0, "y": 0.0}]))
    _, info = env.reset(seed=0)
    assert env.world.layout is env.layout
    assert env.world.current_lights() == info["world"]["lights"]


def test_contract_5_eyes_interface():
    from flycar.env.sensing import eyes

    assert _parameters(eyes.make_eyes) == ["backend", "system", "vehicle", "world", "config"]
    for backend in (eyes.RaycastEyes, eyes.SensorEyes):
        assert _parameters(backend.__init__) == ["system", "vehicle", "world", "config"], backend.__name__
        assert _parameters(backend.update) == [], backend.__name__
        assert _parameters(backend.render) == [], backend.__name__
        assert _parameters(backend.raw_images) == [], backend.__name__


def test_contract_6_terrain(make_env):
    from flycar.env.terrain import Terrain

    assert _parameters(Terrain.__init__) == ["system", "layout", "terrain_config", "visualize"]
    assert _parameters(Terrain.synchronize) == ["time"]
    assert _parameters(Terrain.advance) == ["step"]

    env = make_env(small_scenario())
    env.reset(seed=0)
    assert env.terrain.chrono_terrain is not None


def test_contract_7_physics_step_order(make_env):
    """Record the calls made by one _physics_step() and compare with docs/CONTRACTS.md."""
    env = make_env(small_scenario())
    env.reset(seed=0)
    calls = []

    def recorder(label, method):
        def wrapped(*args):
            calls.append(label)
            return method(*args)
        return wrapped

    def record_vehicle_synchronize(time, terrain):
        assert terrain is env.terrain.chrono_terrain
        calls.append("vehicle.synchronize")
        return original_vehicle_synchronize(time, terrain)

    original_vehicle_synchronize = env.vehicle.synchronize
    env.terrain.synchronize = recorder("terrain.synchronize", env.terrain.synchronize)
    env.world.synchronize = recorder("world.synchronize", env.world.synchronize)
    env.vehicle.synchronize = record_vehicle_synchronize
    env.terrain.advance = recorder("terrain.advance", env.terrain.advance)
    env.world.advance = recorder("world.advance", env.world.advance)
    env.vehicle.advance = recorder("vehicle.advance", env.vehicle.advance)
    env.eyes.update = recorder("eyes.update", env.eyes.update)
    env.contact.poll = recorder("contact.poll", env.contact.poll)

    class RecordingSystem:
        """Passes everything through to the real ChSystem, noting DoStepDynamics."""

        def __init__(self, system):
            self.system = system

        def __getattr__(self, name):
            return getattr(self.system, name)

        def DoStepDynamics(self, step):
            calls.append("system.DoStepDynamics")
            return self.system.DoStepDynamics(step)

    real_system = env.system
    env.system = RecordingSystem(real_system)
    try:
        env._physics_step()
    finally:
        env.system = real_system

    assert calls == [
        "terrain.synchronize", "world.synchronize", "vehicle.synchronize",
        "terrain.advance", "world.advance", "vehicle.advance",
        "system.DoStepDynamics", "eyes.update", "contact.poll",
    ]


def test_contract_8_driver_protocol():
    from flycar.drivers.base import Driver

    assert _parameters(Driver.reset) == []
    assert _parameters(Driver.act) == ["observation"]


# ----------------------------------------------------------------------------
# Ownership
# ----------------------------------------------------------------------------

def test_every_file_has_exactly_one_owner_in_codeowners():
    """If this fails, run: python scripts/check_codeowners.py"""
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "check_codeowners.py")],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr
