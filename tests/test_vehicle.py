"""Vehicle & terrain team: the car, its controls, and the ground."""

import math

import numpy as np
import pytest

from conftest import SCENARIO_FILES, small_scenario
from flycar.env import spaces
from flycar.env.scenario import ScenarioError, load_scenario


def test_respawn_matches_the_sampled_pose(make_env):
    path = SCENARIO_FILES[[p.stem for p in SCENARIO_FILES].index("light_seek")]
    env = make_env(path)
    _, info = env.reset(seed=3)
    spawn_x, spawn_y, spawn_yaw = info["world"]["spawn"]
    x, y, yaw = info["pose"]
    assert math.hypot(x - spawn_x, y - spawn_y) < 0.05
    assert abs(yaw - spawn_yaw) < 0.02


def test_actions_are_saturated_and_rate_limited(make_env):
    env = make_env(small_scenario())
    env.reset(seed=0)
    rates = env.scenario["vehicle"]
    previous = np.zeros(3)
    for _ in range(60):
        _, _, _, _, info = env.step([5.0, -9.0, float("nan")])   # nonsense on purpose
        applied = info["applied_action"]
        assert env.action_space.contains(applied)
        change = np.abs(applied - previous)
        assert change[spaces.THROTTLE] <= rates["throttle_rate_per_s"] * 0.01 + 1e-6
        assert change[spaces.STEERING] <= rates["steering_rate_per_s"] * 0.01 + 1e-6
        previous = applied.copy()
    # After 0.6 s the commands have arrived at the limits (NaN brake became 0).
    np.testing.assert_allclose(applied, [1.0, -1.0, 0.0], atol=1e-6)


def test_positive_steering_turns_left_and_speed_is_governed(make_env):
    env = make_env(small_scenario(arena={"size_m": [60.0, 60.0], "walls": True}))
    env.reset(seed=0)
    for _ in range(400):
        observation, _, _, _, info = env.step([1.0, 0.5, 0.0])
    assert observation["imu"][spaces.IMU_YAW_RATE] > 0.1      # turning left
    assert info["pose"][1] > 0.5                              # ended up on the left (+y)
    top_speed = env.scenario["vehicle"]["max_speed_mps"]
    assert 0.8 * top_speed < observation["imu"][spaces.IMU_SPEED] < top_speed + 0.2


# ----------------------------------------------------------------------------
# Terrain
# ----------------------------------------------------------------------------

def test_terrain_defaults_reproduce_the_flat_rigid_ground():
    assert load_scenario({})["terrain"] == {"type": "rigid_flat", "friction": 0.9}


def test_unimplemented_terrain_type_is_rejected():
    with pytest.raises(ScenarioError, match=r"terrain\.type: scm is not implemented yet"):
        load_scenario({"terrain": {"type": "scm"}})


def test_terrain_is_built_from_its_config_and_is_flat_at_z0(make_env):
    import pychrono as chrono

    env = make_env(small_scenario(terrain={"friction": 0.6}))
    env.reset(seed=0)
    centre = chrono.ChVector3d(0.0, 0.0, 0.0)
    assert env.terrain.chrono_terrain.GetHeight(centre) == pytest.approx(0.0)
    assert env.terrain.chrono_terrain.GetCoefficientFriction(centre) == pytest.approx(0.6)
