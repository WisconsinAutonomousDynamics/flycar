"""Vehicle & terrain team: the car, its controls, and the ground."""

import math

import numpy as np
import pytest

from conftest import SCENARIO_FILES, small_scenario
from flycar.env import spaces
from flycar.env.scenario import ScenarioError, load_scenario

CONTROL_PERIOD_S = 0.01


def _empty_arena_vehicle_scenario(**vehicle_overrides):
    path = next(path for path in SCENARIO_FILES if path.stem == "empty_arena")
    scenario = load_scenario(path)
    scenario["spawn"].update({"x": 0.0, "y": 0.0, "yaw_deg": 0.0})
    scenario["vehicle"].update(vehicle_overrides)
    return scenario


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


@pytest.mark.slow
def test_lower_steering_gain_reduces_yaw_rate(make_env):
    yaw_rates = []
    for gain in (1.0, 0.5):
        env = make_env(_empty_arena_vehicle_scenario(steering_gain=gain))
        observation, _ = env.reset(seed=0)
        for _ in range(250):
            observation, _, _, _, _ = env.step([0.2, 1.0, 0.0])
        yaw_rates.append(abs(float(observation["imu"][spaces.IMU_YAW_RATE])))

    assert yaw_rates[1] < yaw_rates[0] * 0.9


@pytest.mark.slow
def test_rigid_tires_change_braking_distance(make_env):
    distances = []
    for tire_model in ("tmeasy", "rigid"):
        scenario = _empty_arena_vehicle_scenario(tire_model=tire_model)
        scenario["arena"].update({"size_m": [200.0, 200.0], "walls": False})
        env = make_env(scenario)
        observation, _ = env.reset(seed=0)
        for _ in range(1000):
            observation, _, _, _, info = env.step([1.0, 0.0, 0.0])
            if observation["imu"][spaces.IMU_SPEED] >= 3.9:
                break
        assert observation["imu"][spaces.IMU_SPEED] >= 3.9
        start_x = float(info["pose"][0])
        for _ in range(500):
            observation, _, _, _, info = env.step([0.0, 0.0, 1.0])
            if observation["imu"][spaces.IMU_SPEED] <= 0.05:
                break
        assert observation["imu"][spaces.IMU_SPEED] <= 0.05
        distances.append(float(info["pose"][0]) - start_x)

    assert abs(distances[1] - distances[0]) > max(0.05, distances[0] * 0.02)


@pytest.mark.slow
def test_larger_throttle_exponent_slows_partial_throttle_acceleration(make_env):
    times = []
    for exponent in (1.0, 2.0):
        env = make_env(_empty_arena_vehicle_scenario(throttle_curve_exponent=exponent))
        observation, _ = env.reset(seed=0)
        for step in range(1500):
            observation, _, _, _, _ = env.step([0.5, 0.0, 0.0])
            if observation["imu"][spaces.IMU_SPEED] >= 3.0:
                times.append((step + 1) * CONTROL_PERIOD_S)
                break

    assert len(times) == 2, "both throttle settings must reach 3 m/s"
    assert times[1] > times[0] * 1.1


def test_vehicle_tuning_defaults_preserve_the_current_behaviour():
    """Check that the default values in config/vehicle.py reproduce the current behaviour.

    This is a regression test for the vehicle & terrain team. If you change the defaults,
    you must update this test to match the new behaviour.
    """
    scenario = load_scenario({})
    vehicle = scenario["vehicle"]
    assert vehicle["tire_model"] == "tmeasy"
    assert vehicle["steering_gain"] == 1.0
    assert vehicle["throttle_curve_exponent"] == 1.0


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
