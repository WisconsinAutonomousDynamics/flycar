"""The environment itself: construction, observations, timing, determinism."""

import math

import numpy as np
import pytest

from conftest import SCENARIO_FILES, small_scenario
from flycar.env import spaces


# ----------------------------------------------------------------------------
# Construction and observations, for every scenario file
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("path", SCENARIO_FILES, ids=lambda p: p.stem)
def test_scenario_builds_and_observations_match_the_spaces(path, make_env):
    env = make_env(path)
    expected_space = spaces.make_observation_space(env.scenario["eyes"]["resolution"])
    assert env.observation_space == expected_space
    assert env.action_space == spaces.make_action_space()

    observation, info = env.reset(seed=0)
    assert "world" in info

    for _ in range(30):
        # Shapes, dtypes and ranges, all checked against spaces.py.
        assert set(observation) == set(expected_space.spaces)
        for key, box in expected_space.spaces.items():
            assert observation[key].shape == box.shape, key
            assert observation[key].dtype == box.dtype, key
        assert expected_space.contains(observation)

        # Every documented info key is present (except "world": reset only).
        documented = {key for key, _, _, _ in spaces.INFO_FIELDS}
        assert set(info) - {"world"} == documented - {"world"}
        assert info["eye_backend"] == "raycast"
        assert info["camera_left"] is None

        action = env.action_space.sample()
        action[spaces.BRAKE] = 0.0
        observation, reward, terminated, truncated, info = env.step(action)
        assert reward == 0.0
        assert isinstance(terminated, bool) and isinstance(truncated, bool)


def test_step_before_reset_is_an_error(make_env):
    env = make_env(small_scenario())
    with pytest.raises(RuntimeError, match="reset"):
        env.step([0.0, 0.0, 0.0])


# ----------------------------------------------------------------------------
# Timing
# ----------------------------------------------------------------------------

def test_n_steps_advance_time_by_n_control_periods(make_env):
    env = make_env(small_scenario())
    observation, info = env.reset(seed=0)
    assert info["time"] == 0.0
    assert observation["time"][0] == 0.0
    chrono_time_at_start = env.system.GetChTime()

    steps = 37
    for _ in range(steps):
        observation, _, _, _, info = env.step([0.3, 0.0, 0.0])

    assert info["step"] == steps
    assert info["time"] == pytest.approx(steps * 0.01, abs=1e-12)
    assert observation["time"][0] == pytest.approx(steps * 0.01, abs=1e-6)
    # Chrono's own clock agrees: exactly 10 physics steps of 1 ms per control step.
    assert env.system.GetChTime() - chrono_time_at_start == pytest.approx(steps * 0.01, abs=1e-9)


def test_control_period_is_configurable(make_env):
    scenario = small_scenario(
        timing={"control_period_s": 0.02, "physics_step_s": 0.002},
        eyes={"update_period_s": 0.04},
    )
    env = make_env(scenario)
    env.reset(seed=0)
    assert env.substeps == 10
    for _ in range(5):
        _, _, _, _, info = env.step([0.0, 0.0, 0.0])
    assert info["time"] == pytest.approx(0.1)


def test_episode_is_truncated_at_max_time(make_env):
    env = make_env(small_scenario(episode={"max_time_s": 0.2}))
    env.reset(seed=0)
    for step in range(1, 21):
        _, _, terminated, truncated, _ = env.step([0.0, 0.0, 1.0])
        assert not terminated
        assert truncated == (step == 20)


# ----------------------------------------------------------------------------
# Determinism
# ----------------------------------------------------------------------------

def _rollout(env, seed, steps=150):
    """Run a fixed action sequence. Returns everything that should be reproducible."""
    actions = np.random.default_rng(123).uniform([0.2, -1.0, 0.0], [1.0, 1.0, 0.0], size=(steps, 3))
    observation, info = env.reset(seed=seed)
    layout = info["world"]
    poses = [info["pose"].copy()]
    imus = [observation["imu"].copy()]
    eyes = [observation["eye_left"].copy()]
    for action in actions:
        observation, _, _, _, info = env.step(action)
        poses.append(info["pose"].copy())
        imus.append(observation["imu"].copy())
        eyes.append(observation["eye_left"].copy())
    return np.array(poses), np.array(imus), np.array(eyes), layout


def test_same_seed_and_actions_give_the_same_trajectory(make_env):
    path = SCENARIO_FILES[[p.stem for p in SCENARIO_FILES].index("light_seek")]

    env_a = make_env(path)
    poses_1, imus_1, eyes_1, layout_1 = _rollout(env_a, seed=7)
    _rollout(env_a, seed=99)                                    # something else in between
    poses_2, imus_2, eyes_2, layout_2 = _rollout(env_a, seed=7)  # same env, same seed again

    env_b = make_env(path)                                      # a completely fresh env
    poses_3, imus_3, eyes_3, layout_3 = _rollout(env_b, seed=7)

    # Bit-for-bit identical, not merely close.
    assert layout_1 == layout_2 == layout_3
    np.testing.assert_array_equal(poses_1, poses_2)
    np.testing.assert_array_equal(poses_1, poses_3)
    np.testing.assert_array_equal(imus_1, imus_2)
    np.testing.assert_array_equal(imus_1, imus_3)
    np.testing.assert_array_equal(eyes_1, eyes_2)
    np.testing.assert_array_equal(eyes_1, eyes_3)

    # And the car really moved, so this is not trivially true.
    assert np.linalg.norm(poses_1[-1, :2] - poses_1[0, :2]) > 1.0


def test_different_seeds_give_different_layouts(make_env):
    path = SCENARIO_FILES[[p.stem for p in SCENARIO_FILES].index("light_seek")]
    env = make_env(path)
    _, info_a = env.reset(seed=1)
    _, info_b = env.reset(seed=2)
    _, info_c = env.reset()          # no seed: the random stream just continues
    assert info_a["world"]["spawn"] != info_b["world"]["spawn"]
    assert info_b["world"]["spawn"] != info_c["world"]["spawn"]


def test_respawn_matches_the_sampled_pose(make_env):
    path = SCENARIO_FILES[[p.stem for p in SCENARIO_FILES].index("light_seek")]
    env = make_env(path)
    _, info = env.reset(seed=3)
    spawn_x, spawn_y, spawn_yaw = info["world"]["spawn"]
    x, y, yaw = info["pose"]
    assert math.hypot(x - spawn_x, y - spawn_y) < 0.05
    assert abs(yaw - spawn_yaw) < 0.02


# ----------------------------------------------------------------------------
# Vehicle: saturation, rate limiting, signs
# ----------------------------------------------------------------------------

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
# Contact, target, eyes
# ----------------------------------------------------------------------------

def test_driving_into_a_wall_sets_the_front_contact_flag(make_env):
    env = make_env(small_scenario(episode={"max_time_s": 10.0}))
    observation, _ = env.reset(seed=0)
    assert not observation["contact"].any()

    touched_front = False
    for _ in range(600):
        observation, _, _, _, info = env.step([1.0, 0.0, 0.0])
        touched_front = touched_front or observation["contact"][spaces.CONTACT_FRONT] == 1.0
    assert touched_front
    # The wall's inner face is at x = 10 m and the bumper's front is 1.3 m ahead
    # of the chassis origin, so the car must have stopped near x = 8.7 m.
    assert 8.3 < info["pose"][0] < 8.8
    assert abs(observation["imu"][spaces.IMU_SPEED]) < 0.2


def test_reaching_the_target_terminates_the_episode(make_env):
    scenario = small_scenario(lights=[{"x": 7.0, "y": 0.0, "target": True}])
    env = make_env(scenario)
    _, info = env.reset(seed=0)
    assert info["target_distance"] == pytest.approx(7.0, abs=0.05)
    assert info["target_bearing"] == pytest.approx(0.0, abs=0.02)

    for _ in range(500):
        _, _, terminated, truncated, info = env.step([0.6, 0.0, 0.0])
        if terminated or truncated:
            break
    assert terminated and not truncated
    assert info["success"]
    assert info["target_distance"] <= 2.5


def test_no_target_gives_nan_distance(make_env):
    env = make_env(small_scenario())
    _, info = env.reset(seed=0)
    assert math.isnan(info["target_distance"])
    assert math.isnan(info["target_bearing"])
    assert info["success"] is False


def test_a_lamp_on_the_left_is_brighter_in_the_left_eye(make_env):
    lamp_left = small_scenario(lights=[{"x": 5.0, "y": 6.0, "target": True}],
                               lighting={"ambient": 0.05, "sky": 0.02})
    env = make_env(lamp_left)
    observation, info = env.reset(seed=0)
    assert info["target_bearing"] > 0.5                       # the lamp is to the left
    assert observation["eye_left"].max() > observation["eye_right"].max() + 0.1


def test_an_obstacle_hides_the_lamp(make_env):
    lights = [{"x": 9.0, "y": 0.0, "z": 1.0, "target": True}]
    lighting = {"ambient": 0.05, "sky": 0.02}
    visible = make_env(small_scenario(lights=lights, lighting=lighting))
    bright, _ = visible.reset(seed=0)

    # The same scene with a tall, wide box between the car and the lamp.
    box = [{"shape": "box", "size_m": [1.0, 6.0, 4.0], "x": 5.0, "y": 0.0, "clearance_m": 0.0}]
    hidden = make_env(small_scenario(lights=lights, lighting=lighting, obstacles=box))
    dark, _ = hidden.reset(seed=0)

    assert bright["eye_left"].max() > 0.3
    assert dark["eye_left"].max() < 0.5 * bright["eye_left"].max()


def test_raycast_eyes_follow_the_lights_the_world_reports(make_env):
    """Eyes ask World.current_lights() on every picture, so a moved lamp is seen moved."""
    scenario = small_scenario(lights=[{"x": 5.0, "y": 6.0, "target": True}],
                              lighting={"ambient": 0.05, "sky": 0.02})
    env = make_env(scenario)
    env.reset(seed=0)
    left, right = env.eyes.render()
    assert left.max() > right.max() + 0.1                   # lamp on the left

    # Pretend World moved the lamp to the right-hand side.
    moved = [dict(light, y=-6.0) for light in env.world.layout["lights"]]
    env.world.current_lights = lambda: moved
    left, right = env.eyes.render()
    assert right.max() > left.max() + 0.1


def test_sensor_eyes_move_their_scene_lights_when_the_world_moves_a_lamp():
    """SensorEyes.update() logic, without a GPU: a stand-in sensor manager records the calls."""
    from flycar.env.sensing.eyes import SensorEyes

    class FakeScene:
        def __init__(self):
            self.modified = []

        def ModifyPointLight(self, light_id, position, color, max_range):
            self.modified.append((light_id, position.x, position.y))

    class FakeManager:
        def __init__(self):
            self.scene = FakeScene()
            self.updates = 0

        def Update(self):
            self.updates += 1

    class FakeWorld:
        lights = [{"x": 1.0, "y": 2.0, "z": 1.5, "intensity": 1.0, "falloff_m": 6.0, "target": True}]

        def current_lights(self):
            return [dict(light) for light in self.lights]

    eyes = SensorEyes.__new__(SensorEyes)   # skip __init__: it needs a GPU
    eyes.world = FakeWorld()
    eyes.manager = FakeManager()
    eyes.light_ids = [7]
    eyes.light_positions = [(1.0, 2.0, 1.5)]

    eyes.update()
    assert eyes.manager.scene.modified == []                 # nothing moved
    eyes.world.lights[0]["y"] = -3.0
    eyes.update()
    assert eyes.manager.scene.modified == [(7, 1.0, -3.0)]
    assert eyes.manager.updates == 2
