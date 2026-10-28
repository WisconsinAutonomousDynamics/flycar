"""Core & body sensing team: the environment loop, timing, determinism, IMU and contact."""

import math

import numpy as np
import pytest

from conftest import SCENARIO_FILES, small_scenario
from flycar.env import spaces
from flycar.env.scenario import ScenarioError, load_scenario


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


def test_solver_and_thread_defaults_are_todays_settings(make_env):
    timing = load_scenario({})["timing"]
    assert timing["max_solver_iterations"] == 150
    assert timing["num_threads"] == 1

    env = make_env(small_scenario())
    env.reset(seed=0)
    assert env.system.GetSolver().AsIterative().GetMaxIterations() == 150
    threads = [env.system.GetNumThreadsChrono(), env.system.GetNumThreadsCollision(), env.system.GetNumThreadsEigen()]
    assert threads == [1, 1, 1]


def test_solver_iterations_and_threads_are_configurable(make_env):
    env = make_env(small_scenario(timing={"max_solver_iterations": 40, "num_threads": 2}))
    env.reset(seed=0)
    assert env.system.GetSolver().AsIterative().GetMaxIterations() == 40
    threads = [env.system.GetNumThreadsChrono(), env.system.GetNumThreadsCollision(), env.system.GetNumThreadsEigen()]
    assert threads == [2, 2, 2]


@pytest.mark.parametrize("key", ["max_solver_iterations", "num_threads"])
@pytest.mark.parametrize("value", [0, -1, 1.5, True, "4"])
def test_solver_iterations_and_threads_must_be_positive_whole_numbers(key, value):
    with pytest.raises(ScenarioError, match=rf"timing\.{key}"):
        load_scenario({"timing": {key: value}})


def test_fast_single_threaded_settings_stay_deterministic(make_env):
    """The A3 "fast batch" settings still give bit-for-bit identical runs."""
    scenario = small_scenario(timing={"physics_step_s": 0.002, "max_solver_iterations": 50})
    env = make_env(scenario)
    poses_1 = _rollout(env, seed=3)[0]
    poses_2 = _rollout(make_env(scenario), seed=3)[0]
    np.testing.assert_array_equal(poses_1, poses_2)
    assert np.linalg.norm(poses_1[-1, :2] - poses_1[0, :2]) > 1.0


def test_control_period_must_be_a_multiple_of_the_physics_step():
    with pytest.raises(ScenarioError, match="multiple"):
        load_scenario({"timing": {"control_period_s": 0.01, "physics_step_s": 0.003}})


def test_episode_is_truncated_at_max_time(make_env):
    env = make_env(small_scenario(episode={"max_time_s": 0.2}))
    env.reset(seed=0)
    for step in range(1, 21):
        _, _, terminated, truncated, _ = env.step([0.0, 0.0, 1.0])
        assert not terminated
        assert truncated == (step == 20)


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


def test_extras_collects_extra_info_by_team(make_env):
    env = make_env(small_scenario())
    _, info = env.reset(seed=0)
    assert info["extras"] == {}               # nothing provides extra_info() yet

    env.contact.extra_info = lambda: {"touches": 0}
    env.terrain.extra_info = lambda: {"patch": "flat"}
    _, _, _, _, info = env.step([0.0, 0.0, 0.0])
    assert info["extras"] == {"core": {"contact": {"touches": 0}}, "vehicle": {"terrain": {"patch": "flat"}}}


@pytest.mark.parametrize("settings, key", [
    ({"contact": {"mode": "force"}}, "contact.mode"),
    ({"termination": {"flipped": True}}, "termination.flipped"),
    ({"termination": {"stuck_s": 3.0}}, "termination.stuck_s"),
    ({"termination": {"left_arena": True}}, "termination.left_arena"),
])
def test_contact_modes_and_extra_terminations_are_not_implemented_yet(settings, key):
    scenario = load_scenario({})
    assert scenario["contact"] == {"mode": "flag"}
    assert scenario["termination"] == {"flipped": False, "stuck_s": None, "left_arena": False}
    with pytest.raises(ScenarioError, match=key.replace(".", r"\.") + ".*not implemented yet"):
        load_scenario(settings)
