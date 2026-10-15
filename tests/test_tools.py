"""Logger, replay and dashboard (all headless)."""

import numpy as np
import pandas as pd

from conftest import small_scenario
from flycar.drivers import make_driver
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
