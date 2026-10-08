"""spaces.py is the single source of truth, and docs/API.md must agree with it."""

import re
import subprocess
import sys

import numpy as np

from conftest import REPO_ROOT, small_scenario
from flycar.env import spaces

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
