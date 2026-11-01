"""The default behaviour must not change: rerun the recorded baseline.

The reference data lives in tests/baseline/ and was recorded on Linux with
scripts/record_baseline.py (its docstring explains the three parts). If this
test fails, the code now behaves differently for a default scenario. That is a
bug in the change, not a reason to re-record. (Behaviour changes need a
lead-approved ``api-change`` issue.)

* ``test_setup_is_unchanged``: runs on every OS. Every default, layout and
  Chrono build parameter must match exactly (1e-9 relative). New settings and
  constants are fine; changed or removed ones are not.
* ``test_maneuver_is_unchanged``: runs on every OS. A scripted drive must end
  up within a few centimetres of the recording.
* ``test_baseline_episode_is_unchanged``: CI only. Step counts and
  end-of-episode flags must match exactly; every float to 1e-6 relative,
  with a 1e-9 absolute floor for numbers that are essentially zero. The last
  digits of the physics depend on the OS, its glibc version and even the CPU,
  and a closed-loop episode turns those into a different path. So the episodes
  are recorded and checked on GitHub's machine under fixed conditions (see
  .github/workflows/ci.yml and record-baseline.yml), and skipped unless
  FLYCAR_BASELINE_EPISODES=1.
"""

import importlib.util
import json
import math
import os
import sys

import numpy as np
import pytest

from conftest import REPO_ROOT

RELATIVE_TOLERANCE = 1e-6
ABSOLUTE_FLOOR = 1e-9

# Load scripts/record_baseline.py, so the test runs the episodes exactly the
# way they were recorded.
_spec = importlib.util.spec_from_file_location("record_baseline", REPO_ROOT / "scripts" / "record_baseline.py")
record_baseline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(record_baseline)

# Setup values are computed, not simulated, so only rounding may differ.
SETUP_RELATIVE_TOLERANCE = 1e-9
SETUP_ABSOLUTE_FLOOR = 1e-12

# The maneuver may differ by platform rounding. Nudging the spawn pose by
# 1e-12 to 1e-5 (more than any rounding) moved the samples by at most 2 cm and
# 0.004 rad. A changed default moves them far more: ground friction
# 0.9 -> 0.85 by 0.45 m, max_speed_mps 4.0 -> 3.9 by 0.47 m. Settings that the
# maneuver cannot see (a small rate limit change, solver iterations) are caught
# exactly by test_setup_is_unchanged instead.
MANEUVER_POSITION_TOLERANCE_M = 0.15
MANEUVER_YAW_TOLERANCE_RAD = 0.1
MANEUVER_SPEED_TOLERANCE_MPS = 0.1
MANEUVER_YAW_RATE_TOLERANCE_RADPS = 0.1

EXACT_KEYS = ("step", "n_steps", "terminated", "truncated", "success", "contact")
FLOAT_KEYS = ("pose", "applied_action", "imu",
              "eye_left_sum", "eye_left_wsum", "eye_right_sum", "eye_right_wsum")


def assert_same_setup(expected, actual, where="setup"):
    """Every value recorded in ``expected`` must still be in ``actual``, unchanged.

    Keys that only ``actual`` has are allowed (a new, opt-in setting). Lists
    must keep their length (one more body in Chrono is a behaviour change).
    """
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{where}: was a dict, now {actual!r}"
        for key, value in expected.items():
            assert key in actual, f"{where}.{key}: missing (was {value!r})"
            assert_same_setup(value, actual[key], f"{where}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list), f"{where}: was a list, now {actual!r}"
        assert len(actual) == len(expected), f"{where}: had {len(expected)} entries, now {len(actual)}"
        for i, (old, new) in enumerate(zip(expected, actual)):
            assert_same_setup(old, new, f"{where}[{i}]")
    elif isinstance(expected, float) and not isinstance(actual, bool) and isinstance(actual, (int, float)):
        assert math.isclose(actual, expected, rel_tol=SETUP_RELATIVE_TOLERANCE, abs_tol=SETUP_ABSOLUTE_FLOOR), (
            f"{where}: was {expected!r}, now {actual!r}"
        )
    else:
        assert actual == expected, f"{where}: was {expected!r}, now {actual!r}"


def test_setup_is_unchanged():
    path = record_baseline.BASELINE_DIR / "setup.json"
    assert path.exists(), f"missing baseline {path}; it is recorded once, by the leads"
    expected = json.loads(path.read_text())

    # Round-trip through JSON, so both sides have the same types (lists, not tuples).
    actual = json.loads(json.dumps(record_baseline.setup_snapshot()))

    assert_same_setup(expected, actual)


def test_maneuver_is_unchanged():
    path = record_baseline.BASELINE_DIR / "maneuver.npz"
    assert path.exists(), f"missing baseline {path}; it is recorded once, by the leads"
    expected = np.load(path)

    actual = record_baseline.run_maneuver()

    np.testing.assert_array_equal(actual["step"], expected["step"], err_msg="maneuver: step changed")
    # The action limits are plain NumPy, no physics: these match to rounding.
    np.testing.assert_allclose(
        actual["applied_action"], expected["applied_action"], rtol=RELATIVE_TOLERANCE, atol=ABSOLUTE_FLOOR,
        err_msg="maneuver: applied_action changed",
    )
    np.testing.assert_allclose(
        actual["pose"][:, :2], expected["pose"][:, :2], rtol=0, atol=MANEUVER_POSITION_TOLERANCE_M,
        err_msg="maneuver: the car ended up somewhere else (x, y) [m]",
    )
    yaw_difference = np.angle(np.exp(1j * (actual["pose"][:, 2] - expected["pose"][:, 2])))
    np.testing.assert_array_less(
        np.abs(yaw_difference), MANEUVER_YAW_TOLERANCE_RAD,
        err_msg="maneuver: the car faces a different way (yaw) [rad]",
    )
    np.testing.assert_allclose(
        actual["imu"][:, 0], expected["imu"][:, 0], rtol=0, atol=MANEUVER_SPEED_TOLERANCE_MPS,
        err_msg="maneuver: speed changed [m/s]",
    )
    np.testing.assert_allclose(
        actual["imu"][:, 1], expected["imu"][:, 1], rtol=0, atol=MANEUVER_YAW_RATE_TOLERANCE_RADPS,
        err_msg="maneuver: yaw rate changed [rad/s]",
    )


@pytest.mark.skipif(
    not sys.platform.startswith("linux") or os.environ.get("FLYCAR_BASELINE_EPISODES") != "1",
    reason="full episodes only reproduce on CI's machine; CI runs them (FLYCAR_BASELINE_EPISODES=1)",
)
@pytest.mark.parametrize("run", record_baseline.BASELINE_RUNS, ids=lambda run: run[0])
def test_baseline_episode_is_unchanged(run):
    name, scenario_path, driver_name, seed = run
    path = record_baseline.BASELINE_DIR / f"{name}.npz"
    assert path.exists(), f"missing baseline {path}; it is recorded once, by the leads"
    expected = np.load(path)

    actual = record_baseline.run_episode(scenario_path, driver_name, seed)

    assert set(actual) == set(expected.files)
    for key in EXACT_KEYS:
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=f"{name}: {key} changed")
    for key in FLOAT_KEYS:
        np.testing.assert_allclose(
            actual[key], expected[key], rtol=RELATIVE_TOLERANCE, atol=ABSOLUTE_FLOOR,
            err_msg=f"{name}: {key} changed",
        )
