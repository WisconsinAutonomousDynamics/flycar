"""The default behaviour must not change: rerun the recorded baseline episodes.

The reference runs live in tests/baseline/ and were recorded with
scripts/record_baseline.py. If this test fails, the code now behaves
differently for a default scenario. That is a bug in the change, not a reason
to re-record. (Behaviour changes need a lead-approved ``api-change`` issue.)

Tolerances: step counts and the end-of-episode flags must match exactly; every
float must match to 1e-6 relative (with a 1e-9 absolute floor for numbers that
are essentially zero).
"""

import importlib.util

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

EXACT_KEYS = ("step", "n_steps", "terminated", "truncated", "success", "contact")
FLOAT_KEYS = ("pose", "applied_action", "imu",
              "eye_left_sum", "eye_left_wsum", "eye_right_sum", "eye_right_wsum")


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
