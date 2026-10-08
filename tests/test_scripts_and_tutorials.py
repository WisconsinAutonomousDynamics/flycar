"""The things students run from the command line must keep working."""

import subprocess
import sys

import pytest

from conftest import REPO_ROOT

TUTORIALS = sorted((REPO_ROOT / "tutorials").glob("[0-9][0-9]_*.py"))


def _run(arguments, timeout=300):
    return subprocess.run(
        [sys.executable, *[str(a) for a in arguments]],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout,
    )


def test_check_env_passes():
    result = _run([REPO_ROOT / "scripts" / "check_env.py"])
    assert result.returncode == 0, result.stdout + result.stderr
    assert "RESULT: OK" in result.stdout
    assert "Chrono version:" in result.stdout


def test_the_three_tutorials_exist():
    assert [path.name for path in TUTORIALS] == [
        "01_falling_box.py",
        "02_motor_cart.py",
        "03_vehicle_figure8.py",
    ]


@pytest.mark.slow
@pytest.mark.parametrize("path", TUTORIALS, ids=lambda p: p.stem)
def test_tutorial_runs_headless(path):
    result = _run([path, "--headless"])
    assert result.returncode == 0, result.stdout + result.stderr
    assert path.with_suffix(".png").exists()


@pytest.mark.slow
def test_run_py_with_random_driver_and_logging(tmp_path):
    result = _run([
        REPO_ROOT / "run.py",
        "--scenario", REPO_ROOT / "scenarios" / "empty_arena.yaml",
        "--driver", "random", "--eyes", "raycast", "--log", tmp_path,
    ])
    assert result.returncode == 0, result.stdout + result.stderr
    assert "eye backend: raycast" in result.stdout
    assert (tmp_path / "empty_arena_random_seed0.parquet").exists()


@pytest.mark.slow
def test_braitenberg_reaches_the_light_in_most_seeds():
    """Acceptance criterion 3, on the first five seeds."""
    result = _run([
        REPO_ROOT / "run.py",
        "--scenario", REPO_ROOT / "scenarios" / "light_seek.yaml",
        "--driver", "braitenberg", "--eyes", "raycast", "--episodes", "5",
    ], timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    successes = result.stdout.count("reached the target after")
    assert successes >= 4, result.stdout
