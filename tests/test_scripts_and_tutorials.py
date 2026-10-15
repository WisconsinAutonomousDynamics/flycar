"""The learning material and the setup check must keep working. Owned by the project leads."""

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
