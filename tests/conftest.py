"""Shared test setup.

Every test runs headless: no windows are opened, and the eyes use the CPU
``raycast`` backend so results are the same on every machine.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")   # draw plots into memory instead of a window

import pytest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENARIO_DIR = REPO_ROOT / "scenarios"
SCENARIO_FILES = sorted(SCENARIO_DIR.glob("*.yaml"))


def small_scenario(**overrides):
    """A tiny, fully fixed scenario for tests: 20 m arena, car at the centre facing +x."""
    scenario = {
        "name": "test",
        "episode": {"max_time_s": 5.0, "success_radius_m": 2.5},
        "arena": {"size_m": [20.0, 20.0], "walls": True},
        "spawn": {"x": 0.0, "y": 0.0, "yaw_deg": 0.0},
    }
    scenario.update(overrides)
    return scenario


@pytest.fixture
def make_env():
    """Factory for environments that are closed automatically after the test."""
    from flycar.env.core import FlyCarEnv

    created = []

    def factory(scenario, **kwargs):
        kwargs.setdefault("eye_backend", "raycast")
        env = FlyCarEnv(scenario, **kwargs)
        created.append(env)
        return env

    yield factory
    for env in created:
        env.close()
