"""Scenario loading and layout sampling (no Chrono needed for the loader)."""

import numpy as np
import pytest

from conftest import SCENARIO_FILES
from flycar.env.scenario import ScenarioError, load_scenario, sample


def test_there_are_scenarios_to_test():
    names = [path.name for path in SCENARIO_FILES]
    assert "empty_arena.yaml" in names
    assert "light_seek.yaml" in names


@pytest.mark.parametrize("path", SCENARIO_FILES, ids=lambda p: p.stem)
def test_scenario_files_load(path):
    scenario = load_scenario(path)
    assert scenario["name"] == path.stem
    assert scenario["timing"]["control_period_s"] == 0.01   # the default from the brief


def test_defaults_are_filled_in():
    scenario = load_scenario({"name": "tiny"})
    assert scenario["eyes"]["resolution"] == [20, 20]
    assert scenario["arena"]["walls"] is True
    assert scenario["lights"] == []


def test_misspelled_setting_is_rejected():
    with pytest.raises(ScenarioError, match="max_tim_s"):
        load_scenario({"episode": {"max_tim_s": 10}})


def test_control_period_must_be_a_multiple_of_the_physics_step():
    with pytest.raises(ScenarioError, match="multiple"):
        load_scenario({"timing": {"control_period_s": 0.01, "physics_step_s": 0.003}})


def test_only_one_target_light():
    lights = [{"x": 1, "y": 0, "target": True}, {"x": 2, "y": 0, "target": True}]
    with pytest.raises(ScenarioError, match="target"):
        load_scenario({"lights": lights})


def test_sample_number_and_range():
    rng = np.random.default_rng(0)
    assert sample(3.5, rng) == 3.5
    for _ in range(100):
        assert 2.0 <= sample([2.0, 4.0], rng) <= 4.0


def test_layout_is_reproducible_and_respects_constraints():
    from flycar.env.world import sample_layout

    scenario = load_scenario(SCENARIO_FILES[[p.stem for p in SCENARIO_FILES].index("light_seek")])
    first = sample_layout(scenario, np.random.default_rng(42))
    again = sample_layout(scenario, np.random.default_rng(42))
    other = sample_layout(scenario, np.random.default_rng(43))
    assert first == again
    assert first["spawn"] != other["spawn"]

    spawn_x, spawn_y, _ = first["spawn"]
    target_x, target_y = first["target"]
    assert np.hypot(spawn_x - target_x, spawn_y - target_y) >= scenario["spawn"]["min_target_distance_m"]
    assert len(first["obstacles"]) == 4
    assert len(first["walls"]) == 4


def test_scenario_error_is_still_importable_from_scenario_py():
    import flycar.config
    import flycar.env.scenario
    assert flycar.env.scenario.ScenarioError is flycar.config.ScenarioError


def test_two_teams_cannot_declare_the_same_section():
    import types

    from flycar import config

    def validate(scenario, where):
        pass

    clash = types.SimpleNamespace(__name__="flycar.config.clash", SECTIONS={"eyes": {}}, validate=validate)
    with pytest.raises(ImportError, match="eyes"):
        config._collect(config.TEAM_MODULES + [clash])


def test_every_section_comes_from_exactly_one_team_module():
    from flycar import config

    seen = list(config.TOP_LEVEL_DEFAULTS)
    for module in config.TEAM_MODULES:
        seen.extend(module.SECTIONS)
    assert sorted(seen) == sorted(config.DEFAULTS)
    assert len(seen) == len(set(seen))


@pytest.mark.parametrize("settings, message", [
    ({"terrain": {"type": "scm"}}, "terrain.type: scm is not implemented yet"),
    ({"motion": {"lights": [{"x": 1.0}]}}, "motion.lights"),
    ({"motion": {"obstacles": [{"x": 1.0}]}}, "motion.obstacles"),
    ({"contact": {"mode": "force"}}, "contact.mode: force is not implemented yet"),
    ({"termination": {"flipped": True}}, "termination.flipped"),
    ({"termination": {"stuck_s": 3.0}}, "termination.stuck_s"),
    ({"termination": {"left_arena": True}}, "termination.left_arena"),
    ({"eyes": {"noise_std": 0.1}}, "eyes.noise_std"),
    ({"eyes": {"motion_blur": True}}, "eyes.motion_blur"),
])
def test_reserved_settings_reject_anything_but_the_default(settings, message):
    with pytest.raises(ScenarioError, match=message.replace(".", r"\.")):
        load_scenario(settings)


def test_reserved_sections_default_to_todays_behaviour():
    scenario = load_scenario({})
    assert scenario["terrain"] == {"type": "rigid_flat", "friction": 0.9}
    assert scenario["motion"] == {"obstacles": [], "lights": []}
    assert scenario["contact"] == {"mode": "flag"}
    assert scenario["termination"] == {"flipped": False, "stuck_s": None, "left_arena": False}
    assert scenario["eyes"]["noise_std"] == 0.0
    assert scenario["eyes"]["motion_blur"] is False
