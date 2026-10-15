"""World team: scenario files, layout sampling, walls, obstacles and lights."""

import numpy as np
import pytest

from conftest import SCENARIO_FILES, small_scenario
from flycar.env.scenario import ScenarioError, load_scenario


def test_there_are_scenarios_to_test():
    names = [path.name for path in SCENARIO_FILES]
    assert "empty_arena.yaml" in names
    assert "light_seek.yaml" in names


@pytest.mark.parametrize("path", SCENARIO_FILES, ids=lambda p: p.stem)
def test_scenario_files_load(path):
    scenario = load_scenario(path)
    assert scenario["name"] == path.stem
    assert scenario["timing"]["control_period_s"] == 0.01   # the default from the brief


def test_only_one_target_light():
    lights = [{"x": 1, "y": 0, "target": True}, {"x": 2, "y": 0, "target": True}]
    with pytest.raises(ScenarioError, match="target"):
        load_scenario({"lights": lights})


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


def test_different_seeds_give_different_layouts(make_env):
    path = SCENARIO_FILES[[p.stem for p in SCENARIO_FILES].index("light_seek")]
    env = make_env(path)
    _, info_a = env.reset(seed=1)
    _, info_b = env.reset(seed=2)
    _, info_c = env.reset()          # no seed: the random stream just continues
    assert info_a["world"]["spawn"] != info_b["world"]["spawn"]
    assert info_b["world"]["spawn"] != info_c["world"]["spawn"]


def test_current_lights_reports_the_layout_lights_in_the_world_frame(make_env):
    env = make_env(SCENARIO_FILES[[p.stem for p in SCENARIO_FILES].index("light_seek")])
    _, info = env.reset(seed=0)
    lights = env.world.current_lights()
    assert lights == info["world"]["lights"]
    assert set(lights[0]) == {"x", "y", "z", "intensity", "falloff_m", "target"}

    # The result is a copy: changing it does not move the layout's lamp.
    lights[0]["x"] += 100.0
    assert env.world.current_lights() == info["world"]["lights"]


def test_the_world_has_no_terrain(make_env):
    env = make_env(small_scenario())
    env.reset(seed=0)
    assert not hasattr(env.world, "terrain")


@pytest.mark.parametrize("settings, key", [
    ({"motion": {"lights": [{"x": 1.0}]}}, "motion.lights"),
    ({"motion": {"obstacles": [{"x": 1.0}]}}, "motion.obstacles"),
])
def test_moving_things_are_not_implemented_yet(settings, key):
    assert load_scenario({})["motion"] == {"obstacles": [], "lights": []}
    with pytest.raises(ScenarioError, match=key.replace(".", r"\.") + ".*not implemented yet"):
        load_scenario(settings)
