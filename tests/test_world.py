import pytest
import random
import copy

from flycar.env.scenario import load_scenario, ScenarioError
from flycar.env.world import sample_layout
rng = random.Random(26)

# ---- Tester for A3 ---------------------------------------------------------
def test_lamp_overlaps(): 
    """Check if the lamp overlaps with an obstacle."""
    obstacle = {"shape": "box", "size_m": [2.0, 2.0, 1.5], "x": 0, "y": 0, "yaw_deg": 0, "count": 1, "clearance_m": 0.0}
    lamp = {"x": 0, "y": 0, "z": 1.0, "intensity": 1.0, "falloff_m": 1.0,"target": True}
    scenario = copy.deepcopy(load_scenario("scenarios/empty_arena.yaml"))

    scenario["obstacles"] = [obstacle]
    scenario["lights"] = [lamp]


    with pytest.raises(ScenarioError, match="Could not place an obstacle clear"):
        sample_layout(scenario, rng)

def test_too_many_obstacles():
    """Check if there are too many obstacles in the arena."""
    scenario = copy.deepcopy(load_scenario("scenarios/empty_arena.yaml"))
    obstacle = {
        "shape": "box",
        "size_m": [2.0, 2.0, 1.5],
        "x": [-10.0, 10.0],
        "y": [-10.0, 10.0],
        "yaw_deg": [0.0, 90.0],
        "count": 5000,
        "clearance_m": 0.0
    }

    scenario["obstacles"] = [obstacle]

    with pytest.raises(ScenarioError, match="Too many obstacles"):
        sample_layout(scenario, random.Random(26))