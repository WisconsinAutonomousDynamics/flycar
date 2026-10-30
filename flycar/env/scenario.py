"""Scenario files: load a YAML experiment description and fill in defaults.

A scenario is a plain YAML file (see ``scenarios/``). It describes the arena,
the obstacles, the lights, where the car spawns and how long an episode lasts.
Running a new experiment means writing a new YAML file, not editing code.

Randomization rule
------------------
Anywhere a scenario asks for a position or an angle (``x``, ``y``, ``z``,
``yaw_deg``) you may write either

* a single number  -> that exact value every episode, or
* ``[min, max]``   -> a new uniform random value on every ``reset()``.

This file must NOT import pychrono (the tools and tests read scenarios too).
"""

import copy
from pathlib import Path

import yaml

# Every setting a scenario can contain, with its default value.
# A scenario file only needs to list the settings it wants to change.
DEFAULTS = {
    "name": "unnamed",
    "description": "",
    "episode": {
        "max_time_s": 30.0,        # the episode is truncated after this much simulated time
        "success_radius_m": 2.0,   # reaching the target light within this distance ends the episode
    },
    "timing": {
        "control_period_s": 0.01,  # one env.step() = this much simulated time
        "physics_step_s": 0.001,   # Chrono integration step (must divide the control period)
    },
    "arena": {
        "size_m": [40.0, 40.0],    # extent along x and y; the arena is centred on the origin
        "walls": True,
        "wall_height_m": 1.5,
        "wall_thickness_m": 0.5,
    },
    "vehicle": {
        "max_speed_mps": 4.0,          # speed governor: throttle fades out near this speed
        "throttle_rate_per_s": 4.0,    # max change of the throttle command per second
        "steering_rate_per_s": 4.0,    # max change of the steering command per second
        "brake_rate_per_s": 10.0,      # max change of the brake command per second
    },
    "spawn": {
        "x": 0.0,
        "y": 0.0,
        "yaw_deg": 0.0,
        "min_target_distance_m": 0.0,  # re-draw the spawn if it starts closer than this to the target
    },
    "lighting": {
        "ambient": 0.3,   # light that reaches every surface (0 = pitch dark, 1 = very bright)
        "sky": 0.2,       # luminance seen when looking at empty sky
    },
    "lights": [],
    "obstacles": [],
    "eyes": {
        "backend": "auto",         # auto | sensor | raycast
        "resolution": [20, 20],    # rows, columns of each compound eye
        "fov_h_deg": 120.0,        # horizontal field of view of one eye
        "fov_v_deg": 60.0,         # vertical field of view of one eye
        "yaw_offset_deg": 50.0,    # each eye is turned outwards by this angle
        "update_period_s": 0.02,   # how often the eyes take a new picture
        "max_range_m": 80.0,       # rays that hit nothing within this distance see sky
        "shadows": True,           # raycast backend: test whether each surface point can see each light
        "camera_width_px": 240,    # sensor backend: raw image width (height follows from the FOVs)
        "mount_m": [1.0, 0.3, 0.75],   # eye position on the chassis: forward, sideways (mirrored), up
    },
}

# Defaults for one entry of the "lights" list.
LIGHT_DEFAULTS = {
    "x": 0.0,
    "y": 0.0,
    "z": 1.5,              # height of the lamp above the ground
    "intensity": 1.0,      # brightness (1 = saturates the eye up close)
    "falloff_m": 6.0,      # distance at which lit surfaces receive half the light
    "target": False,       # True for the light the car is supposed to reach
}

# Defaults for one entry of the "obstacles" list.
OBSTACLE_DEFAULTS = {
    "shape": "box",              # box | cylinder
    "size_m": [2.0, 2.0, 1.5],   # box: length, width, height. cylinder: [radius, height]
    "x": 0.0,
    "y": 0.0,
    "yaw_deg": 0.0,
    "count": 1,                  # make this many copies (each drawn separately)
    "clearance_m": 4.0,          # keep at least this far from the spawn point and the target
}


class ScenarioError(ValueError):
    """Raised when a scenario file contains something we do not understand."""


def _merge(defaults, overrides, where):
    """Return ``defaults`` updated with ``overrides``, checking the key names.

    A misspelled key (say ``max_tim_s``) would otherwise be silently ignored
    and the default used instead, which is a miserable bug to track down.
    """
    result = copy.deepcopy(defaults)
    for key, value in overrides.items():
        if key not in defaults:
            known = ", ".join(sorted(defaults))
            raise ScenarioError(f"Unknown setting '{key}' in {where}. Known settings: {known}")
        if isinstance(defaults[key], dict):
            if not isinstance(value, dict):
                raise ScenarioError(f"'{key}' in {where} must be a mapping (key: value pairs).")
            result[key] = _merge(defaults[key], value, f"{where}.{key}")
        else:
            result[key] = value
    return result


def load_scenario(source):
    """Load a scenario and fill in every default.

    ``source`` is a path to a YAML file, or an already-loaded dictionary.
    Returns a plain nested dictionary with every key from DEFAULTS present.
    """
    if isinstance(source, dict):
        raw = source
        where = "scenario"
    else:
        path = Path(source)
        if not path.exists():
            raise ScenarioError(f"Scenario file not found: {path}")
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        where = path.name
        raw.setdefault("name", path.stem)

    scenario = _merge(DEFAULTS, raw, where)

    # The list entries have their own defaults.
    scenario["lights"] = [
        _merge(LIGHT_DEFAULTS, light, f"{where}.lights[{i}]")
        for i, light in enumerate(scenario["lights"] or [])
    ]
    scenario["obstacles"] = [
        _merge(OBSTACLE_DEFAULTS, obstacle, f"{where}.obstacles[{i}]")
        for i, obstacle in enumerate(scenario["obstacles"] or [])
    ]

    _validate(scenario, where)
    return scenario


def _validate(scenario, where):
    """Catch the mistakes that would otherwise fail deep inside Chrono."""
    control = scenario["timing"]["control_period_s"]
    physics = scenario["timing"]["physics_step_s"]
    substeps = control / physics
    if physics <= 0 or control <= 0:
        raise ScenarioError(f"{where}: timing values must be positive.")
    if abs(substeps - round(substeps)) > 1e-6 or round(substeps) < 1:
        raise ScenarioError(
            f"{where}: control_period_s ({control}) must be a whole multiple of physics_step_s ({physics})."
        )

    eye_period = scenario["eyes"]["update_period_s"]
    eye_steps = eye_period / control
    if abs(eye_steps - round(eye_steps)) > 1e-6 or round(eye_steps) < 1:
        raise ScenarioError(
            f"{where}: eyes.update_period_s ({eye_period}) must be a whole multiple of control_period_s ({control})."
        )

    if scenario["eyes"]["backend"] not in ("auto", "sensor", "raycast"):
        raise ScenarioError(f"{where}: eyes.backend must be auto, sensor or raycast.")

    targets = [light for light in scenario["lights"] if light["target"]]
    if len(targets) > 1:
        raise ScenarioError(f"{where}: at most one light may have 'target: true'.")

    for obstacle in scenario["obstacles"]:
        if obstacle["shape"] not in ("box", "cylinder"):
            raise ScenarioError(f"{where}: obstacle shape must be 'box' or 'cylinder', got '{obstacle['shape']}'.")
        expected = 3 if obstacle["shape"] == "box" else 2
        if len(obstacle["size_m"]) != expected:
            raise ScenarioError(
                f"{where}: a {obstacle['shape']} obstacle needs {expected} numbers in size_m, got {obstacle['size_m']}."
            )

    #Here is an attempt at A3: 
    for obstacle in scenario["obstacles"]:
        x = obstacle["x"] if isinstance(obstacle["x"], list) else [obstacle["x"]]
        y = obstacle["y"] if isinstance(obstacle["y"], list) else [obstacle["y"]]
        if obstacle["shape"] == "box": 
            if max(x) > 20 - obstacle["size_m"][0]/2 or min(x) < -20 + obstacle["size_m"][0]/2 or max(y) > 20 - obstacle["size_m"][1]/2 or min(y) < -20 + obstacle["size_m"][1]/2: #since the arena is 40x40 and centered at the origin. 
                raise ScenarioError(f"{where}: obstacle at ({obstacle['x']}, {obstacle['y']}) is out of bounds of the arena.")
        else: 
            if max(x) > 20 - obstacle["size_m"][0] or min(x) < -20 + obstacle["size_m"][0] or max(y) > 20 - obstacle["size_m"][0] or min(y) < -20 + obstacle["size_m"][0]: #since the arena is 40x40 and centered at the origin. 
                raise ScenarioError(f"{where}: obstacle at ({obstacle['x']}, {obstacle['y']}) is out of bounds of the arena.")

    if scenario["lighting"]["ambient"] < 0.0 or scenario["lighting"]["ambient"] > 1.0:
        raise ScenarioError(f"{where}: lighting.ambient must be between 0 and 1, got {scenario['lighting']['ambient']}.") 

    if scenario["lighting"]["sky"] < 0.0 or scenario["lighting"]["sky"] > 1.0:
        raise ScenarioError(f"{where}: lighting.sky must be between 0 and 1, got {scenario['lighting']['sky']}.")  

    for light in scenario["lights"]:
        if light["target"] == True:
            x = light["x"] if isinstance(light["x"], list) else [light["x"]]
            y = light["y"] if isinstance(light["y"], list) else [light["y"]]
            if max(x) > 20 or min(x) < -20 or max(y) > 20 or min(y) < -20:
                raise ScenarioError(f"{where}: target lamp at ({light['x']}, {light['y']}) is out of bounds of the arena.")

import pytest
from flycar.env.scenario import load_scenario, ScenarioError

def test_obstacle_within_arena(): 
    """Check if the obstacle is within the arena bounds."""
    scenario = {"name": "test_obstacle_within_arena", "obstacles": [{"shape": "cylinder", "size_m": [1, 1], "x": -30, "y": -30}]}
    try: 
        load_scenario(scenario)
    except ScenarioError as e: 
        assert "out of bounds of arena" in str(e) 
    else: 
        assert False, "Test failed: no exception raised for obstacle out of bounds."

def test_light_is_valid(): 
    """Check if the light is intiated with valid values."""
    scenario = {"name": "test_invalid_light", "lighting": {"ambient": -1}}
    try: 
        load_scenario(scenario)
    except ScenarioError as e: 
         assert "lighting.ambient" in str(e) 
    else: 
         assert False, "Test failed: no exception raised for invalid ambient light parameter."

def test_sky_is_valid(): 
    """Check if the sky is intiated with valid values."""
    scenario = {"name": "test_invalid_sky", "lighting": {"sky": 2}}
    try: 
        load_scenario(scenario)
    except ScenarioError as e: 
        assert "lighting.sky" in str(e) 
    else: 
        assert False, "Test failed: no exception raised for invalid sky parameter."
   
def test_lamp_is_valid(): 
    """Check if the lamp is out of bounds."""
    scenario = {"name": "test_invalid_lamp", "lights": [{"target": True, "lamp": {"x": -21, "y": -21}}]}
    try: 
        load_scenario(scenario)
    except ScenarioError as e: 
        assert "out of bounds lamp" in str(e) 
    else: 
        assert False, "Test failed: no exception raised for invalid lamp location."

def sample(value, rng):
    """Turn a scenario value into a number.

    A number is returned unchanged. A ``[min, max]`` pair gives a uniform
    random number drawn from ``rng`` (a ``numpy.random.Generator``).
    """
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ScenarioError(f"A random range must be [min, max], got {value}.")
        low, high = float(value[0]), float(value[1])
        return float(rng.uniform(low, high))
    return float(value)
