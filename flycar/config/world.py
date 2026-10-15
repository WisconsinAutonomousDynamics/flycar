"""Scenario settings owned by the world team.

Sections: ``arena``, ``lighting``, ``lights`` (a list), ``obstacles`` (a list).

Add a setting by adding a key (with a comment saying what it does and its
units) to a section below, and a check for it in ``validate``. Defaults must
reproduce the current behaviour exactly.
"""

from flycar.config import ScenarioError

SECTIONS = {
    "arena": {
        "size_m": [40.0, 40.0],    # extent along x and y; the arena is centred on the origin
        "walls": True,
        "wall_height_m": 1.5,
        "wall_thickness_m": 0.5,
    },
    "lighting": {
        "ambient": 0.3,   # light that reaches every surface (0 = pitch dark, 1 = very bright)
        "sky": 0.2,       # luminance seen when looking at empty sky
    },
    "lights": [],
    "obstacles": [],
}

ITEM_DEFAULTS = {
    # Defaults for one entry of the "lights" list.
    "lights": {
        "x": 0.0,
        "y": 0.0,
        "z": 1.5,              # height of the lamp above the ground
        "intensity": 1.0,      # brightness (1 = saturates the eye up close)
        "falloff_m": 6.0,      # distance at which lit surfaces receive half the light
        "target": False,       # True for the light the car is supposed to reach
    },
    # Defaults for one entry of the "obstacles" list.
    "obstacles": {
        "shape": "box",              # box | cylinder
        "size_m": [2.0, 2.0, 1.5],   # box: length, width, height. cylinder: [radius, height]
        "x": 0.0,
        "y": 0.0,
        "yaw_deg": 0.0,
        "count": 1,                  # make this many copies (each drawn separately)
        "clearance_m": 4.0,          # keep at least this far from the spawn point and the target
    },
}


def validate(scenario, where):
    """Checks for the arena, lighting, lights and obstacles sections."""
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
