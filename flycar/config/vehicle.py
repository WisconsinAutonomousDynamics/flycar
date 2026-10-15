"""Scenario settings owned by the vehicle & terrain team.

Sections: ``vehicle``, ``spawn``, ``terrain``.

Add a setting by adding a key (with a comment saying what it does and its
units) to a section below, and a check for it in ``validate``. Defaults must
reproduce the current behaviour exactly.
"""

from flycar.config import ScenarioError

SECTIONS = {
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
    # The ground. (Planned: patches, ramps, SCM soft soil.)
    "terrain": {
        "type": "rigid_flat",          # one flat rigid patch slightly larger than the arena
        "friction": 0.9,               # friction coefficient of the ground
    },
}

# Terrain types that are implemented. (Planned: patches, ramps, SCM soft soil.)
TERRAIN_TYPES = ("rigid_flat",)


def validate(scenario, where):
    """Checks for the vehicle, spawn and terrain sections."""
    terrain = scenario["terrain"]
    if terrain["type"] not in TERRAIN_TYPES:
        raise ScenarioError(f"{where}: terrain.type: {terrain['type']} is not implemented yet.")
    if not terrain["friction"] > 0:
        raise ScenarioError(f"{where}: terrain.friction must be positive, got {terrain['friction']}.")
