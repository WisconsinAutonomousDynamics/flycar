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
        "tire_model": "tmeasy",        # tire model to use
        "steering_gain": 1.0,          # steering multiplier; output is clamped to [-1, 1]
        "throttle_curve_exponent": 1.0,# default exponent for the throttle curve (1 = linear)
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
TIRE_MODELS = ("tmeasy", "rigid")


def validate(scenario, where):
    """Checks for the vehicle, spawn and terrain sections."""
    terrain = scenario["terrain"]
    if terrain["type"] not in TERRAIN_TYPES:
        raise ScenarioError(f"{where}: terrain.type: {terrain['type']} is not implemented yet.")
    if not terrain["friction"] > 0:
        raise ScenarioError(f"{where}: terrain.friction must be positive, got {terrain['friction']}.")
    vehicle = scenario["vehicle"]
    if vehicle["tire_model"] not in TIRE_MODELS:
        raise ScenarioError(f"{where}: vehicle.tire_model must be one of {TIRE_MODELS}, got {vehicle['tire_model']!r}.")
    if vehicle["steering_gain"] < 0:
        raise ScenarioError(f"{where}: vehicle.steering_gain must be non-negative, got {vehicle['steering_gain']}.")
    if vehicle["throttle_curve_exponent"] <= 0:
        raise ScenarioError(f"{where}: vehicle.throttle_curve_exponent must be positive, got {vehicle['throttle_curve_exponent']}.")
