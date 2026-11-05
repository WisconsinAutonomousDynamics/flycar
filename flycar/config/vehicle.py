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
    # The ground. (Planned: ramps, SCM soft soil.)
    "terrain": {
        "type": "rigid_flat",          # rigid_flat: one flat rigid patch slightly larger than the arena
                                       # rigid_patches: the same, plus the friction patches below
        "friction": 0.9,               # friction coefficient of the ground
        "patches": [],                 # friction patches ("ice", "mud"), only with type rigid_patches.
                                       # Each: {x, y, size_m: [sx, sy], friction}: centre [m, m],
                                       # size along x and y [m], friction coefficient [-].
                                       # Flat at z = 0 like the ground; fixed positions (no ranges).
    },
}

# Terrain types that are implemented. (Planned: ramps, SCM soft soil.)
TERRAIN_TYPES = ("rigid_flat", "rigid_patches")
TIRE_MODELS = ("tmeasy", "rigid")

# The settings of one entry of terrain.patches (all required).
PATCH_KEYS = ("x", "y", "size_m", "friction")


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
    _validate_patches(scenario, where)


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_patches(scenario, where):
    """Checks for terrain.patches (friction patches, terrain.type rigid_patches)."""
    terrain = scenario["terrain"]
    patches = terrain["patches"]
    if not isinstance(patches, list):
        raise ScenarioError(f"{where}: terrain.patches must be a list, got {patches!r}.")
    if patches and terrain["type"] != "rigid_patches":
        raise ScenarioError(
            f"{where}: terrain.patches needs terrain.type: rigid_patches (got type {terrain['type']})."
        )

    arena_x, arena_y = scenario["arena"]["size_m"]
    boxes = []
    for i, patch in enumerate(patches):
        name = f"terrain.patches[{i}]"
        if not isinstance(patch, dict):
            raise ScenarioError(f"{where}: {name} must be a mapping with keys {', '.join(PATCH_KEYS)}.")
        unknown = sorted(set(patch) - set(PATCH_KEYS))
        if unknown:
            raise ScenarioError(
                f"{where}: unknown setting '{unknown[0]}' in {name}. Known settings: {', '.join(PATCH_KEYS)}"
            )
        missing = [key for key in PATCH_KEYS if key not in patch]
        if missing:
            raise ScenarioError(f"{where}: {name} is missing '{missing[0]}'.")
        for key in ("x", "y"):
            if not _is_number(patch[key]):
                raise ScenarioError(
                    f"{where}: {name}.{key} must be a number (patch positions are fixed, "
                    f"random ranges are not supported), got {patch[key]!r}."
                )
        size = patch["size_m"]
        if not (isinstance(size, (list, tuple)) and len(size) == 2 and all(_is_number(v) and v > 0 for v in size)):
            raise ScenarioError(f"{where}: {name}.size_m must be two positive numbers [sx, sy], got {size!r}.")
        if not (_is_number(patch["friction"]) and patch["friction"] > 0):
            raise ScenarioError(f"{where}: {name}.friction must be positive, got {patch['friction']!r}.")

        # The patch must lie inside the arena (the ground under the walls is plain ground).
        x_min, x_max = patch["x"] - size[0] / 2.0, patch["x"] + size[0] / 2.0
        y_min, y_max = patch["y"] - size[1] / 2.0, patch["y"] + size[1] / 2.0
        if x_min < -arena_x / 2.0 or x_max > arena_x / 2.0 or y_min < -arena_y / 2.0 or y_max > arena_y / 2.0:
            raise ScenarioError(f"{where}: {name} reaches outside the {arena_x} m x {arena_y} m arena.")

        # Patches may touch but not overlap: where two overlapped, the ground
        # would silently use only one of the two frictions.
        for j, (ox_min, ox_max, oy_min, oy_max) in enumerate(boxes):
            if x_min < ox_max and ox_min < x_max and y_min < oy_max and oy_min < y_max:
                raise ScenarioError(f"{where}: {name} overlaps terrain.patches[{j}]; patches must not overlap.")
        boxes.append((x_min, x_max, y_min, y_max))
