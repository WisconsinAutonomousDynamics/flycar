"""Action and observation definitions: the single source of truth.

Everything a controller is allowed to know about the Fly Car interface lives in
this file: what an action is, what an observation is, and what every number
means (order, range, units).

* The environment builds its Gymnasium spaces from here.
* The tests check real observations against here.
* The tables in ``docs/API.md`` are generated from here
  (run ``python scripts/gen_api_docs.py`` after you change anything).

This file must NOT import pychrono. Drivers (and, later, the brain model) import
it to get the index constants below, and they must run without Chrono.
"""

from dataclasses import dataclass

import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class Channel:
    """One number inside an action or observation vector."""

    name: str
    low: float
    high: float
    units: str
    description: str


# ----------------------------------------------------------------------------
# Action:  action = [throttle, steering, brake]
# ----------------------------------------------------------------------------

ACTION_CHANNELS = (
    Channel("throttle", 0.0, 1.0, "fraction", "0 = foot off the pedal, 1 = full throttle."),
    Channel("steering", -1.0, 1.0, "fraction", "-1 = full right, 0 = straight, +1 = full left."),
    Channel("brake", 0.0, 1.0, "fraction", "0 = no braking, 1 = full braking."),
)

# Index constants, so code can say action[STEERING] instead of action[1].
THROTTLE, STEERING, BRAKE = 0, 1, 2


# ----------------------------------------------------------------------------
# Observation: a dictionary. Every entry is a float32 array with a fixed shape.
# ----------------------------------------------------------------------------

# Compound-eye resolution: rows (top to bottom) x columns (left to right).
# A scenario may override it (eyes.resolution), but 20 x 20 is the default.
DEFAULT_EYE_SHAPE = (20, 20)

# obs["imu"] = [speed, yaw_rate, accel_x, accel_y, accel_z]
# Values outside these ranges are clipped, so an observation is always inside
# the observation space.
IMU_CHANNELS = (
    Channel("speed_mps", -30.0, 30.0, "m/s", "Forward speed of the chassis. Negative when rolling backwards."),
    Channel("yaw_rate_radps", -20.0, 20.0, "rad/s", "Turning rate about the chassis up axis. Positive = turning left."),
    Channel("accel_x", -100.0, 100.0, "m/s^2", "Chassis acceleration, forward component (gravity not included)."),
    Channel("accel_y", -100.0, 100.0, "m/s^2", "Chassis acceleration, leftward component (gravity not included)."),
    Channel("accel_z", -100.0, 100.0, "m/s^2", "Chassis acceleration, upward component (gravity not included)."),
)
IMU_SPEED, IMU_YAW_RATE, IMU_ACCEL_X, IMU_ACCEL_Y, IMU_ACCEL_Z = 0, 1, 2, 3, 4

# obs["contact"] = [left, front, right]
CONTACT_CHANNELS = (
    Channel("left", 0.0, 1.0, "flag", "1 if the left side of the chassis touched a wall or obstacle during this control step."),
    Channel("front", 0.0, 1.0, "flag", "1 if the front of the chassis touched a wall or obstacle during this control step."),
    Channel("right", 0.0, 1.0, "flag", "1 if the right side of the chassis touched a wall or obstacle during this control step."),
)
CONTACT_LEFT, CONTACT_FRONT, CONTACT_RIGHT = 0, 1, 2

# obs["time"] = [t]
TIME_CHANNELS = (
    Channel("time", 0.0, float("inf"), "s", "Simulation time since the last reset()."),
)

EYE_DESCRIPTION = (
    "Luminance seen by one compound eye: 0 = black, 1 = saturated. "
    "Row 0 is the top of the view, column 0 is its left edge."
)


# ----------------------------------------------------------------------------
# Info: extra data returned next to the observation.
#
# Info is for logging, plotting, debugging and scoring. It contains ground
# truth that a real fly would not have, so a Driver never receives it.
# ----------------------------------------------------------------------------

# (key, type, units, description)
INFO_FIELDS = (
    ("pose", "float64 array (3,)", "m, m, rad", "Ground-truth chassis pose [x, y, yaw] in the world frame. yaw = 0 faces +x, positive is counter-clockwise seen from above."),
    ("target_distance", "float", "m", "Horizontal distance from the chassis to the target light. NaN if the scenario has no target."),
    ("target_bearing", "float", "rad", "Direction of the target relative to the car's heading, in (-pi, pi]. 0 = dead ahead, positive = to the left. NaN if there is no target."),
    ("eye_backend", "str", "-", 'Which eye implementation is running: "sensor" (GPU camera) or "raycast" (CPU).'),
    ("camera_left", "uint8 array (h, w, 3) or None", "RGB 0-255", "Raw left-eye camera image. None with the raycast backend."),
    ("camera_right", "uint8 array (h, w, 3) or None", "RGB 0-255", "Raw right-eye camera image. None with the raycast backend."),
    ("applied_action", "float32 array (3,)", "fraction", "The [throttle, steering, brake] that actually reached the vehicle after saturation and rate limiting."),
    ("time", "float", "s", "Simulation time since reset (same as obs['time'], as a Python float)."),
    ("step", "int", "-", "Number of control steps since reset."),
    ("success", "bool", "-", "True once the car is within the scenario's success radius of the target."),
    ("world", "dict", "-", "Only returned by reset(): the layout of this episode (arena size, walls, obstacles, lights, target), as plain Python data for plotting."),
)


# ----------------------------------------------------------------------------
# Gymnasium spaces
# ----------------------------------------------------------------------------

def _box_from_channels(channels):
    """Build a 1-D float32 Box whose bounds come from a tuple of Channels."""
    low = np.array([c.low for c in channels], dtype=np.float32)
    high = np.array([c.high for c in channels], dtype=np.float32)
    return spaces.Box(low=low, high=high, dtype=np.float32)


def make_action_space():
    """Box(low=[0, -1, 0], high=[1, 1, 1]) ordered [throttle, steering, brake]."""
    return _box_from_channels(ACTION_CHANNELS)


def make_observation_space(eye_shape=DEFAULT_EYE_SHAPE):
    """The Dict observation space. ``eye_shape`` is (rows, columns) per eye."""
    eye_shape = (int(eye_shape[0]), int(eye_shape[1]))
    return spaces.Dict(
        {
            "eye_left": spaces.Box(low=0.0, high=1.0, shape=eye_shape, dtype=np.float32),
            "eye_right": spaces.Box(low=0.0, high=1.0, shape=eye_shape, dtype=np.float32),
            "imu": _box_from_channels(IMU_CHANNELS),
            "contact": _box_from_channels(CONTACT_CHANNELS),
            "time": _box_from_channels(TIME_CHANNELS),
        }
    )


# ----------------------------------------------------------------------------
# Documentation generator (used by scripts/gen_api_docs.py and the tests)
# ----------------------------------------------------------------------------

def _channel_table(channels):
    """Markdown table with one row per channel."""
    lines = [
        "| Index | Name | Range | Units | Meaning |",
        "|---|---|---|---|---|",
    ]
    for index, c in enumerate(channels):
        lines.append(f"| {index} | `{c.name}` | [{c.low:g}, {c.high:g}] | {c.units} | {c.description} |")
    return lines


def api_markdown():
    """Return the generated part of docs/API.md as one string."""
    rows, cols = DEFAULT_EYE_SHAPE
    lines = []

    lines.append("### Action")
    lines.append("")
    lines.append(f"`Box(shape=({len(ACTION_CHANNELS)},), dtype=float32)`, ordered `[throttle, steering, brake]`.")
    lines.append("")
    lines.extend(_channel_table(ACTION_CHANNELS))
    lines.append("")

    lines.append("### Observation")
    lines.append("")
    lines.append("A `Dict`. Every value is a `float32` NumPy array with a fixed shape.")
    lines.append("")
    lines.append("| Key | Shape | Range | Units | Meaning |")
    lines.append("|---|---|---|---|---|")
    lines.append(f"| `eye_left` | ({rows}, {cols}) | [0, 1] | luminance | Left eye. {EYE_DESCRIPTION} |")
    lines.append(f"| `eye_right` | ({rows}, {cols}) | [0, 1] | luminance | Right eye. {EYE_DESCRIPTION} |")
    lines.append(f"| `imu` | ({len(IMU_CHANNELS)},) | see below | see below | Motion of the chassis. |")
    lines.append(f"| `contact` | ({len(CONTACT_CHANNELS)},) | [0, 1] | flag | Touch sensors `[left, front, right]`. |")
    lines.append("| `time` | (1,) | [0, inf) | s | Simulation time since the last `reset()`. |")
    lines.append("")
    lines.append(f"The eye shape is (rows, columns). {rows} x {cols} is the default; a scenario can change it with `eyes.resolution`.")
    lines.append("")

    lines.append("#### `imu`")
    lines.append("")
    lines.extend(_channel_table(IMU_CHANNELS))
    lines.append("")

    lines.append("#### `contact`")
    lines.append("")
    lines.extend(_channel_table(CONTACT_CHANNELS))
    lines.append("")

    lines.append("### Info")
    lines.append("")
    lines.append("Ground truth and debugging data. Drivers never see it.")
    lines.append("")
    lines.append("| Key | Type | Units | Meaning |")
    lines.append("|---|---|---|---|")
    for key, type_text, units, description in INFO_FIELDS:
        lines.append(f"| `{key}` | {type_text} | {units} | {description} |")

    return "\n".join(lines) + "\n"
