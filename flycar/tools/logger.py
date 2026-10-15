"""Per-step logging to a Parquet file.

One episode becomes one ``.parquet`` file with one row per control step.
Parquet is a compact table format that pandas reads in one line::

    import pandas as pd
    table = pd.read_parquet("logs/light_seek_braitenberg_seed0.parquet")
    table.plot(x="time", y="speed")

Columns
-------
step, time                              control step number, simulation time [s]
action_throttle / _steering / _brake    what the driver asked for
applied_throttle / _steering / _brake   what reached the car (after rate limiting)
x, y, yaw                               ground-truth pose [m, m, rad]
speed, yaw_rate, accel_x/y/z            the IMU observation
contact_left / _front / _right          the contact observation
eye_left_mean, eye_right_mean           average brightness of each eye (the "summary")
eye_left, eye_right                     the full eye image, flattened row by row.
                                        Stored every ``eye_every`` steps, empty otherwise.
target_distance, target_bearing         ground truth [m, rad]
success                                 True once the target is reached
extras                                  ONLY with log_extras=True (run.py --log-extras):
                                        info["extras"] as a JSON string. Left out by
                                        default, so logs stay the same size.

The world layout (walls, obstacles, lights) and run details are stored in the
file's metadata, so ``replay.py`` can redraw the arena from the log alone.
"""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from flycar.env.spaces import (
    BRAKE, CONTACT_FRONT, CONTACT_LEFT, CONTACT_RIGHT,
    IMU_ACCEL_X, IMU_ACCEL_Y, IMU_ACCEL_Z, IMU_SPEED, IMU_YAW_RATE,
    STEERING, THROTTLE,
)

METADATA_KEY = b"flycar"


class EpisodeLogger:
    """Collects one row per control step and writes a Parquet file on close().

    Parameters
    ----------
    path : str or Path
        Where to write the ``.parquet`` file.
    world : dict
        ``info["world"]`` from ``env.reset()``.
    details : dict
        Anything else worth remembering (scenario name, driver, seed ...).
        Must be plain data that JSON can store.
    eye_every : int
        Store the full eye images every this many steps. They are by far the
        biggest part of a log, and at 10 per second they replay smoothly.
    log_extras : bool
        Also store ``info["extras"]`` (the teams' unstable debug data) in an
        ``extras`` column, as one JSON string per step. Off by default.
    """

    def __init__(self, path, world, details=None, eye_every=10, log_extras=False):
        self.path = Path(path)
        self.world = world
        self.details = dict(details or {})
        self.eye_every = int(eye_every)
        self.log_extras = bool(log_extras)
        self.rows = []

    def log(self, action, observation, info):
        """Record one control step. Call it right after ``env.step``."""
        action = np.asarray(action, dtype=np.float32)
        applied = info["applied_action"]
        imu = observation["imu"]
        contact = observation["contact"]
        step = int(info["step"])

        keep_eyes = step % self.eye_every == 0
        row = {
            "step": step,
            "time": float(info["time"]),
            "action_throttle": float(action[THROTTLE]),
            "action_steering": float(action[STEERING]),
            "action_brake": float(action[BRAKE]),
            "applied_throttle": float(applied[THROTTLE]),
            "applied_steering": float(applied[STEERING]),
            "applied_brake": float(applied[BRAKE]),
            "x": float(info["pose"][0]),
            "y": float(info["pose"][1]),
            "yaw": float(info["pose"][2]),
            "speed": float(imu[IMU_SPEED]),
            "yaw_rate": float(imu[IMU_YAW_RATE]),
            "accel_x": float(imu[IMU_ACCEL_X]),
            "accel_y": float(imu[IMU_ACCEL_Y]),
            "accel_z": float(imu[IMU_ACCEL_Z]),
            "contact_left": float(contact[CONTACT_LEFT]),
            "contact_front": float(contact[CONTACT_FRONT]),
            "contact_right": float(contact[CONTACT_RIGHT]),
            "eye_left_mean": float(observation["eye_left"].mean()),
            "eye_right_mean": float(observation["eye_right"].mean()),
            "eye_left": observation["eye_left"].ravel().tolist() if keep_eyes else None,
            "eye_right": observation["eye_right"].ravel().tolist() if keep_eyes else None,
            "target_distance": float(info["target_distance"]),
            "target_bearing": float(info["target_bearing"]),
            "success": bool(info["success"]),
        }
        if self.log_extras:
            row["extras"] = json.dumps(info.get("extras", {}), default=_plain)
        self.rows.append(row)

        # Remember the eye resolution so replay can un-flatten the images.
        self.details["eye_shape"] = list(observation["eye_left"].shape)
        self.details["eye_backend"] = info["eye_backend"]

    def close(self):
        """Write the file and return its path."""
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Turn the list of rows into one list per column.
        columns = {}
        if self.rows:
            for name in self.rows[0]:
                columns[name] = [row[name] for row in self.rows]

        # The eye images are lists of float32; tell Arrow so it stores them compactly.
        arrays = {}
        for name, values in columns.items():
            if name in ("eye_left", "eye_right"):
                arrays[name] = pa.array(values, type=pa.list_(pa.float32()))
            else:
                arrays[name] = pa.array(values)
        table = pa.table(arrays)

        metadata = {"world": self.world, "details": self.details}
        table = table.replace_schema_metadata({METADATA_KEY: json.dumps(metadata).encode("utf-8")})
        pq.write_table(table, self.path)
        return self.path


def _plain(value):
    """Turn NumPy values inside info["extras"] into things JSON can store."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def load_log(path):
    """Read a log written by EpisodeLogger.

    Returns ``(table, world, details)``: a pandas DataFrame with one row per
    control step, the world layout, and the run details.
    """
    arrow_table = pq.read_table(path)
    raw = (arrow_table.schema.metadata or {}).get(METADATA_KEY)
    if raw is None:
        raise ValueError(f"{path} is not a Fly Car log (no 'flycar' metadata).")
    metadata = json.loads(raw.decode("utf-8"))
    return arrow_table.to_pandas(), metadata["world"], metadata["details"]
