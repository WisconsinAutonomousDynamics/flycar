"""Inertial measurements: speed, yaw rate and acceleration.

Everything here is derived from the chassis state that Chrono already knows,
so it works on every machine. (The Chrono sensor module also has an IMU with
noise models; swapping it in later would only touch this file.)

Output: float32 ``[speed_mps, yaw_rate_radps, accel_x, accel_y, accel_z]``.
The meaning and range of each entry is defined in ``flycar/env/spaces.py``.
"""

import numpy as np

from flycar.env.spaces import IMU_CHANNELS


class Imu:
    """Reads the chassis motion once per control step."""

    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.low = np.array([c.low for c in IMU_CHANNELS], dtype=np.float32)
        self.high = np.array([c.high for c in IMU_CHANNELS], dtype=np.float32)
        self.reset()

    def reset(self):
        """Remember the current velocity, so the first acceleration is not a spike."""
        self.previous_velocity = self.vehicle.velocity()

    def read(self, dt):
        """Return the IMU observation. ``dt`` is the time since the last read [s]."""
        rotation = self.vehicle.rotation_matrix()   # chassis axes expressed in the world
        velocity = self.vehicle.velocity()          # world frame

        # Acceleration = change of velocity over the control period.
        # (Averaging over the whole control step is far less noisy than the
        # instantaneous acceleration of a single physics step.)
        acceleration_world = (velocity - self.previous_velocity) / dt
        self.previous_velocity = velocity

        # World frame -> chassis frame. Gravity is not included.
        acceleration = rotation.T @ acceleration_world

        reading = np.array(
            [
                self.vehicle.forward_speed(),
                self.vehicle.yaw_rate(),
                acceleration[0],
                acceleration[1],
                acceleration[2],
            ],
            dtype=np.float32,
        )
        # Clip to the documented ranges so the observation always fits its space.
        return np.clip(reading, self.low, self.high)
