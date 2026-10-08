"""Drive the car yourself.

    python run.py --scenario scenarios/empty_arena.yaml --driver keyboard --dashboard

Keys (click the dashboard window first so it has keyboard focus):

    W or Up      throttle
    S or Down    brake
    A or Left    steer left
    D or Right   steer right

This driver does not read the keyboard itself. Whoever owns a window (the
dashboard) calls ``on_key_press`` and ``on_key_release``, and the driver only
remembers which keys are currently down. That keeps this file free of any
window-system or simulator code.
"""

import numpy as np

THROTTLE_KEYS = {"w", "up"}
BRAKE_KEYS = {"s", "down"}
LEFT_KEYS = {"a", "left"}
RIGHT_KEYS = {"d", "right"}


class KeyboardDriver:
    """Turns held keys into ``[throttle, steering, brake]``."""

    # run.py checks this flag and insists on a window that can deliver key events.
    needs_key_events = True

    def __init__(self):
        self.reset()

    def reset(self):
        self.keys_down = set()

    def on_key_press(self, key):
        """Called by the window when a key goes down. ``key`` is a name like 'w' or 'left'."""
        if key:
            self.keys_down.add(key.lower())

    def on_key_release(self, key):
        """Called by the window when a key comes back up."""
        if key:
            self.keys_down.discard(key.lower())

    def act(self, observation):
        throttle = 1.0 if self.keys_down & THROTTLE_KEYS else 0.0
        brake = 1.0 if self.keys_down & BRAKE_KEYS else 0.0

        steering = 0.0
        if self.keys_down & LEFT_KEYS:
            steering += 1.0   # positive steering = left
        if self.keys_down & RIGHT_KEYS:
            steering -= 1.0

        # The keys are all-or-nothing, but the car's rate limiter (vehicle.py)
        # turns these jumps into smooth pedal and wheel movements.
        return np.array([throttle, steering, brake], dtype=np.float32)
