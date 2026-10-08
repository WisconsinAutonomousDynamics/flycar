"""A driver that picks random actions. Useful for testing the environment.

Choosing a brand-new random action every 10 ms would just make the car twitch
on the spot, so this driver holds each random action for a little while.
"""

import numpy as np


class RandomDriver:
    """Holds a random throttle and steering for ``hold_s`` seconds, then re-draws.

    Parameters
    ----------
    seed : int
        Seed for this driver's own random generator. The driver re-seeds on
        every ``reset()``, so it produces the same actions in every episode.
    hold_s : float
        How long each random action is held [s].
    """

    def __init__(self, seed=0, hold_s=0.5):
        self.seed = seed
        self.hold_s = hold_s
        self.reset()

    def reset(self):
        self.rng = np.random.default_rng(self.seed)
        self.action = np.zeros(3, dtype=np.float32)
        self.next_change_time = 0.0

    def act(self, observation):
        time = float(observation["time"][0])
        if time >= self.next_change_time:
            throttle = self.rng.uniform(0.2, 1.0)
            steering = self.rng.uniform(-1.0, 1.0)
            brake = 0.0
            self.action = np.array([throttle, steering, brake], dtype=np.float32)
            self.next_change_time = time + self.hold_s
        return self.action
