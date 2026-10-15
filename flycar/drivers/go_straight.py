"""A driver that drives straight ahead at half throttle. The smallest possible driver.

It ignores the observation completely. It exists to show the Driver interface:
``reset()`` once per episode, then ``act(observation)`` every control step.
"""

import numpy as np


class GoStraight:
    def reset(self):
        pass                      # called at the start of each episode

    def act(self, observation):   # called every 10 ms of simulated time
        return np.array([0.5, 0.0, 0.0], dtype=np.float32)
