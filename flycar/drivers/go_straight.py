import numpy as np

class GoStraight:
    def reset(self):
        pass

    def act(self, observation):
        return np.array([0.5, 0.0, 0.0], dtype=np.float32)