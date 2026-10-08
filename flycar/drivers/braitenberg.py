"""A Braitenberg vehicle: steer towards the light.

Valentino Braitenberg's thought experiment ("Vehicles", 1984): wire two light
sensors straight to the steering, and the machine appears to "want" the light,
with no map, no memory and no plan.

    brighter on the left   ->  steer left
    brighter on the right  ->  steer right
    equally bright         ->  go straight

This is the phototaxis baseline for the project. It uses ONLY the two eye
images. It is also the simplest possible example of a brain: a later
connectome model replaces the few lines of arithmetic in ``act`` and nothing
else.

Why "the brightest pixels" and not the average?
-----------------------------------------------
A lamp is small and bright. A wall lit by that lamp is large and dim. Averaged
over a whole eye, the big dim wall wins, and the car drives into the wall
instead of to the lamp. Looking only at the brightest few pixels of each eye
asks the right question: "which eye sees the brightest *thing*?"
(Try ``top_pixels=400``, which is the plain average, and watch it fail.)
"""

import numpy as np


class BraitenbergDriver:
    """Phototaxis from the left/right difference in eye brightness.

    Parameters
    ----------
    gain : float
        How hard to steer for a given left/right imbalance. Too low and the
        car turns lazily and misses; too high and it weaves.
    cruise_throttle : float
        Constant throttle, 0 to 1.
    top_pixels : int
        Each eye reports the average of its this-many brightest pixels.
    """

    def __init__(self, gain=6.0, cruise_throttle=0.5, top_pixels=10):
        self.gain = gain
        self.cruise_throttle = cruise_throttle
        self.top_pixels = top_pixels

    def reset(self):
        pass   # no memory, so nothing to forget

    def brightness(self, eye):
        """One number per eye: the average of its brightest pixels."""
        pixels = np.sort(eye.ravel())           # dimmest ... brightest
        return float(pixels[-self.top_pixels:].mean())

    def act(self, observation):
        left = self.brightness(observation["eye_left"])
        right = self.brightness(observation["eye_right"])

        # Relative difference, between -1 (all light on the right) and +1 (all
        # on the left). Dividing by the total makes the response the same in a
        # dim arena as in a bright one.
        imbalance = (left - right) / (left + right + 1e-6)

        # Positive steering turns left, i.e. towards the brighter left eye.
        steering = float(np.clip(self.gain * imbalance, -1.0, 1.0))

        return np.array([self.cruise_throttle, steering, 0.0], dtype=np.float32)
