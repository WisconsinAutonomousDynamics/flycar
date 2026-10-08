"""The Driver interface: the seam where a brain plugs in.

A driver is anything that turns an observation into an action. The random
driver, the Braitenberg vehicle, a person at a keyboard and (later) a fly
connectome model all look the same from the environment's point of view::

    driver.reset()                      # once, at the start of every episode
    action = driver.act(observation)    # once per control step (every 10 ms)

Rules for drivers
-----------------
* A driver sees ONLY the observation dictionary. Never ``info`` (that is
  ground truth), never the environment object.
* A driver must not import ``pychrono``. A test enforces this. It keeps
  controllers independent of the simulator, so the same code could later drive
  a different simulator or a real robot.
* ``act`` may do as much internal work as it likes. A brain model will
  typically run many of its own small time steps inside one ``act`` call
  (for example 100 steps of 0.1 ms to cover the 10 ms control period).

See ``docs/API.md`` for the full observation and action reference.
"""

from typing import Protocol, runtime_checkable

import numpy as np

# How long one control step lasts by default [s]. A brain model can use this
# to decide how many internal sub-steps to run per act() call.
DEFAULT_CONTROL_PERIOD_S = 0.01


@runtime_checkable
class Driver(Protocol):
    """Anything with these two methods is a driver. No inheritance needed."""

    def reset(self) -> None:
        """Forget everything from the previous episode."""
        ...

    def act(self, observation: dict) -> np.ndarray:
        """Return the action ``[throttle, steering, brake]`` for this control step.

        ``observation`` is the dictionary described in ``docs/API.md``:
        ``eye_left``, ``eye_right``, ``imu``, ``contact`` and ``time``.
        """
        ...
