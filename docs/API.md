# Fly Car API

This is the contract between the simulation and whatever drives the car. If
you are writing a controller (or, later, wiring in the brain model), this page
is everything you need. You should not have to read the simulator code.

**The API is meant to stay stable.** Controllers written today should still run
when the brain model arrives. If you need to change an action or observation,
change `flycar/env/spaces.py` (the single source of truth), regenerate this
page, and tell the team.

## The loop

```python
from flycar.env.core import FlyCarEnv

env = FlyCarEnv("scenarios/light_seek.yaml")
observation, info = env.reset(seed=0)

done = False
while not done:
    action = [0.5, 0.0, 0.0]          # [throttle, steering, brake]
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()
```

`FlyCarEnv` is a standard [Gymnasium](https://gymnasium.farama.org/) environment.

| Call | Returns | Notes |
|---|---|---|
| `FlyCarEnv(scenario, render_mode=None, eye_backend=None)` | the environment | `scenario` is a YAML path. `render_mode="human"` opens a 3D window. `eye_backend` is `"auto"`, `"sensor"` or `"raycast"`. |
| `env.reset(seed=None, options=None)` | `(observation, info)` | Builds a fresh world. The same seed gives the same layout and, with the same actions, the same trajectory. |
| `env.step(action)` | `(observation, reward, terminated, truncated, info)` | Advances one control period (10 ms by default). |
| `env.close()` | | Closes any window. |

- `reward` is always `0.0` for now. It exists only so the environment is Gymnasium-compatible.
- `terminated` is `True` when the car gets within the scenario's `success_radius_m` of the target light.
- `truncated` is `True` when the episode reaches the scenario's `max_time_s`.

<!-- BEGIN GENERATED FROM flycar/env/spaces.py (do not edit by hand) -->

### Action

`Box(shape=(3,), dtype=float32)`, ordered `[throttle, steering, brake]`.

| Index | Name | Range | Units | Meaning |
|---|---|---|---|---|
| 0 | `throttle` | [0, 1] | fraction | 0 = foot off the pedal, 1 = full throttle. |
| 1 | `steering` | [-1, 1] | fraction | -1 = full right, 0 = straight, +1 = full left. |
| 2 | `brake` | [0, 1] | fraction | 0 = no braking, 1 = full braking. |

### Observation

A `Dict`. Every value is a `float32` NumPy array with a fixed shape.

| Key | Shape | Range | Units | Meaning |
|---|---|---|---|---|
| `eye_left` | (20, 20) | [0, 1] | luminance | Left eye. Luminance seen by one compound eye: 0 = black, 1 = saturated. Row 0 is the top of the view, column 0 is its left edge. |
| `eye_right` | (20, 20) | [0, 1] | luminance | Right eye. Luminance seen by one compound eye: 0 = black, 1 = saturated. Row 0 is the top of the view, column 0 is its left edge. |
| `imu` | (5,) | see below | see below | Motion of the chassis. |
| `contact` | (3,) | [0, 1] | flag | Touch sensors `[left, front, right]`. |
| `time` | (1,) | [0, inf) | s | Simulation time since the last `reset()`. |

The eye shape is (rows, columns). 20 x 20 is the default; a scenario can change it with `eyes.resolution`.

#### `imu`

| Index | Name | Range | Units | Meaning |
|---|---|---|---|---|
| 0 | `speed_mps` | [-30, 30] | m/s | Forward speed of the chassis. Negative when rolling backwards. |
| 1 | `yaw_rate_radps` | [-20, 20] | rad/s | Turning rate about the chassis up axis. Positive = turning left. |
| 2 | `accel_x` | [-100, 100] | m/s^2 | Chassis acceleration, forward component (gravity not included). |
| 3 | `accel_y` | [-100, 100] | m/s^2 | Chassis acceleration, leftward component (gravity not included). |
| 4 | `accel_z` | [-100, 100] | m/s^2 | Chassis acceleration, upward component (gravity not included). |

#### `contact`

| Index | Name | Range | Units | Meaning |
|---|---|---|---|---|
| 0 | `left` | [0, 1] | flag | 1 if the left side of the chassis touched a wall or obstacle during this control step. |
| 1 | `front` | [0, 1] | flag | 1 if the front of the chassis touched a wall or obstacle during this control step. |
| 2 | `right` | [0, 1] | flag | 1 if the right side of the chassis touched a wall or obstacle during this control step. |

### Info

Ground truth and debugging data. Drivers never see it.

| Key | Type | Units | Meaning |
|---|---|---|---|
| `pose` | float64 array (3,) | m, m, rad | Ground-truth chassis pose [x, y, yaw] in the world frame. yaw = 0 faces +x, positive is counter-clockwise seen from above. |
| `target_distance` | float | m | Horizontal distance from the chassis to the target light. NaN if the scenario has no target. |
| `target_bearing` | float | rad | Direction of the target relative to the car's heading, in (-pi, pi]. 0 = dead ahead, positive = to the left. NaN if there is no target. |
| `eye_backend` | str | - | Which eye implementation is running: "sensor" (GPU camera) or "raycast" (CPU). |
| `camera_left` | uint8 array (h, w, 3) or None | RGB 0-255 | Raw left-eye camera image. None with the raycast backend. |
| `camera_right` | uint8 array (h, w, 3) or None | RGB 0-255 | Raw right-eye camera image. None with the raycast backend. |
| `applied_action` | float32 array (3,) | fraction | The [throttle, steering, brake] that actually reached the vehicle after saturation and rate limiting. |
| `time` | float | s | Simulation time since reset (same as obs['time'], as a Python float). |
| `step` | int | - | Number of control steps since reset. |
| `success` | bool | - | True once the car is within the scenario's success radius of the target. |
| `extras` | dict | - | Unstable debug data, not part of the API; drivers never see it. Keyed by team slug (vehicle, world, vision, core), then by component: e.g. extras['core']['contact']. Empty when no component provides any. |
| `world` | dict | - | Only returned by reset(): the layout of this episode (arena size, walls, obstacles, lights, target), as plain Python data for plotting. |

<!-- END GENERATED -->

## Conventions

**Frames.** The world frame has z up. The chassis frame has **x forward, y
left, z up**, with its origin midway between the axles. `yaw = 0` means the car
faces world +x, and yaw increases counter-clockwise seen from above.

**Signs.** Positive steering, positive yaw rate and positive bearing all mean
**left**.

**Units.** SI everywhere: metres, seconds, radians, m/s, m/s². The one
exception is scenario files, which use degrees for angles because people write
them by hand.

**Timing.** One `step()` is one control period (`timing.control_period_s`,
default 10 ms). The action is held constant for that period while Chrono takes
several smaller physics steps (`timing.physics_step_s`, default 1 ms). After
`N` calls to `step()`, `info["time"]` is exactly `N × control period`.

**Determinism.** With the 3D window off, the same seed and the same actions
give exactly the same trajectory, every time, on the same machine. Across
different operating systems the last digits of the arithmetic can differ, and
a long run may drift apart, so compare numbers produced on one machine.

**The eyes are slower than the controller.** They take a new picture every
`eyes.update_period_s` (default 20 ms, every second control step). In between,
the observation repeats the previous picture, as a real camera would.

**What the eyes see.** Each eye is a pinhole camera, 120° wide and 60° tall by
default, turned 50° outwards. Together they cover about 220° with a 20° overlap
straight ahead; the car is blind to the rear. The bottom rows of each image
show the car's own bodywork.

**What the car does with an action.** The car does not obey instantly. Inside
`flycar/env/vehicle.py` every action is

1. *saturated* to the legal range (and `NaN` is replaced by 0),
2. *rate limited*, so pedals and steering take a fraction of a second to move,
3. *governed*: the throttle fades out near the scenario's `vehicle.max_speed_mps`.

`info["applied_action"]` shows what survived steps 1 and 2. There is no reverse
gear.

## Eye backends

| Backend | Needs | How it works |
|---|---|---|
| `raycast` | nothing (CPU) | One ray per pixel through Chrono's collision system; each hit point is shaded by ambient light, by every lamp it can see (shadow rays), and lamps in view add a glow. |
| `sensor` | NVIDIA GPU with OptiX | One Chrono `ChCameraSensor` per eye, averaged down to the eye grid. The raw images are in `info["camera_left"]` and `info["camera_right"]`. |

`auto` (the default) tries `sensor` and falls back to `raycast`. The choice is
printed when the environment is created and reported in `info["eye_backend"]`.
The two backends use the same camera geometry but are **not** calibrated to
give identical brightness, so compare experiments within one backend. All
tests and the reference results use `raycast`.

## Writing a new driver

A driver is any object with two methods. It does not inherit from anything.

- `reset()` is called once at the start of each episode.
- `act(observation)` is called once per control step and returns the action.

Two rules, both enforced by tests for the built-in drivers:

1. **Use only the observation.** Never `info` (it is ground truth a fly would
   not have) and never the environment object.
2. **Do not import `pychrono`.** Import the index constants from
   `flycar.env.spaces` if you want names instead of numbers.

Here is a complete driver in 20 lines. It steers towards the brighter eye,
holds a speed limit using the IMU, and brakes whenever it touches something.

```python
import numpy as np
from flycar.env.spaces import IMU_SPEED


class SlowSeeker:
    """Steer towards the brighter eye, hold 2 m/s, and stop on contact."""

    def reset(self):
        pass  # this driver has no memory

    def act(self, observation):
        left = float(observation["eye_left"].max())
        right = float(observation["eye_right"].max())
        steering = np.clip(4.0 * (left - right) / (left + right + 1e-6), -1.0, 1.0)

        speed = observation["imu"][IMU_SPEED]
        throttle = 0.5 if speed < 2.0 else 0.0

        brake = 1.0 if observation["contact"].any() else 0.0
        return np.array([throttle, steering, brake], dtype=np.float32)
```

Save it as `flycar/drivers/slow_seeker.py` and run it:

```
python run.py --scenario scenarios/light_seek.yaml --driver flycar.drivers.slow_seeker:SlowSeeker --dashboard
```

To give it a short name like `--driver slow_seeker`, add one line to
`BUILTIN_DRIVERS` in `flycar/drivers/__init__.py`.

### Where a brain plugs in

A brain model is just another driver. The environment calls `act` every 10 ms
of simulated time and does not care what happens inside. A spiking model that
needs a 0.1 ms time step would run 100 of its own steps inside each `act`
call:

```python
class BrainDriver:
    def reset(self):
        self.brain.reset_state()

    def act(self, observation):
        self.brain.set_visual_input(observation["eye_left"], observation["eye_right"])
        for _ in range(100):                 # 100 x 0.1 ms = one 10 ms control period
            self.brain.step(0.0001)
        return self.brain.read_motor_output()   # -> [throttle, steering, brake]
```

Nothing in the environment, the logger, the dashboard or `run.py` has to
change for that.
