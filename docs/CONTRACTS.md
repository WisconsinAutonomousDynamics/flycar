# Frozen contracts

Five teams work on Fly Car in parallel. They can do that because they agree on
the interfaces between their pieces. This page lists those interfaces.

**No team may change anything on this page without a lead-approved
[`api-change`](../.github/ISSUE_TEMPLATE/api-change.md) issue.** Adding
something new next to a contract is fine (a new method, a new optional
argument with a default that keeps today's behaviour). Renaming, removing, or
changing the meaning, type or units of anything below is not.

`tests/test_contracts.py` checks that the names in contracts 2 to 8 exist with
the signatures given here, so a pull request that renames one fails CI.
`tests/test_baseline.py` checks that default behaviour does not change.

## 1. Action, observation and info: `flycar/env/spaces.py`

Everything in `spaces.py` is frozen: the action channels, the observation keys,
shapes, dtypes, ranges and units, and the info fields. The human-readable
version is the generated part of [`docs/API.md`](API.md).

The one exception is the **contents** of `info["extras"]`: a dict keyed by team
slug (`vehicle`, `world`, `vision`, `core`), then by component, holding
unstable debug data. Teams may put anything in their own slot through an
`extra_info()` method. Drivers never see it, and nobody may depend on it.

## 2. The layout dict: `sample_layout(scenario, rng)` in `flycar/env/world.py`

`sample_layout()` turns a scenario into one concrete episode. The result is
plain Python data, returned by `env.reset()` as `info["world"]`, stored in
every log, and read by the dashboard, the replay tool, the eyes and the
environment. Its keys and types:

| Key | Type | Meaning |
|---|---|---|
| `arena` | dict | `size_m`: `[float, float]` extent along x and y [m], centred on the origin; `walls`: bool; `wall_height_m`: float [m]; `wall_thickness_m`: float [m] |
| `walls` | list of dict | One per wall (empty if `arena.walls` is false). Each: `shape`: `"box"`; `x`, `y`: float [m] centre; `yaw`: float [rad]; `size_m`: `[length, width, height]` floats [m] |
| `obstacles` | list of dict | One per placed obstacle. Each: `shape`: `"box"` or `"cylinder"`; `x`, `y`: float [m] centre; `yaw`: float [rad]; `size_m`: list of float, `[length, width, height]` for a box, `[radius, height]` for a cylinder [m] |
| `lights` | list of dict | One per lamp. Each: `x`, `y`, `z`: float [m]; `intensity`: float; `falloff_m`: float [m]; `target`: bool |
| `target` | `[float, float]` or `None` | x, y of the target lamp [m]; `None` if no lamp has `target: true` |
| `spawn` | `[float, float, float]` | Spawn pose x [m], y [m], yaw [rad] |
| `lighting` | dict | `ambient`: float; `sky`: float |

The same scenario and the same `numpy.random.Generator` state always give the
same layout.

## 3. The car: `FlyCarVehicle` in `flycar/env/vehicle.py`

The eyes, the IMU, the contact sensor and the environment read the car only
through these:

| Name | Kind | Returns / does |
|---|---|---|
| `pose()` | method | `[x, y, yaw]` in the world frame [m, m, rad] |
| `position()` | method | chassis origin in the world frame, shape (3,) [m] |
| `velocity()` | method | chassis origin velocity in the world frame, shape (3,) [m/s] |
| `rotation_matrix()` | method | 3x3 matrix whose columns are the chassis x, y, z axes in the world frame |
| `forward_speed()` | method | speed along the chassis x axis [m/s] |
| `yaw_rate()` | method | angular velocity about the chassis z axis [rad/s], positive = left |
| `set_action(action, control_period)` | method | applies an action (saturated, rate limited, governed); returns the applied action |
| `hold_brake()` | method | full brake, no throttle (used while the car settles) |
| `release_controls()` | method | zero inputs and forget the control history |
| `applied_action` | attribute | float32 (3,), the last applied `[throttle, steering, brake]` |
| `inputs` | attribute | the `veh.DriverInputs` Chrono reads each step |
| `vehicle` | attribute | the underlying `ChWheeledVehicle` (the 3D window attaches to it) |

## 4. The world: `World` in `flycar/env/world.py`, and the tag scheme

| Name | Kind | Meaning |
|---|---|---|
| `World.current_lights()` | method | list of light dicts (same keys as `layout["lights"]` entries), positions in the world frame, as they are *now*. The eyes call it whenever they need the lights. |
| `World.layout` | attribute | the layout dict of contract 2 |
| `WORLD_TAG_BASE` | constant in `world.py` (1000) | every wall and obstacle body has a tag `>= WORLD_TAG_BASE`; the ground does not |
| `CHASSIS_TAG` | constant in `vehicle.py` (500) | the tag of the chassis body |

The contact sensor relies on the tags to tell "touched a wall or obstacle"
apart from "touching the ground".

## 5. The eyes: `flycar/env/sensing/eyes.py`

Both backends (`RaycastEyes`, `SensorEyes`) are created with
`make_eyes(backend, system, vehicle, world, config)` and offer:

| Method | Called | Returns |
|---|---|---|
| `update()` | after every physics step | nothing |
| `render()` | when the eyes are due for a new picture | `(left, right)`, each float32 `(rows, cols)` in [0, 1] |
| `raw_images()` | every control step | `(left, right)` raw RGB uint8 images, or `(None, None)` |

## 6. The ground: `Terrain` in `flycar/env/terrain.py`

| Name | Kind | Meaning |
|---|---|---|
| `Terrain(system, layout, terrain_config, visualize=False)` | constructor | builds the ground from the scenario's `terrain` section |
| `chrono_terrain` | attribute | the Chrono `ChTerrain` passed to the vehicle |
| `synchronize(time)` | method | phase 1 of the step |
| `advance(step)` | method | phase 2 of the step |

## 7. The physics step order: `FlyCarEnv._physics_step()` in `flycar/env/core.py`

Once per physics step, in exactly this order:

1. `terrain.synchronize(time)`
2. `world.synchronize(time)`
3. `vehicle.synchronize(time, terrain.chrono_terrain)`
4. (3D window only) `window.Synchronize(time, vehicle.inputs)`
5. `terrain.advance(step)`
6. `world.advance(step)`
7. `vehicle.advance(step)`
8. (3D window only) `window.Advance(step)`
9. `system.DoStepDynamics(step)`
10. `eyes.update()`
11. `contact.poll()`

Changing the order changes the trajectory, which the baseline test would catch.

## 8. The driver protocol: `Driver` in `flycar/drivers/base.py`

A driver is any object with:

- `reset()`: called once at the start of each episode;
- `act(observation)`: called once per control step, returns
  `[throttle, steering, brake]`.

A driver uses only the observation (never `info` or the environment) and never
imports `pychrono`. This is what lets the connectome brain plug in as just
another driver.
