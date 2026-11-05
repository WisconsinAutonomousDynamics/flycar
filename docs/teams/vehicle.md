# Team: Vehicle & terrain (`vehicle`)

This page belongs to the team. Keep it up to date here instead of in the README.
The Phase 1 work guide (tasks, parts A to C) is [`teams/team-vehicle.md`](../../teams/team-vehicle.md).

## What you own

| Path | What it is |
|---|---|
| `flycar/env/vehicle.py` | The Gator, the bumper box, action -> driver inputs (saturation, rate limits, speed governor) |
| `flycar/env/terrain.py` | The ground: `Terrain`, built from the scenario's `terrain` section |
| `flycar/config/vehicle.py` | Scenario sections `vehicle`, `spawn`, `terrain`: defaults and checks |
| `tests/test_vehicle.py` | Your tests |
| `scenarios/team_vehicle.yaml` | Your scratch scenario |
| `flycar/drivers/contrib/vehicle_driver.py` | Your capstone driver (observation only, no pychrono) |
| `teams/team-vehicle.md` | Your Phase 1 work guide |
| `docs/teams/vehicle.md` | This page |

## Contracts you must not break

Other teams build on these. Changing them needs a lead-approved `api-change`
issue (see [`docs/CONTRACTS.md`](../CONTRACTS.md)):

- **`FlyCarVehicle` accessors** (contract 3): `pose()`, `position()`,
  `velocity()`, `rotation_matrix()`, `forward_speed()`, `yaw_rate()`,
  `set_action()`, `hold_brake()`, `release_controls()`, and the attributes
  `applied_action`, `inputs`, `vehicle`. The eyes, the IMU and the contact
  sensor read the car only through these.
- **`CHASSIS_TAG`** and `BUMPER_X_MAX_M` (contract 4): the contact sensor uses
  them to recognise and zone chassis contacts.
- **`Terrain`** (contract 6): `chrono_terrain`, `synchronize(time)`,
  `advance(step)`.
- **The step order in `core._physics_step()`** (contract 7): terrain, world,
  vehicle; synchronize before advance.

## Your reserved config

```yaml
terrain:
  type: rigid_flat     # rigid_flat or rigid_patches; planned: ramps, scm
  friction: 0.9
  patches: []          # friction patches, only with type rigid_patches (see below)
```

Any other `terrain.type` is rejected with "not implemented yet" until you
implement it in `flycar/config/vehicle.py` and `flycar/env/terrain.py`.

## Friction patches (B1)

Areas of the arena where the car grips less ("ice") or differently ("mud"):

```yaml
terrain:
  type: rigid_patches
  friction: 0.9                 # the rest of the ground, as before
  patches:
    - x: 5.0                    # centre [m]
      y: 0.0
      size_m: [4.0, 4.0]        # size along x and y [m]
      friction: 0.2             # ice
    - x: -5.0
      y: 0.0
      size_m: [4.0, 4.0]
      friction: 0.5             # mud
```

- Every patch needs `x`, `y`, `size_m` and `friction`. Positions are fixed
  numbers (no `[min, max]` ranges: the terrain gets no random generator yet).
  Patches must lie inside the arena and must not overlap each other
  (touching is fine).
- `type: rigid_flat` with no patches is exactly the old ground. Patches with
  `rigid_flat` are rejected rather than silently ignored.
- Each patch is a `RigidTerrain` box patch with its own `ChContactMaterialNSC`,
  top surface at z = 0, level with the ground, so there is no step for the
  tires to bump over. `RigidTerrain` gives a point the friction of the patch
  added first among the highest ones, so the patches are added before the
  ground. (Chrono 10.0, `RigidTerrain::FindPoint`.)
- With TMeasy tires (default) the tires use the patch friction. With
  `vehicle.tire_model: rigid` the tires collide with both the patch and the
  ground box, so a patch only partly applies.
- In the 3D window and to GPU cameras a patch is coloured from icy blue
  (friction 0.1 or less) to dark brown (1.2 or more). The raycast eyes give
  every surface the same albedo, so they do not see patches.
- `tests/test_vehicle.py` checks the friction and height on and off the
  patches, and that full throttle across a friction-0.1 strip gives more
  wheel slip (A2's `info["extras"]`) than the same stretch of plain ground.
- `scenarios/team_vehicle.yaml` has an ice strip between the spawn area and
  the lamp to try it with the keyboard driver.

## Debug data from the car (A2)

`FlyCarVehicle.extra_info()` reports these every step in
`info["extras"]["vehicle"]["vehicle"]`. They are debug data, not API: the
names may change. Log them with `run.py --log-extras`.

| Key | Meaning | Unit |
|---|---|---|
| `wheels.<front\|rear>_<left\|right>.omega_radps` | wheel speed, positive rolling forwards | rad/s |
| `wheels.<...>.slip` | longitudinal slip, positive when driving harder than rolling | - |
| `wheels.<...>.normal_force_n` | vertical tire force (NaN with rigid tires, whose report is not usable) | N |
| `governor_scale` | throttle factor of the speed governor: 1 = no cut, 0 = full cut | - |
| `engine_speed_radps` | engine speed | rad/s |
| `gear` | current gear (the Gator stays in 1) | - |

Example, cruising in `team_vehicle.yaml`: `governor_scale` 0.12, engine
193 rad/s, front-left wheel 14.3 rad/s with slip ~0 and 1900 N. The four
normal forces add up to the car's weight, about 8,900 N, at rest.

## Things to know

- Do not change any default or the order in which Chrono bodies are created:
  `tests/test_baseline.py` compares every run against recorded trajectories.
- To expose debug data (say, the patch under each wheel), add an
  `extra_info()` method returning a dict to `FlyCarVehicle` or `Terrain`. It
  shows up in `info["extras"]["vehicle"]`. It is not part of the API.

## Running your tests

```
pytest tests/test_vehicle.py
python run.py --scenario scenarios/team_vehicle.yaml --driver braitenberg --dashboard
```

## Fly-like vehicle settings

```yaml
vehicle:
  max_speed_mps: 2.0             # top speed governor
  steering_rate_per_s: 20.0      # how quickly the steering command can change
  steering_gain: 2.0             # steering multiplier; scaled command is clamped to [-1, 1]
  brake_rate_per_s: 20.0         # how quickly the brake command can increase
  throttle_curve_exponent: 0.125 # stronger throttle response at partial input
```

### Measured handling

Measured with these settings in a headless, seed-0 empty arena at a fixed spawn
(0, 0, 0), with walls disabled and the `raycast` eye backend:

| A1 measurement | Result |
|---|---:|
| Governed top speed | 1.985 m/s (mean over the final 2 s) |
| 0–3 m/s time | Not reached with the 2.0 m/s governor; reaches 1.90 m/s in 0.27 s. |
| Braking distance | 0.649 m, from 1.985 m/s to 0.040 m/s (0.52 s) |
| Minimum turning radius | 2.750 m at a mean 0.938 m/s, full steering; circle-fit radial RMS error 0.00034 m. For this low-speed test only, the speed governor was set to 1.0 m/s. |
| Steering step response | 0.14 s to the first sample at or above 90% of the final yaw-rate magnitude (0.725 rad/s). |

Measurements used the settings above in a headless, seed-0 `empty_arena` at a
fixed spawn `(0, 0, 0)`, with a `200 m x 200 m` arena, walls disabled, and the
`raycast` eye backend. Top speed was averaged over the final 2 s of a 12 s
full-throttle run. Braking began at the resulting top speed and ended when
speed fell to 0.05 m/s or less. The turning-radius run used the same settings
except for a 1.0 m/s speed governor; the circle was fit to the final 8 s of a
12 s full-throttle, full-steering run. Steering step response began after a
straight full-throttle approach to governed speed; the final yaw rate is the
mean of the last 0.5 s of the response.