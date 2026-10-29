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
  type: rigid_flat     # only value implemented; planned: patches, ramps, scm
  friction: 0.9
```

Any other `terrain.type` is rejected with "not implemented yet" until you
implement it in `flycar/config/vehicle.py` and `flycar/env/terrain.py`.

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