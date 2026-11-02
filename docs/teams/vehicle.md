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

## Handling numbers (A1)

`tests/test_vehicle.py` measures how the default car drives and fails if any
number leaves its tolerance band, so an accidental change to the car shows up
in `pytest`. Run it with the table printed:

```
python -m pytest tests/test_vehicle.py -s
```

Today's numbers (default car, `max_speed_mps: 4.0`, recorded 2026-10-15 on
Windows, re-checked on main after the baseline fix on 2026-10-21):

| Measurement | How | Value | Tolerance |
|---|---|---|---|
| Top speed | full throttle 15 s, mean of the last 1 s | 3.974 m/s | ±5 % |
| 0 -> 3 m/s | full throttle from rest, first step with speed >= 3 m/s | 0.520 s | ±10 % |
| Braking distance | full brake from top speed until speed < 0.02 m/s | 2.081 m | ±10 % |
| Turning radius | full left steering, `max_speed_mps: 1.5`, circle fit to `info["pose"]` | 2.806 m | ±5 % |
| Steering t90 | steering step 0 -> 0.5 at `max_speed_mps: 2.0`, time to 90 % of final yaw rate | 0.140 s | ±15 % |

Test conditions: `scenarios/empty_arena.yaml` (file unchanged) with, in the
test only, the spawn fixed at the origin facing +x, a 200 m x 200 m arena so
the car never reaches a wall, and `max_speed_mps` set per test. The tests read
the car only through `env.reset()`, `env.step()`, `info["pose"]` and
`obs["imu"]`.

Notes:
- The top speed follows `max_speed_mps`: with 2.0 and 5.0 the test measures
  about 1.99 and 4.97 m/s (`test_top_speed_follows_max_speed_setting`).
- Steering t90 is mostly set by the steering rate limit
  (`steering_rate_per_s: 4.0`, so a 0.5 step takes 0.125 s).
- If you change the car on purpose (A3 knobs, tires, terrain defaults), re-run
  the suite and update `BASELINE` in `tests/test_vehicle.py` and this table.

## Running your tests

```
pytest tests/test_vehicle.py
python run.py --scenario scenarios/team_vehicle.yaml --driver braitenberg --dashboard
```
