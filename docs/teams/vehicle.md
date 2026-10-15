# Team: Vehicle & terrain (`vehicle`)

This page belongs to the team. Keep it up to date here instead of in the README.

## What you own

| Path | What it is |
|---|---|
| `flycar/env/vehicle.py` | The Gator, the bumper box, action -> driver inputs (saturation, rate limits, speed governor) |
| `flycar/env/terrain.py` | The ground: `Terrain`, built from the scenario's `terrain` section |
| `flycar/config/vehicle.py` | Scenario sections `vehicle`, `spawn`, `terrain`: defaults and checks |
| `tests/test_vehicle.py` | Your tests |
| `scenarios/team_vehicle.yaml` | Your scratch scenario |
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
