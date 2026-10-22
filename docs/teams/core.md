# Team: Core & body sensing (`core`)

This page belongs to the team. Keep it up to date here instead of in the README.
The Phase 1 work guide (tasks, parts A to C) is [`teams/team-core.md`](../../teams/team-core.md).

## What you own

| Path | What it is |
|---|---|
| `flycar/env/core.py` | `FlyCarEnv`: `reset()`, `step()`, the Chrono system and the time loop |
| `flycar/env/sensing/imu.py` | Speed, yaw rate, acceleration |
| `flycar/env/sensing/contact.py` | Left / front / right touch flags |
| `flycar/config/core.py` | Scenario sections `episode`, `timing`, `contact`, `termination`: defaults and checks |
| `tests/test_core.py` | Your tests |
| `scenarios/team_core.yaml` | Your scratch scenario |
| `flycar/drivers/contrib/core_driver.py` | Your capstone driver (observation only, no pychrono) |
| `teams/team-core.md` | Your Phase 1 work guide |
| `docs/teams/core.md` | This page |

## Contracts you must not break

Changing these needs a lead-approved `api-change` issue (see
[`docs/CONTRACTS.md`](../CONTRACTS.md)):

- **Everything in `flycar/env/spaces.py`** (contract 1): the action, the
  observation (`imu`, `contact`, `time` ...) and the info fields. Only the
  *contents* of `info["extras"]` are free.
- **The step order in `_physics_step()`** (contract 7): terrain, world,
  vehicle, window; synchronize before advance; then
  `system.DoStepDynamics`, then `eyes.update()` and `contact.poll()`.
- Determinism: same seed + same actions = the same trajectory, bit for bit.

You depend on: the `FlyCarVehicle` accessors, `CHASSIS_TAG` and
`WORLD_TAG_BASE`, the eyes interface, and `Terrain`.

## Your reserved config

```yaml
contact:
  mode: flag            # planned: force
termination:
  flipped: false        # planned: end the episode when the car rolls over
  stuck_s: null         # planned: end it after this long without moving
  left_arena: false     # planned: end it when the car leaves the arena
```

Any other value is rejected with "not implemented yet" until you implement it.

## Things to know

- `info["extras"]` is collected in `_extras()` from every component that has
  an `extra_info()` method. Add one to `Imu` or `ContactSensor` to expose,
  say, contact forces (`info["extras"]["core"]`).
- `tests/test_baseline.py` must keep passing: default behaviour may not change.

## Running your tests

```
pytest tests/test_core.py tests/test_baseline.py
python run.py --scenario scenarios/team_core.yaml --driver braitenberg --dashboard
```


## A1 — Core Profiling

A1 implements optional wall-clock profiling of FlyCar's core environment.

- Configuration: `timing.profile`, disabled by default.
- Output: `info["extras"]["core"]["env"]`.
- Measures physics, synchronization, advancement, vision, contact,
  IMU, observation, and info generation.
- Preserves the existing physics execution order and default behavior.
- Resets profiling measurements between episodes.

The 10-episode, 20×20 CPU raycast benchmark recorded an overall
real-time factor of 0.689×.

Full validation: 100 passed, 4 skipped.
Baseline validation: 2 passed, 4 skipped.

See `docs/A1_PROFILING.md` for the complete methodology and results.

