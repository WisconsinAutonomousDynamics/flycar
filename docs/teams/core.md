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

## Speed settings (A3)

```yaml
timing:
  physics_step_s: 0.001        # Chrono integration step [s]
  max_solver_iterations: 150   # contact/joint solver iterations per physics step
  num_threads: 1               # Chrono, collision and Eigen threads
```

The defaults are the settings from before A3, so `test_baseline.py` is unaffected.

### Recommendation

- **For reproducible experiments, and anything compared with the baselines, keep the defaults.**
- **For fast batch runs, use `physics_step_s: 0.002` with `max_solver_iterations: 50`.**
  Whole-step real-time factor goes from 5.4 to 10.4 (×1.9), physics alone from 8.3 to 35 (×4.2).
  Handling stays within about 6 %, and runs are still bit-for-bit deterministic.
  Trajectories are *not* the same as with the defaults (0.12 m apart after 3 s), so do not mix
  results from the two settings.
- If you want to stay closer to the default trajectories, use `max_solver_iterations: 50` alone:
  ×1.5 overall, 2 mm from the default trajectory.
- Avoid:
  - `max_solver_iterations: 10`: the simulation blows up (NaN poses).
  - `physics_step_s: 0.005`: handling clearly changes (0→3 m/s takes 60 % longer, turn radius +15 %).
  - `num_threads > 1`: slower, see below.

With the fast settings, physics is only 30 % of the wall time; the eyes and the observation are the rest.

### Results

Measured on an Apple M4 Pro with pychrono 10.0.0, headless, raycast eyes at 20×20.

- **RTF:** simulated s per wall-clock s over 10 `light_seek.yaml` episodes with the braitenberg driver.
- **Physics RTF:** the same, counting only the time spent in `_physics_step()`.
- **Same seed identical:** the same seed and actions were run 3 times (twice in one env, once in a fresh env),
  and every `pose` matched bit for bit.
- **Drift:** the largest distance from the default-settings trajectory over the same 300 steps.
- **Handling:** measured on a 300 m flat arena.
  - v_max: top speed at full throttle.
  - 0→3: time to reach 3 m/s.
  - brake: braking distance from top speed.
  - R: turning radius at full steering, throttle 0.3.
  - t90: time to 90 % of the final yaw rate after a 0.5 steering step.

| setting | RTF | physics RTF | same seed identical | drift [m] | v_max [m/s] | 0→3 [s] | brake [m] | R [m] | t90 [s] |
|---|---|---|---|---|---|---|---|---|---|
| defaults (1 ms, 150 it, 1 thread) | 5.4 | 8.3 | yes | 0 | 3.98 | 0.52 | 2.03 | 2.89 | 0.24 |
| `physics_step_s: 0.002` | 7.9 | 16.2 | yes | 0.12 | 3.97 | 0.55 | 2.04 | 2.83 | 0.22 |
| `physics_step_s: 0.005` | 11.0 | 39.7 | yes | 0.81 | 4.01 | 0.84 | 2.46 | 3.32 | 0.23 |
| `max_solver_iterations: 100` | 6.4 | 11.2 | yes | 0.008 | 3.98 | 0.52 | 2.03 | 2.89 | 0.24 |
| `max_solver_iterations: 50` | 8.3 | 18.2 | yes | 0.002 | 3.98 | 0.52 | 2.03 | 2.89 | 0.24 |
| `max_solver_iterations: 25` | 9.6 | 26.5 | yes | 0.013 | 3.98 | 0.52 | 2.04 | 2.89 | 0.24 |
| `max_solver_iterations: 10` | blows up | | | | | | | | |
| `num_threads: 2` | 5.2 | 7.8 | yes | 0 | same as defaults | | | | |
| `num_threads: 4` | 5.0 | 7.5 | yes | 0 | same as defaults | | | | |
| `num_threads: 8` | 4.7 | 6.6 | yes | 0 | same as defaults | | | | |
| **2 ms + 50 it** | **10.4** | **35.0** | **yes** | 0.12 | 3.97 | 0.55 | 2.04 | 2.83 | 0.22 |

Driving into the wall still sets the front contact flag and stops the car at x = 8.70 m in every
setting except 10 iterations. Chassis height and tilt stayed in the same range in every setting
except 5 ms, which tilts slightly more (2.3° instead of 1.7°). So there is no tire instability at 2 ms.

**Why threads do not help:** Chrono is built with OpenMP, but our scene (one car, a few walls and
obstacles) is far too small for its parallel loops to pay off. CPU time per wall second stayed at 1.0
with 8 threads, so the extra threads only add overhead. In these runs the threaded results were even
bit-identical to single-threaded, but Chrono does not guarantee that, so `num_threads > 1` stays
documented as non-reproducible. For more throughput, run separate episodes in separate processes
(one env per core) instead.

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
