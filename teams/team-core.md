# Team Core & Body Sensing (`core`): Work Guide

## Your mission
You own the heart of the simulation: `FlyCarEnv`, which builds the Chrono system, runs the physics loop and assembles every observation, plus the fly's body senses (IMU and touch).

Your work decides **how fast** the whole simulation runs (which matters enormously once a brain model runs inside every step) and **when an episode ends**. You are also the team others will ask for small hooks, so part of your job is answering their `cross-team-request` issues quickly.

## What you own
| Path | What it is |
|---|---|
| `flycar/env/core.py` | `FlyCarEnv`: `reset()`, `step()`, the physics loop, observation and info |
| `flycar/env/sensing/imu.py` | Speed, yaw rate, acceleration |
| `flycar/env/sensing/contact.py` | Left / front / right touch flags |
| `flycar/config/core.py` | Scenario sections `episode`, `timing`, `contact`, `termination` |
| `tests/test_core.py` | Your tests |
| `scenarios/team_core.yaml` | Your scratch scenario |
| `docs/teams/core.md` | Your team page |
| `flycar/drivers/contrib/core_driver.py` | Your capstone driver (Part C) |

Do not edit any other file. If you need something from another team, open a `cross-team-request` issue.

## Before you start (everyone on the team)
1. Finish all three tutorials. `tutorials/01_falling_box.py` (the system and the step loop) is the foundation of `core.py`.
2. Read `flycar/env/core.py` top to bottom. Draw the two clocks on paper: one `step()` = one control period = several physics steps.
3. Read contracts 1 and 7 in `docs/CONTRACTS.md`, and `INFO_FIELDS` in `flycar/env/spaces.py`.
4. Run:
   ```
   pytest tests/test_core.py tests/test_baseline.py
   python run.py --scenario scenarios/team_core.yaml --driver braitenberg --episodes 5
   ```

## Rules specific to your team
- **The physics step order is a contract** (contract 7). Do not reorder `_physics_step()`. Adding timing code around the calls is fine; changing what is called or when is not.
- **Determinism is your responsibility.** Same seed + same actions = same trajectory, bit for bit, headless. That is why the system runs single-threaded today. Any change that might affect this (threads, solver, collision settings) goes behind a config key with today's behavior as default.
- **The observation and info fields are frozen** (`spaces.py`). New data you want to expose (timings, forces, termination reasons) goes in `extra_info()` on `Imu` or `ContactSensor`, which appear under `info["extras"]["core"]`. If `FlyCarEnv` itself should report something, add a `("env", "core")`-style source to `EXTRA_INFO_SOURCES` in `core.py` (your file).
- **Contact channels are documented as 0/1 flags.** Making them carry forces changes their meaning, which is an `api-change` (see B2).
- Other teams cannot touch `core.py`. When they need a hook (an extra argument, a new call), they will ask you. Keep those changes small, optional and default-off.

---

## Part A: Speed

### A1. Profiling (one person)
**Goal:** know exactly where the time goes in one `step()`.

- Add timers in `_physics_step()` and `step()` around each phase: terrain, world, vehicle, `DoStepDynamics`, `eyes.update()`, `contact.poll()`, and once per control step `eyes.render()`, IMU, observation and info assembly.
- Report the accumulated times through an `extra_info()` on the environment (add it to `EXTRA_INFO_SOURCES`). Use `time.perf_counter()`.
- The timers must not change the simulation. Measure their own overhead and keep it negligible.

**Done when:** you can run 10 episodes of `light_seek.yaml` and produce a table on your team page: percentage of wall time per phase, and the **real-time factor** (simulated seconds per wall-clock second), for the raycast backend at 20×20.

### A2. Benchmark test (one person)
**Goal:** catch anyone who makes the simulation slower.

Add a `@pytest.mark.slow` test in `tests/test_core.py` that runs a fixed episode headless and reports steps per second. Do not make it fail on an absolute number (machines differ). Instead, record the result on your team page and print it, so a big regression is visible in CI logs. Share the numbers with Vision: the eyes will likely be the largest slice.

### A3. Solver and threading experiments (one person)
**Goal:** find out how much faster the physics can go, and what it costs.

Try, one at a time, behind new `timing` keys whose defaults are today's settings:
- `physics_step_s` of 2 ms instead of 1 ms (watch for tire instability),
- solver iterations (`max_solver_iterations`, default 150),
- `num_threads` > 1 (default 1).

For each: real-time factor, whether the A1 handling of the car still looks right (ask Vehicle for their handling tests), and **whether two runs with the same seed still match exactly**. Write the results on your team page as a recommendation: "for fast batch runs use X; for reproducible experiments keep the defaults".

**Chrono you learn:** `ChSystem` solver settings, time-step stability, multithreading and floating-point determinism.

---

## Part B: Better body senses and episode endings

### B1. Termination conditions (one person)
**Goal:** implement the reserved `termination` keys, so stuck or crashed episodes end early instead of wasting time.

- `flipped: true`: end the episode when the chassis up axis (`rotation_matrix()[:, 2]`) tilts past a threshold.
- `stuck_s: N`: end it when the car has moved less than a small distance in the last `N` seconds.
- `left_arena: true`: end it when the car's position is outside `layout["arena"]["size_m"]` (relevant when `walls: false`).
- These end the episode with `terminated=True` (or `truncated`, decide and document which) **without** setting `success`.
- Report *why* the episode ended in `info["extras"]["core"]` (e.g. `"termination_reason": "stuck"`).
- Remove the keys from `NOT_YET_IMPLEMENTED` and replace the "not implemented yet" tests.

**Done when:** each condition has a test that triggers it on purpose (e.g. a driver that brakes forever for `stuck_s`), and with all defaults off, `test_baseline.py` still passes.

### B2. Contact forces (one person)
**Goal:** know not just *whether* the car touched something but *how hard*.

- In `_ContactCollector.OnReportContact`, also record the contact force (Chrono gives it in the contact frame; work out what you need from `plane` and `force`).
- Expose per-zone peak force in newtons through `ContactSensor.extra_info()`.
- The reserved `contact.mode: force` would change what `obs["contact"]` means. Do **not** implement it as forces in newtons. Instead, open an `api-change` issue proposing what the observation should be (for example a normalized 0 to 1 "pressure" per zone, saturating at some force), so the leads can decide. Implement it only after approval.

**Done when:** a test drives into a wall at two speeds and the faster impact reports a larger force in extras.

### B3. IMU realism (one person)
**Goal:** an IMU that behaves more like a real sensor, optionally.

- Add `imu` options to your config (new section, defaults off): `noise_std` per channel, and `bias`. Use `self.np_random` from the environment so noise is seeded; pass it to `Imu` as an optional argument.
- Stretch: an option to use the Chrono sensor-module IMU (`ChAccelerometerSensor` / `ChGyroscopeSensor`) when available, falling back to today's derived IMU.
- Check the acceleration signal while driving over Vehicle's ramps; it should show the expected pitch effects.

**Done when:** defaults reproduce today exactly; with noise on, two runs with the same seed give identical noisy readings.

---

## Requests you will probably get (handle them as they arrive)
These come from the other teams' guides. Each should be a small, optional, default-off change in `core.py`:
- **Vision:** pass `rng=self.np_random` into `make_eyes(...)` (for pixel noise).
- **Vehicle:** pass `rng=self.np_random` into `Terrain(...)` (for randomized terrain).
- **World:** judge `success` against the target's *current* position when lamps move (e.g. using a new `World.current_target()`), falling back to `layout["target"]`.

When you merge one, comment on the issue so the requesting team knows they can use it.

---

## Part C: Capstone driver
Write `flycar/drivers/contrib/core_driver.py`, using only the observation. You own touch and the IMU, so make them matter: a light-seeker that backs off and turns away when a contact flag fires, and slows down when `imu` yaw rate is high.

## Pull request checklist
- [ ] Only files from "What you own" changed.
- [ ] `pytest` passes, including `tests/test_baseline.py` and `tests/test_contracts.py`.
- [ ] `_physics_step()` call order unchanged.
- [ ] Determinism checked: two runs with the same seed give identical `pose` traces.
- [ ] New config keys default to today's behavior and have validation checks.
- [ ] Performance numbers on `docs/teams/core.md` updated if you changed anything in the loop.
