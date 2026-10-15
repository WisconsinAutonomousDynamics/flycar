# Team Tools & Drivers (`tools`): Work Guide

## Your mission
You own everything that *uses* the simulation: the run script, the logger, the replay and dashboard, the drivers, and the infrastructure for running many episodes. The other four teams build the environment; you make it possible to run it a thousand times, see what happened, and compare controllers fairly.

Two things you build are the backbone of the end of Phase 1: the **batch runner** that scores every team's capstone driver, and the **3D replay** that lets anyone re-watch a run in Chrono. You also write the drivers that prove the API is usable from the outside, which is exactly what the brain will need later.

## What you own
| Path | What it is |
|---|---|
| `flycar/tools/` | Logger, replay, dashboard, plotting, and the new tools below |
| `flycar/drivers/` | The built-in drivers and `make_driver()` (the `Driver` protocol in `base.py` is frozen) |
| `run.py` | The reference reset / act / step loop |
| `tests/test_tools.py` | Your tests |
| `scenarios/team_tools.yaml` | Your scratch scenario |
| `docs/teams/tools.md` | Your team page |

The other teams' capstone drivers will live in `flycar/drivers/contrib/<slug>_driver.py`. Each team owns its own file there; you own the folder's `__init__.py` and anything that runs them.

Do not edit any other file. If you need something from another team, open a `cross-team-request` issue.

## Before you start (everyone on the team)
1. Finish all three tutorials.
2. Read `run.py`, `flycar/drivers/base.py`, `flycar/drivers/braitenberg.py`, and the docstrings of `flycar/tools/logger.py`, `replay.py`, `dashboard.py`.
3. Read contracts 1, 2 and 8 in `docs/CONTRACTS.md`, and all of `docs/API.md`.
4. Run:
   ```
   pytest tests/test_tools.py
   python run.py --scenario scenarios/team_tools.yaml --driver braitenberg --episodes 3 --log
   python -m flycar.tools.replay logs/<one of the files>.parquet
   ```

## Rules specific to your team
- **Drivers see only the observation.** Never `info`, never the env, never `pychrono` (a test enforces this). This is the rule that lets the brain plug in later; protect it.
- **`info["extras"]` is unstable debug data.** Tools may *display* it when present, but nothing may *depend* on it: everything must still work when a key is missing.
- **You read the layout dict and info fields** (contracts 1 and 2); you do not change them. If you need a new field, open an `api-change` issue.
- **Tools may import the environment** (`flycar.env`) and pychrono; drivers may not.
- Keep `run.py` short. It is the example everybody reads first. New features go in `flycar/tools/` and are called from `run.py` with one or two lines.

---

## Part A: Run many episodes and compare

### A1. Batch runner (one or two people)
**Goal:** one command that runs N seeds of a scenario with any driver and produces a results table.

Create `flycar/tools/batch.py`:
```
python -m flycar.tools.batch --scenario scenarios/light_seek.yaml --driver braitenberg --seeds 0-49 --out results/
```
- For each seed: success, time to target, final distance, number of control steps with any contact flag, path length, termination reason (from `info["extras"]["core"]` if present).
- Write one CSV (or Parquet) row per episode, and print a summary: success rate with a confidence interval, median time to target.
- Run episodes in parallel with `multiprocessing` (one env per process; Chrono objects cannot be shared between processes). Make sure results are identical whether run with 1 worker or 8.
- Support several drivers and several scenarios in one call (`--driver a,b --scenario x.yaml,y.yaml`) to produce a comparison table.

**Done when:** a test runs 3 seeds with 1 and 2 workers and gets identical tables, and running 50 seeds of `light_seek.yaml` with Braitenberg reproduces the success rate `run.py --episodes 50` reports.

### A2. Domain randomization (one person)
**Goal:** test drivers on variations of a scenario without anyone writing new YAML files.

- `load_scenario()` also accepts a dict, so you can load a scenario, perturb it, and pass the dict to `FlyCarEnv`. Do this in `flycar/tools/randomize.py`.
- Perturbations, each with a range given on the command line or in a small YAML "randomization spec": lamp `intensity`, `falloff_m`, `lighting.ambient`, `terrain.friction`, `vehicle.max_speed_mps`, eye `resolution`.
- Draw perturbations from a generator seeded by the episode seed, so a batch run is reproducible.
- Plug it into the batch runner (`--randomize spec.yaml`).

**Done when:** the batch runner can show how Braitenberg's success rate drops as ambient light rises, as a table or plot.

### A3. Leaderboard (one person)
**Goal:** a single page comparing every driver on every scenario.

Extend the batch runner to output a summary table (and a matplotlib figure using `flycar/tools/plotting.py`'s style): rows are drivers, columns are scenarios, cells are success rate. Automatically pick up every driver in `flycar/drivers/contrib/` plus the built-ins.

---

## Part B: Seeing what happened

### B1. 3D replay in Chrono (one or two people)
**Goal:** re-watch any logged run in a Chrono 3D window, scrubbing back and forth, without re-simulating.

Create `flycar/tools/replay3d.py`:
```
python -m flycar.tools.replay3d logs/light_seek_braitenberg_seed0.parquet
```
- Rebuild the arena from the log's stored `world` layout using the existing `World` and `Terrain` classes with `visualize=True` (read-only use: do not change them).
- Show the car as a body you place at each logged pose (a simple box the size of the bumper, or the Gator mesh if you can load it without building a whole vehicle).
- Controls: play/pause, speed, step forward/back. Overlay the current time, the action and the contact flags.
- If the log contains `extras` (from `--log-extras`), show what you can, and silently skip what is missing.

**Chrono you learn:** building a scene without running the dynamics, the Irrlicht visual system, cameras and overlays.

**Done when:** replaying a logged Braitenberg run shows the car following the same path as the 2D replay plot, and the tool works on a log made without `--log-extras`.

### B2. Dashboard improvements (one person)
**Goal:** make the live dashboard more useful for the other teams' debugging.

- Show contact flags as colored markers on the car outline on the map.
- An optional panel that plots any numeric value from `info["extras"]` chosen on the command line (`--plot-extra core.env.step_ms`, for example), so other teams can watch their own debug data live.
- Keep the blitting approach so it stays fast; measure the frame time before and after.

### B3. Log format checks (one person)
**Goal:** make sure old logs keep working as the project grows.

- Add a `format_version` to the log metadata.
- Add a test that writes a log, reads it back with `load_log()`, and checks every documented column.
- Keep a small sample log in `tests/` (ask the leads to add it to CODEOWNERS) and test that replay still opens it.

---

## Part C: Drivers

### C1. Obstacle-avoiding light-seeker (one person, can start any time)
**Goal:** the first project on the README roadmap. Braitenberg drives straight into any obstacle between it and the lamp.

Write `flycar/drivers/avoider.py` and register it in `BUILTIN_DRIVERS`:
- seek the lamp like Braitenberg,
- when a `contact` flag fires, back off (brake, steer away from the touched side) for a short time,
- optionally, treat large dark areas low in the eye image as "something close" and steer around them before touching.

**Done when:** it beats Braitenberg's success rate on `cluttered.yaml` (from World) or on `light_seek.yaml` over 50 seeds, measured with your batch runner.

### C2. Capstone organizer (the whole team)
Every team writes a driver in `flycar/drivers/contrib/<slug>_driver.py`. You:
- create `flycar/drivers/contrib/__init__.py` and a short README in that folder explaining the rules (observation only, no pychrono, class with `reset()` and `act()`),
- make sure `make_driver("flycar.drivers.contrib.world_driver:WorldDriver")` works for each,
- run the final leaderboard across every scenario in `scenarios/`, with each team's new features switched on in their `team_*.yaml`. This is the first time all features run together, so it doubles as the integration test: record anything that breaks and open issues for the owning team.

## What other teams may ask you for
- **Everyone:** plotting their `extra_info()` data on the dashboard or in replays.
- **World:** drawing new layout keys (moving obstacles, a future `layout["terrain"]`), once approved.
- **Vehicle / Core:** batch runs for their tuning and performance experiments.

## Pull request checklist
- [ ] Only files from "What you own" changed.
- [ ] `pytest` passes, including `tests/test_contracts.py` (driver rules).
- [ ] Drivers import nothing from `pychrono` and use only the observation.
- [ ] Tools work when `info["extras"]` is empty.
- [ ] `run.py` is still short and readable.
- [ ] `docs/teams/tools.md` documents every new command with an example.
