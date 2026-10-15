# Team World (`world`): Work Guide

## Your mission
You own everything in the arena that is not the car or the ground: walls, obstacles and lamps, plus the scenario files that describe them. Today the world is static: boxes and cylinders that never move, lamps that never move, and two scenarios.

By the end of Phase 1 there should be a **library of scenarios** that test different behaviors, and a world where **obstacles and lamps can move**. A fly that can find a lamp in a static room is nice; a fly that can follow a moving one is the real test.

## What you own
| Path | What it is |
|---|---|
| `flycar/env/world.py` | `sample_layout()` (rolls the dice) and `World` (Chrono bodies) |
| `flycar/config/world.py` | Scenario sections `arena`, `lighting`, `lights`, `obstacles`, `motion` |
| `scenarios/` | Every scenario **except** `empty_arena.yaml`, `light_seek.yaml` (frozen) and the other teams' `team_*.yaml` scratch files |
| `tests/test_world.py` | Your tests |
| `scenarios/team_world.yaml` | Your scratch scenario |
| `docs/teams/world.md` | Your team page |
| `flycar/drivers/contrib/world_driver.py` | Your capstone driver (Part C) |

Do not edit any other file. If you need something from another team, open a `cross-team-request` issue.

## Before you start (everyone on the team)
1. Finish all three tutorials. `tutorials/02_motor_cart.py` (bodies, joints, motors) is the most relevant one for you.
2. Read `flycar/env/world.py` top to bottom. Notice the two steps: `sample_layout()` makes plain numbers with no Chrono, and `World` turns them into bodies. Keep that separation.
3. Read contracts 2, 4 and 7 in `docs/CONTRACTS.md`.
4. Run:
   ```
   pytest tests/test_world.py tests/test_baseline.py
   python run.py --scenario scenarios/team_world.yaml --driver braitenberg --dashboard
   ```

## Rules specific to your team
- **The `layout` dict is a contract.** The eyes, logger, replay and dashboard all read it. You may *add* a new key; you may not rename, remove or change the meaning of an existing one. If you add a key, tell Tools (they may want to draw or log it) with a `cross-team-request`.
- **New obstacle shapes are a contract change.** `layout["obstacles"][i]["shape"]` is documented as `"box"` or `"cylinder"`, and the dashboard and replay only know how to draw those. Until an `api-change` issue is approved, build new things (mazes, corridors, U-shapes) out of boxes and cylinders.
- **Tags:** every wall and obstacle body must get a tag `>= WORLD_TAG_BASE`, or the contact sensor will not see it. Use `_add_solid()`, which already does this.
- **Randomness:** everything random must come from the `rng` passed into `sample_layout()`. No `random`, no `np.random` globals, no time-based seeds. Same seed, same layout.
- **Frozen scenarios:** never edit `empty_arena.yaml` or `light_seek.yaml`. Copy them.
- **Eyes ask you where the lights are**, through `World.current_lights()`. Never make the eyes (or anyone else) read `layout["lights"]` for current positions.

---

## Part A: A scenario library

### A1. Scenario designs (one or two people)
**Goal:** a set of scenarios that each test one thing, so every driver (and later the brain) can be compared fairly.

Write each as a new YAML file in `scenarios/`, with a header comment explaining what it tests and how success is judged:
- `corridor.yaml`: a long, narrow walled corridor with the lamp at the end. Tests going straight without hitting walls.
- `two_lamps.yaml`: one target lamp and one bright non-target lamp ("distractor"). Tests whether a driver can be fooled.
- `occluded_lamp.yaml`: the lamp starts hidden behind a large obstacle. Tests what a driver does when it sees nothing.
- `cluttered.yaml`: many small obstacles between spawn and lamp. Tests obstacle handling.
- `dim_lamp.yaml`: a weak lamp with short falloff in a bright arena. Tests sensitivity.

Use only features that exist today. Randomize positions with `[min, max]` ranges where it makes sense, and set `spawn.min_target_distance_m` so the task is never trivial.

**Done when:** every scenario loads (the existing `test_scenario_files_load` test picks them up automatically), and you have run the Braitenberg driver on 20 seeds of each and recorded its success rate in `docs/teams/world.md`.

### A2. Maze generator (one person)
**Goal:** randomized mazes without hand-placing boxes.

- Add a `maze` section to `flycar/config/world.py` (default: off), e.g. `enabled`, `cells: [nx, ny]`, `wall_thickness_m`, `wall_height_m`, `gap_m`.
- In `sample_layout()`, when enabled, generate a random maze from `rng` (recursive backtracker is a classic) and **append its walls to `layout["obstacles"]` as boxes**. That way the eyes, contact sensor, dashboard and replay all work with no changes anywhere else.
- Put spawn in one corner cell and the target lamp in the opposite one.

**Done when:** `scenarios/maze.yaml` runs, the same seed always gives the same maze, different seeds give different mazes, and a test checks every maze has a path from spawn to target.

### A3. Layout validation (one person)
**Goal:** catch broken scenarios before they reach Chrono.

Extend `validate()` and `sample_layout()` with checks that are missing today, for example:
- obstacles placed outside the arena,
- a target lamp outside the arena or inside an obstacle,
- `count` of obstacles that cannot fit (raise a clear error instead of looping until `MAX_SAMPLING_TRIES`),
- `lighting.ambient` and `sky` outside [0, 1].

Each check needs a test with a deliberately broken scenario dict and the expected error message.

---

## Part B: Things that move

The hooks are already in place. `World.synchronize(time)` and `World.advance(step)` are called every physics step (contract 7), and are empty today. `World.current_lights()` is what the eyes read every time they take a picture. You only need to fill these in.

### B1. Moving obstacles (one or two people)
**Goal:** obstacles that slide back and forth or go around in circles.

- Design the `motion.obstacles` config: each entry refers to an obstacle and gives a motion, e.g. `{obstacle: 0, kind: linear, axis_deg: 90, amplitude_m: 3.0, period_s: 6.0, phase_deg: 0}` and `kind: circle`.
- Implement it in `World` with Chrono motion. Two options; try both on a small test and pick one:
  - a `ChLinkMotorLinearPosition` (or rotation motor) driven by a `ChFunctionSine`, with the obstacle body no longer fixed;
  - a kinematic body you move yourself in `synchronize(time)`, setting position **and** velocity so contacts behave.
- Moving bodies keep their tag `>= WORLD_TAG_BASE`, so the contact sensor and the raycast eyes see them with no change on their side.
- Remove the matching entry from `NOT_YET_IMPLEMENTED` and replace the "not implemented yet" test with real tests.

**Chrono you learn:** motors, links, `ChFunction`s, the difference between fixed, kinematic and dynamic bodies.

**Done when:** in `team_world.yaml` an obstacle sweeps across the car's path, the car's contact flags fire when it is hit, the motion is identical across two runs with the same seed, and `test_baseline.py` still passes (no motion configured = nothing moves).

### B2. Moving lamps (one person)
**Goal:** a target that moves, so the fly has to track it.

- Design `motion.lights` the same way (linear, circle, waypoints).
- Make `current_lights()` return the lamp's **current** position.
- Move the lamp's visual ball too (today it is a fixed body created only when `visualize` is on), so the 3D window and the GPU cameras show it in the right place. The sensor backend already moves its scene lights when `current_lights()` changes.
- Success is judged in `core.py` from `layout["target"]`, which is the *initial* position. Open a `cross-team-request` to Core asking them to judge success against the target's current position (for example via a new `World.current_target()` method that you add).

**Done when:** the Braitenberg driver visibly follows a slowly circling lamp in `team_world.yaml`, and a test checks that the raycast eye image changes as the lamp moves while the car stands still.

### Randomized motion
If you want motion parameters (phase, speed) randomized per episode, draw them in `sample_layout()` from `rng` and store them under a new layout key such as `layout["motion"]`. That is an addition, which is allowed; tell Tools so the logger keeps it.

---

## Part C: Capstone driver
Write `flycar/drivers/contrib/world_driver.py`, using only the observation. Since you built the scenarios, aim for the hard ones: a driver that handles `two_lamps.yaml` or `occluded_lamp.yaml` better than Braitenberg (for example by turning in place to search when both eyes are dark).

## What other teams may ask you for
- **Vision** depends on `current_lights()` and `layout["lighting"]`.
- **Core** may need a way to ask where the target is now (see B2).
- **Tools** reads the layout for the map, the logs and the 3D replay; give them a heads-up before adding a layout key.

## Pull request checklist
- [ ] Only files from "What you own" changed (other teams' `team_*.yaml` are theirs).
- [ ] `pytest` passes, including `tests/test_baseline.py` and `tests/test_contracts.py`.
- [ ] New config keys default to today's behavior and have validation checks.
- [ ] Everything random comes from `rng`.
- [ ] `docs/teams/world.md` lists every scenario with a one-line description.
