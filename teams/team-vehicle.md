# Team Vehicle & Terrain (`vehicle`): Work Guide

## Your mission
You own the car and the ground it drives on. Today that is a stock Gator on one flat, rigid patch. By the end of Phase 1 the car should handle like a slow, nimble "fly", and the ground should offer more than a flat floor: patches with different grip, ramps, and (as a stretch) soft soil.

Every other team reads the car through your accessors, so your job is half physics, half keeping a promise: **the car may get better, but its interface never changes.**

## What you own
| Path | What it is |
|---|---|
| `flycar/env/vehicle.py` | `FlyCarVehicle`: the Gator, the bumper box, action → driver inputs |
| `flycar/env/terrain.py` | `Terrain`: the ground |
| `flycar/config/vehicle.py` | Scenario sections `vehicle`, `spawn`, `terrain` |
| `tests/test_vehicle.py` | Your tests |
| `scenarios/team_vehicle.yaml` | Your scratch scenario |
| `docs/teams/vehicle.md` | Your team page (keep it current) |
| `flycar/drivers/contrib/vehicle_driver.py` | Your capstone driver (Part C) |

Do not edit any other file. If you need something from another team, open a `cross-team-request` issue.

## Before you start (everyone on the team)
1. Finish all three tutorials. `tutorials/03_vehicle_figure8.py` is the one your code is built on.
2. Read, in this order: `flycar/env/vehicle.py`, `flycar/env/terrain.py`, `flycar/config/vehicle.py`, contracts 3, 6 and 7 in `docs/CONTRACTS.md`.
3. Run these and make sure they pass on your machine:
   ```
   pytest tests/test_vehicle.py tests/test_baseline.py
   python run.py --scenario scenarios/team_vehicle.yaml --driver keyboard
   ```
4. Drive the car with the keyboard driver for five minutes. Notice how it accelerates, turns and stops. That is the behavior you are about to measure and improve.

## Rules specific to your team
- **Never change a default.** `tests/test_baseline.py` replays recorded runs; a new default tire model, mass or friction would break it. New behavior goes behind a new config key whose default reproduces today.
- **Never change the order Chrono objects are created in** (`reset()` builds terrain, then vehicle, then world). Contract 7 and the baseline both depend on it.
- **Adding a config key:** add it with a comment (meaning and units) to the right section in `flycar/config/vehicle.py`, and a check in `validate()`. The loader rejects unknown keys, so a key that is not declared there cannot be used.
- **The car cannot drive backwards.** The action space has no reverse, and adding one is a change to `spaces.py` (an `api-change` issue), not something you can do in `vehicle.py`.
- Any data you want to show (wheel slip, tire forces, which patch each wheel is on) goes in an `extra_info()` method on `FlyCarVehicle` or `Terrain`. It appears in `info["extras"]["vehicle"]`. It is debug data, not API.

---

## Part A: Measure and tune the car

### A1. Handling test suite (one person)
**Goal:** know, in numbers, how the car behaves, and catch anyone who changes it by accident.

Write tests in `tests/test_vehicle.py` (mark the slow ones `@pytest.mark.slow`) that build an env on `empty_arena.yaml` and measure:
- **Top speed** under full throttle (should settle near `vehicle.max_speed_mps`).
- **0 → 3 m/s time.**
- **Braking distance** from top speed with full brake.
- **Minimum turning radius** at low speed with full steering (fit a circle to the `pose` trace).
- **Steering step response:** time from a steering step to 90% of the final yaw rate.

Use only `env.reset()`, `env.step()` and `info["pose"]`, `obs["imu"]`. Print a small table when run with `-s`, and assert each value within a tolerance band around today's numbers.

**Chrono you learn:** how the vehicle's Synchronize/Advance loop turns inputs into motion; tire and driveline behavior.

**Done when:** the suite passes, and changing `max_speed_mps` in a scratch scenario makes the top-speed test move as expected.

### A2. Expose the car's internals (one person)
**Goal:** give everyone (and the dashboard via `--log-extras`) a look inside the car.

Add `FlyCarVehicle.extra_info()` returning plain Python numbers:
- per-wheel angular speed and longitudinal slip,
- per-wheel tire normal force,
- the governor's throttle scale (how much it is currently cutting throttle),
- the engine speed and current gear, if the Gator's powertrain exposes them.

Find the right Chrono calls by reading the installed PyChrono vehicle demos and the `ChWheeledVehicle` / `ChWheel` / `ChTire` docs for your Chrono version. Do not guess names.

**Done when:** `python run.py --scenario scenarios/team_vehicle.yaml --log-extras` writes a log whose `extras` column contains these values, and a test checks the keys exist and are finite.

### A3. Tuning knobs (one person)
**Goal:** make "fly-like" handling something you can set in YAML.

Add new keys to the `vehicle` section, each defaulting to today's behavior:
- `tire_model`: `tmeasy` (today) or another model the Gator supports, such as `rigid`.
- `steering_gain`: multiplies the steering command before it reaches Chrono (default `1.0`).
- `throttle_curve_exponent`: shapes throttle response, `throttle ** n` (default `1.0`).

Then, in `scenarios/team_vehicle.yaml`, find a setting that feels "fly-like": slow top speed, quick turns, stops fast. Record the handling numbers from A1 for that setting in `docs/teams/vehicle.md`.

**Done when:** each knob has a validation check, a test proving the default is unchanged, and a test proving a non-default value changes the A1 measurement it should.

---

## Part B: Terrain that is more than a floor

### B1. Friction patches (one or two people)
**Goal:** areas of the arena where the car grips more or less (think "ice" and "mud").

- Add `patches` to the `terrain` section (default `[]`). Each entry: `x`, `y`, `size_m: [sx, sy]`, `friction`.
- Add a new `terrain.type` value, `rigid_patches`, to `TERRAIN_TYPES` and implement it in `terrain.py` with additional `RigidTerrain` patches, each with its own `ChContactMaterialNSC`.
- With `visualize=True`, color each patch by its friction so it is visible in the 3D window and to GPU cameras.
- Make sure the base patch and the extra patches do not overlap in height in a way that makes the tires jitter (check the patch documentation for your Chrono version).

**Chrono you learn:** `RigidTerrain` patches, contact materials, how tires query friction.

**Done when:** a test drives straight across a low-friction patch with full throttle and measures more wheel slip (from A2's extras) than on the base patch; `type: rigid_flat` behaves exactly as before.

### B2. Ramps and bumps (one person)
**Goal:** a ground that is not flat.

- Add `ramps` to the `terrain` section (default `[]`): `x`, `y`, `yaw_deg`, `length_m`, `width_m`, `height_m`.
- Implement ramps as terrain (a rotated patch, a mesh patch, or a heightmap patch), not as World obstacles. They must be part of the ground, so the contact sensor does not flag them as walls (their bodies must **not** carry a tag `>= WORLD_TAG_BASE`).
- Check that the raycast eyes see the ramps: rays hit collision shapes, so they should. Add a test that a ramp in front of the car changes the eye image.

**Done when:** the car can drive up and over a ramp in `team_vehicle.yaml`, the IMU shows the pitch-induced acceleration, and contact flags stay 0 while doing it.

### B3. Stretch: soft soil (SCM)
**Goal:** `terrain.type: scm`, deformable ground that leaves tracks.

- Use `veh.SCMTerrain`. Expect it to be much slower; measure the real-time factor and write it on your team page.
- Check two things carefully and document the answers: whether the raycast eyes can see SCM ground (it may not live in the collision system the same way), and whether runs stay deterministic.
- If the eyes cannot see it, that is fine for a stretch goal: say so in the validator message when `scm` is combined with `eyes.backend: raycast`.

### Terrain randomization and visualization (needs other teams)
`Terrain` receives the layout and its config, but no random generator, and the dashboard only draws what is in `info["world"]`. So today patches and ramps are **fixed** per scenario and **not drawn** on the dashboard map. If you want either:
- **Randomized terrain features:** open a `cross-team-request` to Core asking them to pass `rng=self.np_random` to `Terrain` as an optional argument.
- **Drawing patches and ramps on the map:** open an `api-change` issue proposing a new `layout["terrain"]` key, and a `cross-team-request` to Tools to draw it.

Do not work around these by editing their files.

---

## Part C: Capstone driver
Write a driver in `flycar/drivers/contrib/vehicle_driver.py` that uses only the observation (never `info`, never `pychrono`). Since you know the car best, an obvious choice is a light-seeker that brakes into turns using `imu` speed and yaw rate. It will be run across many random seeds on every team's scenario and compared with the Braitenberg baseline.

## What other teams may ask you for
- **Vision** may need the eye mount to stay valid if you change the car's body or ride height; tell them before you merge anything that moves `chassis_body`.
- **Core** reads `position()`, `velocity()`, `rotation_matrix()` every step; keep them cheap.
- **Tools** may ask for more fields in your `extra_info()` for the dashboard.

## Pull request checklist
- [ ] Only files from "What you own" changed.
- [ ] `pytest` passes locally, including `tests/test_baseline.py` and `tests/test_contracts.py`.
- [ ] Every new config key has a comment, a default that keeps today's behavior, and a validation check.
- [ ] The "not implemented yet" test for any feature you implemented was replaced with a real test.
- [ ] `docs/teams/vehicle.md` describes the new settings with an example YAML snippet.
