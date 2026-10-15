# Team Vision (`vision`): Work Guide

## Your mission
You own the fly's eyes. Two backends produce the same compound-eye image: `RaycastEyes` (CPU, any laptop) and `SensorEyes` (Chrono's GPU camera, needs an NVIDIA GPU with OptiX).

The eyes are the most important input the connectome brain will get, and today they are also the slowest part of every simulation step. By the end of Phase 1 the eyes should be **fast**, the two backends should **agree**, and the image should look **more like what a fly sees**, all without changing the observation's shape or meaning.

## What you own
| Path | What it is |
|---|---|
| `flycar/env/sensing/eyes.py` | Both eye backends and the shared camera model |
| `flycar/env/sensing/sensor_probe.py` | Checks, in a separate process, whether the GPU backend works |
| `flycar/config/vision.py` | Scenario section `eyes` |
| `tests/test_vision.py` | Your tests |
| `scenarios/team_vision.yaml` | Your scratch scenario |
| `docs/teams/vision.md` | Your team page |
| `flycar/drivers/contrib/vision_driver.py` | Your capstone driver (Part C) |

Do not edit any other file. If you need something from another team, open a `cross-team-request` issue.

## Before you start (everyone on the team)
1. Finish all three tutorials.
2. Read `flycar/env/sensing/eyes.py` top to bottom, especially `pinhole_ray_directions()` and `RaycastEyes._render_one_eye()`. Make sure you can explain how one pixel's brightness is computed.
3. Read contracts 1 and 5 in `docs/CONTRACTS.md`, and the `eye_left` / `eye_right` rows in `docs/API.md`.
4. Find out who on the team has an NVIDIA GPU: run `python flycar/env/sensing/sensor_probe.py`. That person owns the GPU work below.
5. Run:
   ```
   pytest tests/test_vision.py tests/test_baseline.py
   python run.py --scenario scenarios/team_vision.yaml --driver braitenberg --dashboard
   ```

## Rules specific to your team
- **The observation is frozen:** `eye_left` and `eye_right` are float32 `(rows, cols)` in [0, 1], row 0 at the top, column 0 at the left. Everything you do must keep that true. A different resolution is fine (it is a scenario setting); a different shape or meaning is an `api-change`.
- **The eyes interface is frozen:** `update()`, `render() -> (left, right)`, `raw_images()`, created by `make_eyes(backend, system, vehicle, world, config)`. You may add an *optional* argument with a default.
- **Raycast is the reference backend.** CI and the baseline force it (`FLYCAR_EYE_BACKEND=raycast`). Any default change to raycast output breaks `test_baseline.py`; put changes behind new `eyes` keys.
- **Lights come from `world.current_lights()`**, every time. Never cache positions across frames and never read `layout["lights"]` for positions; World will make lamps move.
- **Determinism:** with `noise_std: 0` and the raycast backend, the same seed must give the same image, bit for bit.
- Debug data (timings, ray counts) goes in `extra_info()` on the eyes class and appears in `info["extras"]["vision"]`.

---

## Part A: Fast and trustworthy

### A1. Profile and speed up the raycast backend (one or two people)
**Goal:** make the CPU eyes several times faster so laptops can run many episodes.

1. **Measure first.** Add timing to `RaycastEyes` and report it through `extra_info()`: milliseconds per `render()`, split into primary rays, shadow rays and lamp glow; number of rays cast. Write the baseline numbers on your team page for `light_seek.yaml` at 20×20.
2. **Then optimize**, measuring after every change. Ideas, roughly in order of payoff:
   - Shadow rays are cast per pixel per lamp. Skip them where the lamp's contribution is already too small to matter (partly done), or compute shadows on a coarser grid and upsample.
   - Pixels that hit the flat ground near the car change slowly; consider whether anything can be reused between frames (be careful: it must stay exactly deterministic).
   - Reduce Python overhead in the inner loop (attribute lookups, list conversions).
   - Check whether your Chrono version offers a batched ray query; if it does, try it.
3. Any optimization that changes pixel values even slightly goes behind a new config key (e.g. `eyes.shadow_resolution`), default = today's exact output.

**Chrono you learn:** the collision system and `RayHit`, collision shapes vs visual shapes.

**Done when:** `render()` time at 20×20 on `light_seek.yaml` is at least 3× faster with default settings and `test_baseline.py` still passes, or you have documented why the remaining time is unavoidable.

### A2. Backend parity (the GPU person, with one helper)
**Goal:** the two backends should show "the same world", so a driver tuned on one works on the other.

- Write a test (skipped when the sensor backend is unavailable) that builds the same scene with both backends at a fixed car pose and compares images: where the lamp appears (brightest pixel location), and the overall brightness ranking of regions.
- Today brightness is not calibrated between backends. Find a brightness scale (and maybe a gamma) for `SensorEyes` that makes the images match as closely as possible, and add it as a config key.
- Check the image orientation carefully on both: row 0 = top, column 0 = left, left eye turned left.

**Done when:** the parity test passes with documented tolerances, and the Braitenberg driver's success rate on 20 seeds of `light_seek.yaml` is within a few episodes between the two backends.

### A3. Eye visualizer (one person)
**Goal:** see what the eyes see, from the eye's own point of view.

Write a small script (put it at the bottom of `eyes.py` under `if __name__ == "__main__":`, so it stays in your file) that builds an env, parks the car at a given pose, and saves a figure with: both eye images, the ray hit points drawn on a top-down arena map, and which pixels are in shadow. This is your main debugging tool for Part B.

---

## Part B: More like a fly

### B1. Pixel noise (one person)
**Goal:** implement the reserved `eyes.noise_std` key.

- Add Gaussian noise with that standard deviation to every pixel, then clip to [0, 1].
- Noise needs a seeded random generator, and the eyes are not given one today. Open a `cross-team-request` to Core asking them to pass `rng=self.np_random` into `make_eyes(...)` as an optional argument. Until that lands, write and test the code with an `rng` argument that defaults to `None`.
- Remove `noise_std` from `NOT_YET_IMPLEMENTED` and replace the "not implemented yet" test.

**Done when:** `noise_std: 0` gives exactly today's images; a non-zero value gives noisy images that are identical for the same seed and different for different seeds.

### B2. Motion blur (one person)
**Goal:** implement the reserved `eyes.motion_blur` key.

Real eyes integrate light over a short time, so fast turns smear the image. One approach: when enabled, `update()` (called every physics step) accumulates a few intermediate frames or a running average, and `render()` returns the blurred result. Keep the cost in check; measure it with A1's timers.

**Done when:** blur is off by default (baseline unchanged), and with it on, a test shows a smeared lamp while the car spins in place and a sharp one while it stands still.

### B3. A fly-like eye layout (one or two people)
**Goal:** sample the world the way a fly's compound eye does, without changing the `(rows, cols)` shape.

Ideas to explore and present to the team:
- A wider total field of view with a small overlap in the middle (binocular zone). Today each eye is a 120° pinhole turned out by 50°; try other `fov_h_deg` / `yaw_offset_deg` combinations.
- **Non-uniform sampling:** more pixels pointing forward, fewer at the sides, by replacing `pinhole_ray_directions()` with an equal-angle or custom mapping behind a new `eyes.projection` key (`pinhole` default, `equiangular`, ...). The GPU backend would need the same mapping (sample its high-res image accordingly).
- Pixel acceptance angle: each "ommatidium" averages a small cone, not a single ray (cast a few rays per pixel and average).

**Done when:** at least one new projection is implemented in the raycast backend behind a config key, documented with pictures from A3's visualizer, and the Braitenberg driver still works with it.

---

## Part C: Capstone driver
Write `flycar/drivers/contrib/vision_driver.py`, using only the observation. You know the images best, so use more than the brightest pixels: for example, steer away from large dark regions (obstacles) while steering toward the lamp.

## What other teams may ask you for
- **World** will make lamps move; your code must keep working when `current_lights()` returns new positions every frame.
- **Vehicle** may change the car's body; the eye mount is relative to `chassis_body`.
- **Core** will profile the whole step; your `extra_info()` timings help them.
- **Tools** shows eye images on the dashboard and in replays.

## Pull request checklist
- [ ] Only files from "What you own" changed.
- [ ] `pytest` passes, including `tests/test_baseline.py` and `tests/test_contracts.py`.
- [ ] Output shape, dtype, range and orientation are unchanged.
- [ ] New config keys default to today's exact output and have validation checks.
- [ ] Timing numbers before/after are on `docs/teams/vision.md` for any performance change.
