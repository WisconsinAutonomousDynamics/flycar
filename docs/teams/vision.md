# Team: Vision (`vision`)

This page belongs to the team. Keep it up to date here instead of in the README.
The Phase 1 work guide (tasks, parts A to C) is [`teams/team-vision.md`](../../teams/team-vision.md).

## What you own

| Path | What it is |
|---|---|
| `flycar/env/sensing/eyes.py` | The compound eyes: `RaycastEyes` (CPU) and `SensorEyes` (GPU) |
| `flycar/env/sensing/sensor_probe.py` | Tests, in a separate process, whether the GPU backend works |
| `flycar/config/vision.py` | Scenario section `eyes`: defaults and checks |
| `tests/test_vision.py` | Your tests |
| `scenarios/team_vision.yaml` | Your scratch scenario |
| `flycar/drivers/contrib/vision_driver.py` | Your capstone driver (observation only, no pychrono) |
| `teams/team-vision.md` | Your Phase 1 work guide |
| `docs/teams/vision.md` | This page |

## Contracts you must not break

Changing these needs a lead-approved `api-change` issue (see
[`docs/CONTRACTS.md`](../CONTRACTS.md)):

- **The eyes interface** (contract 5): `update()` (called every physics step),
  `render() -> (left, right)` (float32 `(rows, cols)` in [0, 1]) and
  `raw_images()`.
- **The observation** (contract 1): `eye_left` and `eye_right` as defined in
  `flycar/env/spaces.py`.

You depend on: `World.current_lights()` and `World.layout["lighting"]`, and
the `FlyCarVehicle` accessors `position()`, `rotation_matrix()` and
`chassis_body` (for camera mounting).

## Your reserved config

```yaml
eyes:
  noise_std: 0.0       # planned: pixel noise
  motion_blur: false   # planned: blur while turning
```

Any other value is rejected with "not implemented yet" until you implement it.

## Things to know

- Use the `raycast` backend for anything others must reproduce; all tests and
  the baseline use it. CI forces it with `FLYCAR_EYE_BACKEND=raycast`.
- Never read `layout["lights"]` directly: ask `world.current_lights()`.
- To expose debug data (say, ray timings), add `extra_info()` to the eyes
  class; it shows up in `info["extras"]["vision"]`.

## Running your tests

```
pytest tests/test_vision.py
python run.py --scenario scenarios/team_vision.yaml --driver braitenberg --dashboard
```

## Backend parity (A2)

With default settings the GPU (`sensor`) eyes do not see the same world as the
CPU (`raycast`) eyes, and a Braitenberg driver steers **away** from a lamp on
the sensor backend. These scenario settings make the two agree:

```yaml
eyes:
  sensor_linearize: true       # undo the camera's sRGB encoding
  sensor_light_falloff: true   # lamps get dimmer with distance
  sensor_light_range: 5.0      # how far the lamp light reaches, x falloff_m
  sensor_lamp_glow: true       # draw the same lamp glow as the raycast eyes
  # sensor_brightness_scale stays 1.0
```

All five settings default to today's sensor images, so nothing changes unless a
scenario asks for them. They have no effect on the raycast backend.

### What was wrong, and the fix for each

| Mismatch | Evidence | Setting |
|---|---|---|
| The camera stores sRGB-encoded values; raycast works in linear light. | Night sky set to 0.04 read 0.231 on the camera, 0.044 after decoding; day sky 0.20 read 0.478, 0.195 after decoding. | `sensor_linearize` |
| Chrono's point lights keep the same brightness at any distance unless `const_color=False`; `max_range` alone does nothing. | Walls 10 m from a lamp were nearly white on the GPU and dim on raycast. Images were bit-identical for every `max_range` until `const_color=False`. | `sensor_light_falloff`, `sensor_light_range` |
| The lamp ball is plain yellow paint on the GPU, not a light, so the brightest pixels were the lit walls. | Braitenberg's left-minus-right signal for a lamp on the left: raycast +0.52, sensor -0.04 (wrong way). | `sensor_lamp_glow` |

The glow is the same arithmetic as the raycast backend's (`lamp_glow()` in
`eyes.py`), drawn from the same pixel rays, with the same ray check for lamps
hidden behind walls or obstacles.

### How the values were chosen

Eight fixed scenes (car at the arena centre facing +x; lamp ahead, left, right,
close, hidden behind a box; no lamp; night and day lighting), rendered with both
backends. `sensor_light_range` was swept over 0.5 to 10 and the brightness scale
over 0.5 to 2.0, minimising the mean pixel error. Ranges 5 to 7 were equally
good (mean error 0.029 to 0.031); 5.0 with scale 1.0 has the smallest error in
the driver's left-minus-right signal and needs no scale.

| Settings | Mean pixel error | Left-minus-right error | Turns toward a lamp on one side |
|---|---|---|---|
| Default (uncalibrated) | 0.22 to 0.32 per scene | 0.47 to 0.56 | 0 of 3 scenes |
| sRGB decode, constant light, glow, best scale (0.70) | 0.055 | 0.101 | 3 of 3 |
| **Calibrated (above)** | **0.031** | **0.017** | **3 of 3** |

### The parity test

`test_calibrated_sensor_eyes_match_the_raycast_eyes` in `tests/test_vision.py`
renders six scenes on both backends with the calibrated settings and checks:

| Check | Tolerance | Worst measured |
|---|---|---|
| Mean \|sensor - raycast\| over both eyes | 0.08 | 0.058 (lamp 4 m away) |
| Braitenberg steering command, sensor vs raycast | 0.1 | 0.00 (identical in every scene) |
| Brightest pixel of a visible lamp | 1 pixel | 1 pixel |

It needs a working GPU, so it is **skipped in CI** (`FLYCAR_EYE_BACKEND=raycast`)
and on machines where `sensor_probe.py` fails. To run it on a GPU machine, make
sure `FLYCAR_EYE_BACKEND` is not set, then:

```
pytest tests/test_vision.py -k calibrated
```

### Driver-level check

Braitenberg, 20 seeds of `light_seek.yaml` (same layouts for every row):

| Eyes | Reached the lamp | Seeds that failed |
|---|---|---|
| raycast (reference) | 19 of 20 | 14 |
| sensor, default settings | 1 of 20 | all but 2 |
| **sensor, calibrated** | **17 of 20** | 7, 12, 18 |

Of the 16 seeds that both runs complete, 13 reach the lamp within 0.1 s of the
raycast time, so most episodes drive almost the same path. The remaining
differences are single episodes where a small brightness difference sends the
car round the other side of an obstacle (seed 14 fails on raycast but succeeds
on the GPU; seeds 7, 12 and 18 the other way round).

To repeat it, add the `eyes` settings above to a copy of `light_seek.yaml`
(the frozen scenarios must not change) and run
`python run.py --scenario <copy> --driver braitenberg --episodes 20 --eyes sensor`.

### Known differences that remain

- The lamp ball is still drawn as a dull ball under the glow, and the sensor's
  light sits just above the ball, so the ground right under a lamp is in its
  shadow. Making the ball emissive would be cleaner but it is World's file
  (`world.py`).
- Surfaces keep their own colours on the GPU (walls 0.75 grey, obstacles
  brown, textured ground), while raycast uses one albedo for everything. The
  differences are small after calibration but not zero.
- The bottom rows show the Gator's 3D body on the GPU and the bumper's
  collision box on raycast.
- Measured on one machine (Windows 11, RTX 4060 Laptop, driver 617.14).

### Windows: GPU eyes need the shader-path fix

The PyChrono 10 conda package for Windows installs the sensor shaders in
`<env>\Library\sensor_shaders\` but looks for them in
`<env>\Library\lib\sensor_ptx\`, so every camera failed with "PTX file not
found" and the eyes silently fell back to raycast. `use_installed_shaders()` in
`sensor_probe.py` points Chrono at the installed folder only when the default
one is missing the shaders; `SensorEyes` uses the same helper. Other platforms
are unaffected.
