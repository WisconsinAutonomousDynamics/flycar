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

## Performance measurements

CPU raycast optimization, measured with `scenarios/light_seek.yaml` at 20×20.
Each result is 200 fixed-pose renders after 20 warm-up renders. “Before” is
`origin/main`; “after” is this implementation. Values are wall-clock ms per
`render()` on Python 3.12.15, PyChrono 10.0.0, macOS arm64.

| Seed | Rays | Median before | Median after | Change | p95 before | p95 after |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1,175 | 1.105 ms | 0.970 ms | -12.2% | 1.268 ms | 1.022 ms |
| 1 | 1,220 | 1.240 ms | 1.118 ms | -9.8% | 1.425 ms | 1.190 ms |
| 2 | 1,219 | 1.141 ms | 1.045 ms | -8.4% | 1.224 ms | 1.107 ms |

The after measurements split the median render time as follows:

| Seed | Total | Primary rays | Shadow rays | Lamp glow |
|---:|---:|---:|---:|---:|
| 0 | 0.985 ms | 0.627 ms | 0.266 ms | 0.021 ms |
| 1 | 1.119 ms | 0.609 ms | 0.416 ms | 0.021 ms |
| 2 | 1.036 ms | 0.584 ms | 0.359 ms | 0.020 ms |

Reproduce the after measurements with:

```
python -m scripts.benchmark_eyes
```

## Running your tests

```
pytest tests/test_vision.py
python run.py --scenario scenarios/team_vision.yaml --driver braitenberg --dashboard
```
