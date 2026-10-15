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
