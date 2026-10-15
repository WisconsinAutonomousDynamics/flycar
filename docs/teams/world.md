# Team: World (`world`)

This page belongs to the team. Keep it up to date here instead of in the README.
The Phase 1 work guide (tasks, parts A to C) is [`teams/team-world.md`](../../teams/team-world.md).

## What you own

| Path | What it is |
|---|---|
| `flycar/env/world.py` | `sample_layout()` (rolls the dice) and `World` (walls, obstacles, lamps) |
| `scenarios/` | Every scenario file **except** the frozen baselines `empty_arena.yaml` and `light_seek.yaml` and the other teams' `team_*.yaml` scratch files |
| `flycar/config/world.py` | Scenario sections `arena`, `lighting`, `lights`, `obstacles`, `motion`: defaults and checks |
| `tests/test_world.py` | Your tests |
| `scenarios/team_world.yaml` | Your scratch scenario |
| `flycar/drivers/contrib/world_driver.py` | Your capstone driver (observation only, no pychrono) |
| `teams/team-world.md` | Your Phase 1 work guide |
| `docs/teams/world.md` | This page |

## Contracts you must not break

Changing these needs a lead-approved `api-change` issue (see
[`docs/CONTRACTS.md`](../CONTRACTS.md)):

- **The `layout` dict** returned by `sample_layout()` (contract 2): every key
  and type. The logger, replay, dashboard and eyes all read it.
- **`World.current_lights()`** and **`World.layout`** (contract 4). The eyes
  ask `current_lights()` every time they take a picture, so moving lamps only
  needs a change here.
- **The tag scheme** (contract 4): walls and obstacles get tags
  `>= WORLD_TAG_BASE`; the contact sensor relies on it.
- **The step order in `core._physics_step()`** (contract 7):
  `World.synchronize(time)` / `World.advance(step)` are where moving things
  will be updated.

## Your reserved config

```yaml
motion:
  obstacles: []   # planned: moving obstacles
  lights: []      # planned: moving lights
```

Anything but empty lists is rejected with "not implemented yet" until you
implement it.

## Things to know

- `empty_arena.yaml` and `light_seek.yaml` are frozen: the baseline test
  replays them. Make a new file for a new experiment.
- To expose debug data, add `extra_info()` to `World`; it shows up in
  `info["extras"]["world"]`.

## Running your tests

```
pytest tests/test_world.py
python run.py --scenario scenarios/team_world.yaml --driver braitenberg --dashboard
```
