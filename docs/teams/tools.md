# Team: Tools & drivers (`tools`)

This page belongs to the team. Keep it up to date here instead of in the README.

## What you own

| Path | What it is |
|---|---|
| `flycar/tools/` | Logger, replay, dashboard, plotting helpers |
| `flycar/drivers/` | The controllers and `make_driver()` (but see the frozen `Driver` protocol below) |
| `run.py` | Runs a driver in a scenario |
| `tests/test_tools.py` | Your tests (tools, drivers, `run.py`) |
| `scenarios/team_tools.yaml` | Your scratch scenario |
| `docs/teams/tools.md` | This page |

You have no section in `flycar/config/`: tools and drivers do not read
scenario settings.

## Contracts you must not break

Changing these needs a lead-approved `api-change` issue (see
[`docs/CONTRACTS.md`](../CONTRACTS.md)):

- **The `Driver` protocol** in `flycar/drivers/base.py` (contract 8):
  `reset()` and `act(observation) -> [throttle, steering, brake]`.
- **Drivers never import `pychrono`** and only use `flycar.drivers` and
  `flycar.env.spaces` (enforced by `tests/test_tools.py`).
- You read the `layout` dict (contract 2) and the info fields (contract 1);
  do not rely on the contents of `info["extras"]`, which are unstable.

## Things to know

- The logger leaves `info["extras"]` out unless `run.py --log-extras` (or
  `EpisodeLogger(..., log_extras=True)`) is used, so logs stay small.
- `run.py` is the reference loop shown in the README; keep it short.

## Running your tests

```
pytest tests/test_tools.py
python run.py --scenario scenarios/team_tools.yaml --driver braitenberg --dashboard
```
