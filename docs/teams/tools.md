# Team: Tools & drivers (`tools`)

This page belongs to the team. Keep it up to date here instead of in the README.
The Phase 1 work guide (tasks, parts A to C) is [`teams/team-tools.md`](../../teams/team-tools.md).

## What you own

| Path | What it is |
|---|---|
| `flycar/tools/` | Logger, replay, dashboard, plotting helpers |
| `flycar/drivers/` | The controllers and `make_driver()` (but see the frozen `Driver` protocol below). In `contrib/`, you own `__init__.py`; each `<slug>_driver.py` belongs to that team |
| `run.py` | Runs a driver in a scenario |
| `tests/test_tools.py` | Your tests (tools, drivers, `run.py`) |
| `scenarios/team_tools.yaml` | Your scratch scenario |
| `teams/team-tools.md` | Your Phase 1 work guide |
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

## Batch runner (`flycar/tools/batch.py`)

Runs many seeds of one or more scenarios with one or more drivers, optionally
in parallel, and scores them. With `all` it becomes a leaderboard (see below).

```
python -m flycar.tools.batch --scenario scenarios/light_seek.yaml --driver braitenberg --seeds 0-49 --workers 4 --out outputs/
python -m flycar.tools.batch --scenario all --driver all --seeds 0-19 --workers 4 \
    --out outputs/ --figure outputs/leaderboard.png
```

| Option | Meaning |
|---|---|
| `--scenario` | One scenario YAML, several separated by commas, or `all` for every file in `scenarios/` |
| `--no-team-scenarios` | With `--scenario all`: leave out the `team_*.yaml` scratch scenarios |
| `--driver` | One driver name, several separated by commas, or `all` (default: `braitenberg`) |
| `--seeds` | `0-49` (both ends included), `7`, or a mix such as `0-9,20` (default: `0-9`) |
| `--workers` | Number of processes (default: 1) |
| `--out DIR` | Write `results.csv` (one row per episode), `summary.csv` (one row per scenario and driver) and `leaderboard.csv` (the success-rate grid, best driver first) into `DIR` (nothing is written if omitted) |
| `--figure FILE` | Draw the success-rate grid as an image, e.g. `outputs/leaderboard.png` |
| `--print` | Prints the table to stdout |

The terminal shows one line per (scenario, driver): successes out of episodes,
the success rate with a 95 % Wilson confidence interval, and the median time to
target over the successful episodes (plus a note if some episodes crashed). With
several drivers or scenarios it also prints a table of success rates (rows:
drivers, columns: scenarios). A scenario without a target lamp has no success
rate, so it is reported as "not applicable" and shows as `NaN` in that table.

`results.csv` has one row per episode:

| Column | Meaning |
|---|---|
| `scenario`, `driver`, `seed` | Which episode this is |
| `success` | True if the car reached the target lamp |
| `time_to_target` | Simulation time at success [s]; empty (NaN) if it never got there |
| `target_distance` | Distance to the target at the end [m]; empty if there is no target |
| `contact_steps` | Control steps in which any contact flag was set |
| `path_length` | Distance driven [m], summed over the poses of the episode |
| `duration_s` | Simulation time at the end of the episode [s] |
| `termination_reason` | `success`, `timeout` or `stopped` (from `terminated` / `truncated`, so it works without `info["extras"]`), or `error` if the episode crashed |
| `error` | Empty, or the exception (type and message) of a crashed episode. Its measurements are empty (NaN) and `success` is False |

Every episode builds its own environment and driver, so an episode depends only
on its scenario, driver and seed. The workers are separate processes started
with `spawn` (Chrono objects cannot be shared), and the rows come back in job
order, so the table is the same with 1 worker or 8. The eyes use the `raycast`
backend.

A driver that raises an exception does not stop the batch. The episode becomes a
row with `termination_reason` `error`, it counts as a failure, and the terminal
lists which drivers crashed and with what error, so the owning team can be told.
For the full traceback, run the failing seed with `run.py`
(`python run.py --scenario ... --driver ... --seed N`).

### Leaderboard

`--driver all` runs the built-in drivers that need no person (not `keyboard`)
and every team driver in `flycar/drivers/contrib/`. Together with
`--scenario all` it compares every driver on every scenario. In the figure there
is one row per driver, best first (ranked by mean success rate over the
scenarios that have a target), and one column per scenario. Each cell shows:

| Cell | Meaning |
|---|---|
| `90%` and `18/20` | Success rate, and successes out of episodes |
| `n/a` | The scenario has no target lamp, so there is no success rate |
| `not run` | This driver was not run on this scenario |
| `err` | Number of episodes that crashed (they count as failures) |

Team drivers are found by file name: a module named `<slug>_driver.py` must
contain exactly one class with `reset()` and `act()`, and is run as
`flycar.drivers.contrib.<slug>_driver:<ClassName>`. A module that cannot be
imported, or has no or several driver classes, is reported as a warning and
skipped. A missing `contrib/` folder is fine.

A full leaderboard is every scenario x every driver x every seed, and an episode
that times out runs the full 40 simulated seconds. Start with a few seeds (for
example `--seeds 0-4`) and several `--workers`.

## Running your tests

```
pytest tests/test_tools.py
python run.py --scenario scenarios/team_tools.yaml --driver braitenberg --dashboard
```

