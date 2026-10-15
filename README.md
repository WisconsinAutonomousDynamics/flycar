# Fly Car

A car in the [Project Chrono](https://projectchrono.org/) physics simulator that
will, eventually, be driven by a fruit-fly brain model built from the
[FlyWire](https://flywire.ai/) connectome.

**This repository is the first half of that: the "fly-ready" car.** It is a
simulated vehicle with fly-like senses (two low-resolution compound eyes, an
inertial sensor, touch sensors) and a small, stable API that any controller can
drive. The brain comes later and plugs in as just another controller.

It is also where we learn [PyChrono](https://api.projectchrono.org/pychrono_introduction.html),
starting with the three scripts in `tutorials/`.

```
python run.py --scenario scenarios/light_seek.yaml --driver braitenberg --dashboard
```

That command drops the car into a dark arena and lets a tiny "Braitenberg"
controller (a few lines of arithmetic) steer it to a lamp using nothing but its
two eyes.

## Setup

There are two ways to get a working environment. Both give you the same
packages. Pick one.

| | Use it if |
|---|---|
| **A. Conda** | You are on Windows, Linux or an Apple-silicon Mac and want 3D windows to just work. Recommended. |
| **B. Devcontainer** | You have an Intel Mac, you already use Docker, or conda is giving you trouble. Runs headless by default. |

### A. Conda

1. Install [Miniforge](https://conda-forge.org/download/) (or use an Anaconda /
   Miniconda you already have).
2. In a terminal, from this folder:

   ```
   conda env create -f environment.yml
   conda activate flycar
   python scripts/check_env.py
   ```

The first command downloads several GB and uses about 6 GB of disk (PyChrono
ships with GPU libraries whether or not you have a GPU). Expect it to take
5 to 15 minutes.

`check_env.py` should end with `RESULT: OK`. It lists four Chrono modules:

- `pychrono` and `pychrono.vehicle` are **required**.
- `pychrono.irrlicht` gives you the 3D window. Nice to have.
- `pychrono.sensor` gives you GPU cameras, and only really works with an NVIDIA
  GPU. **You do not need it.** Without it the eyes use a CPU method instead
  (see [Eyes](#eyes-two-backends)).

**Run `conda activate flycar` in every new terminal.** On Windows, forgetting
it does not produce an error: `import pychrono` simply hangs forever.

### B. Devcontainer

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
   (on Windows, with the WSL 2 backend) and [VS Code](https://code.visualstudio.com/)
   with the **Dev Containers** extension.
2. Open this folder in VS Code and run **Dev Containers: Reopen in Container**
   from the command palette.

The first build takes a while. When it finishes, `check_env.py` runs
automatically and prints its report in the terminal. `python` and `pytest` are
already on the path; there is nothing to activate.

- **GPU:** if the host has an NVIDIA GPU and the NVIDIA Container Toolkit, the
  container gets the GPU automatically. If not, it starts normally without one.
- **No windows:** the container has no display, so use `--log` and the replay
  tool instead of `--dashboard` and `--render`.

## Start here: the tutorials

Three short, heavily commented scripts. Read them top to bottom, run them,
then try the "things to try" listed at the top of each file.

| Script | What it teaches |
|---|---|
| `python tutorials/01_falling_box.py` | A `ChSystem`, a body, a floor, the step loop. Compares a falling box with the textbook formula. |
| `python tutorials/02_motor_cart.py` | Several bodies, joints, and motors. A two-wheeled cart steered by wheel speeds. |
| `python tutorials/03_vehicle_figure8.py` | `pychrono.vehicle`, driver inputs, and the Synchronize / Advance pattern. A Gator drives a figure-8. |

Add `--headless` to any of them to skip the windows and save the plot as a PNG.

## Run a scenario

```
python run.py --scenario scenarios/light_seek.yaml --driver braitenberg
```

| Option | What it does |
|---|---|
| `--driver NAME` | `random`, `braitenberg`, `keyboard`, or your own (see below). |
| `--dashboard` | Live matplotlib window: map, both eyes, control traces. |
| `--render` | 3D window that follows the car. |
| `--episodes 20` | Run 20 episodes (seeds 0 to 19) and print the success rate. |
| `--seed 7` | Start from a particular seed. The same seed gives the same episode. |
| `--log` | Save a Parquet log per episode into `logs/`. |
| `--log-extras` | Also store `info["extras"]`, the teams' unstable debug data, in the log. |
| `--eyes raycast` | Force an eye backend (`auto`, `sensor`, `raycast`). |
| `--realtime` | Do not run faster than real time. |

Drive it yourself (W/A/S/D or the arrow keys, with the dashboard window focused):

```
python run.py --scenario scenarios/empty_arena.yaml --driver keyboard
```

Look at a run afterwards:

```
python run.py --scenario scenarios/light_seek.yaml --driver braitenberg --log
python -m flycar.tools.replay logs/light_seek_braitenberg_seed0.parquet
```

The replay shows the path over the arena map, the controls over time, and what
the eyes saw at a few moments. It is the fastest way to find out *why* a
controller did something. Logs are ordinary Parquet files:
`pandas.read_parquet("logs/....parquet")` gives you one row per control step.

### Scenarios

A scenario is a YAML file in `scenarios/`: arena size, walls, obstacles,
lights, where the car spawns, how long an episode lasts. **A new experiment is
a new YAML file, not a code change.** Wherever a scenario wants a position or
an angle you can write a number, or `[min, max]` to have it drawn at random on
every reset. Copy `scenarios/light_seek.yaml` and edit it; every available
setting and its default is listed in `flycar/config/`, one file per team
(`core.py`, `vehicle.py`, `world.py`, `vision.py`).

### Eyes: two backends

| Backend | Needs | Notes |
|---|---|---|
| `raycast` | Nothing. Any laptop. | Casts one ray per eye pixel through Chrono's collision system and shades what it hits. |
| `sensor` | NVIDIA GPU with OptiX | Chrono's GPU camera, averaged down to the eye grid. |

The default, `auto`, uses `sensor` if it works on your machine and `raycast`
otherwise, and prints which one it picked. **Use `--eyes raycast` for anything
you want teammates to reproduce**: the two backends see the same scene but are
not calibrated to produce identical numbers.

## Write a driver

A driver is a class with two methods. It gets the observation, it returns
`[throttle, steering, brake]`:

```python
import numpy as np

class GoStraight:
    def reset(self):
        pass                      # called at the start of each episode

    def act(self, observation):   # called every 10 ms of simulated time
        return np.array([0.5, 0.0, 0.0], dtype=np.float32)
```

Save it as `flycar/drivers/go_straight.py` and run it with

```
python run.py --scenario scenarios/empty_arena.yaml --driver flycar.drivers.go_straight:GoStraight
```

The observation is a dictionary: `eye_left` and `eye_right` (20 × 20
brightness images), `imu` (speed, turn rate, acceleration), `contact` (left,
front, right touch flags) and `time`. **[docs/API.md](docs/API.md) documents
every field with its units** and has a fuller example.

Two rules keep drivers honest and portable:

1. A driver uses only the observation. No ground truth, no simulator objects.
2. A driver never imports `pychrono`.

`flycar/drivers/braitenberg.py` is the best one to read first.

## Tests

```
pytest                   # everything, a minute or two
pytest -m "not slow"     # skip the tutorials and full episodes, about half a minute
```

The tests run headless and check, among other things, that the same seed and
actions reproduce a trajectory bit for bit, that `N` steps advance time by
exactly `N` control periods, and that no driver imports `pychrono`.

Each team has its own test file (`tests/test_vehicle.py`, `test_world.py`,
`test_vision.py`, `test_core.py`, `test_tools.py`). `tests/test_baseline.py`
replays recorded reference runs and fails if a change alters the default
behaviour; `tests/test_contracts.py` guards the frozen interfaces.

If you change `flycar/env/spaces.py`, run `python scripts/gen_api_docs.py` to
update the tables in `docs/API.md` (a test will remind you).

## How the code is organised

Five teams own the code, each with its own page in [`docs/teams/`](docs/teams/):
what it owns, its config section, its tests and scratch scenario, and the
contracts it must not break.

| Team | Owns | Page |
|---|---|---|
| Vehicle & terrain | `flycar/env/vehicle.py`, `flycar/env/terrain.py` | [vehicle.md](docs/teams/vehicle.md) |
| World | `flycar/env/world.py`, `scenarios/` | [world.md](docs/teams/world.md) |
| Vision | `flycar/env/sensing/eyes.py`, `sensor_probe.py` | [vision.md](docs/teams/vision.md) |
| Core & body sensing | `flycar/env/core.py`, `sensing/imu.py`, `sensing/contact.py` | [core.md](docs/teams/core.md) |
| Tools & drivers | `flycar/tools/`, `flycar/drivers/`, `run.py` | [tools.md](docs/teams/tools.md) |

The interfaces between them are frozen and listed in
[`docs/CONTRACTS.md`](docs/CONTRACTS.md); `.github/CODEOWNERS` says who reviews
what. Start reading at `run.py`.

The one design rule: **everything left of the line knows about Chrono,
everything right of it does not.**

```
 Chrono side                      |   Controller side
 env/core, world, vehicle,        |   drivers/, and later the brain
 sensing                          |
        observation  ------------>|
                    <------------ |   action
```

The only thing that crosses the line is the observation dictionary going one
way and the three-number action going back. That is what makes it possible to
swap the controller for a brain without touching the simulator.

## Roadmap

1. **Now: the vehicle environment.** A car with fly-like senses, a stable API,
   scenarios, logging, and a phototaxis baseline to beat.
2. **Next: better behaviours with simple controllers.** The Braitenberg
   baseline drives straight into any obstacle that sits between it and the
   lamp. Avoiding that using `contact` and the eye images is a good first
   project.
3. **Later: the connectome brain.** A FlyWire-based model becomes a `Driver`:
   eye images go in as visual input, motor neuron activity comes out as
   throttle and steering. `docs/API.md` shows the shape of that class.

## Troubleshooting

**`python` hangs with no output on Windows.** The conda environment is not
activated. Run `conda activate flycar`. `python scripts/check_env.py` detects
this and tells you.

**`check_env.py` says `pychrono.sensor` is OK, but the environment prints
`eye backend: raycast`.** Expected on any machine without an NVIDIA GPU. The
module imports but cannot render, so the environment tests it properly and
falls back. To see the underlying error:
`python flycar/env/sensing/sensor_probe.py`.

**`conda env create` cannot find `pychrono`.** PyChrono 10 has no Intel-Mac
build. Use the devcontainer.

**The 3D window is black or will not open.** Run without `--render`; use
`--dashboard`, or `--log` and the replay tool.

**API names in an online tutorial do not exist.** Chrono renames things
between releases (this project pins PyChrono 10.0.0). The demos that match
your installed version are the reference. Find them with:
`python -c "import pychrono, os; print(os.path.join(os.path.dirname(pychrono.__file__), 'demos'))"`
