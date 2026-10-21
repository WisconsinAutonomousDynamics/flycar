"""Record the behaviour baseline in tests/baseline/.

    python scripts/record_baseline.py

Run it on Linux (that is what CI uses). ``tests/test_baseline.py`` reruns
everything recorded here and checks that nothing changed. It is how we enforce
"new features are opt-in, defaults are unchanged": if a pull request changes
the behaviour of a default scenario, the baseline test fails.

There are three parts, because physics results are not identical across
operating systems. PyChrono's Windows, macOS and Linux builds round the last
digit differently, and a car driving for a while turns a 1e-12 difference into
centimetres (and a closed-loop episode into a different number of steps).

``setup.json`` (checked exactly, on every OS)
    Everything that goes INTO the simulation: every scenario default, the
    frozen scenarios as loaded, the layouts they produce for a few seeds, the
    tunable constants of ``flycar/env``, and what Chrono was actually built
    with (solver, bodies in creation order with their mass, inertia and
    contact materials, the car's geometry and tires). Plain Python and Chrono
    getters, no time integration, so it is the same everywhere.

``maneuver.npz`` (checked loosely, on every OS)
    One drive with a fixed list of actions (accelerate, turn left, turn right,
    brake) on flat ground. No driver feedback and no walls, so platform
    differences stay at a few centimetres, while changing a default such as the
    ground friction or the top speed moves the car by tens of centimetres.

``<episode>.npz`` (checked to 1e-6, on Linux only)
    Full closed-loop reference episodes. These catch the smallest changes, but
    only reproduce on the OS they were recorded on, so CI runs them.

**Do not re-record to make a failing baseline test pass.** A failure means the
code now behaves differently. Find out why. Only the project leads re-record,
and only when a behaviour change was agreed on in an ``api-change`` issue.

What is stored, every ``RECORD_EVERY``-th control step (step 0 is the state
right after ``reset()``):

    step              control step number
    pose              info["pose"]            [x, y, yaw]
    applied_action    info["applied_action"]  [throttle, steering, brake]
    imu               obs["imu"]
    contact           obs["contact"]
    eye_left_sum      checksums of each eye image (see eye_checksums below)
    eye_left_wsum
    eye_right_sum
    eye_right_wsum

and, once per episode: the number of steps, and whether it ended terminated,
truncated or with success.
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))   # so "import flycar" works from anywhere

BASELINE_DIR = REPO_ROOT / "tests" / "baseline"

# Store one sample every this many control steps.
RECORD_EVERY = 10

# (file name, scenario file, driver name, seed). Always headless and raycast.
BASELINE_RUNS = [
    ("light_seek_braitenberg_seed0", "scenarios/light_seek.yaml", "braitenberg", 0),
    ("light_seek_braitenberg_seed1", "scenarios/light_seek.yaml", "braitenberg", 1),
    ("light_seek_braitenberg_seed2", "scenarios/light_seek.yaml", "braitenberg", 2),
    ("empty_arena_random_seed0", "scenarios/empty_arena.yaml", "random", 0),
]


# The scenarios whose setup is recorded. Only the frozen (lead-owned) ones: the
# team_*.yaml scenarios are the teams' own and are meant to change.
SETUP_SCENARIOS = ["scenarios/empty_arena.yaml", "scenarios/light_seek.yaml"]

# Layouts are recorded for these seeds.
SETUP_SEEDS = [0, 1, 2, 3, 4]

# Modules whose UPPER_CASE constants (numbers, strings, flags) are recorded.
CONSTANT_MODULES = [
    "flycar.env.core",
    "flycar.env.vehicle",
    "flycar.env.terrain",
    "flycar.env.world",
    "flycar.env.sensing.eyes",
    "flycar.env.sensing.contact",
]

# The maneuver: a scenario with only defaults (except a fixed spawn and enough
# time), and a list of (action, number of control steps).
MANEUVER_SCENARIO = {
    "name": "baseline_maneuver",
    "spawn": {"x": 0.0, "y": 0.0, "yaw_deg": 0.0},
    "episode": {"max_time_s": 60.0},
}
MANEUVER_PLAN = [
    ([1.0, 0.0, 0.0], 300),    # accelerate for 3 s
    ([1.0, 1.0, 0.0], 200),    # full left for 2 s
    ([1.0, 0.0, 0.0], 100),    # straight for 1 s
    ([1.0, -1.0, 0.0], 200),   # full right for 2 s
    ([0.0, 0.0, 1.0], 200),    # brake for 2 s
]
# Store one maneuver sample every this many control steps.
MANEUVER_RECORD_EVERY = 50


def eye_checksums(image):
    """Two numbers that change if any pixel of an eye image changes.

    ``sum`` is the total brightness. ``wsum`` weights every pixel by its
    position (1, 2, 3, ... row by row), so that two pixels swapping values also
    shows up. Floats rather than a hash, so the test can allow for the last
    digit of rounding.
    """
    pixels = np.asarray(image, dtype=np.float64).ravel()
    weights = np.arange(1, len(pixels) + 1, dtype=np.float64)
    return float(pixels.sum()), float(np.dot(pixels, weights))


def run_episode(scenario_path, driver_name, seed):
    """Run one full episode headless with raycast eyes. Returns a dict of arrays."""
    from flycar.drivers import make_driver
    from flycar.env.core import FlyCarEnv

    env = FlyCarEnv(REPO_ROOT / scenario_path, render_mode=None, eye_backend="raycast")
    driver = make_driver(driver_name)

    samples = {
        "step": [], "pose": [], "applied_action": [], "imu": [], "contact": [],
        "eye_left_sum": [], "eye_left_wsum": [], "eye_right_sum": [], "eye_right_wsum": [],
    }

    def record(observation, info):
        samples["step"].append(int(info["step"]))
        samples["pose"].append(np.array(info["pose"], dtype=np.float64))
        samples["applied_action"].append(np.array(info["applied_action"], dtype=np.float64))
        samples["imu"].append(np.array(observation["imu"], dtype=np.float64))
        samples["contact"].append(np.array(observation["contact"], dtype=np.float64))
        left_sum, left_wsum = eye_checksums(observation["eye_left"])
        right_sum, right_wsum = eye_checksums(observation["eye_right"])
        samples["eye_left_sum"].append(left_sum)
        samples["eye_left_wsum"].append(left_wsum)
        samples["eye_right_sum"].append(right_sum)
        samples["eye_right_wsum"].append(right_wsum)

    observation, info = env.reset(seed=seed)
    driver.reset()
    record(observation, info)

    while True:
        action = driver.act(observation)
        observation, _, terminated, truncated, info = env.step(action)
        if info["step"] % RECORD_EVERY == 0:
            record(observation, info)
        if terminated or truncated:
            break
    env.close()

    result = {name: np.array(values) for name, values in samples.items()}
    result["n_steps"] = np.array(int(info["step"]))
    result["terminated"] = np.array(bool(terminated))
    result["truncated"] = np.array(bool(truncated))
    result["success"] = np.array(bool(info["success"]))
    return result


def _plain(value):
    """Turn tuples and NumPy numbers into plain JSON-friendly Python values."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _is_simple(value):
    """A number, string or flag, or a flat tuple/list of numbers.

    Not lists of names (like ``EXTRA_INFO_SOURCES``): those are registries
    that may grow, which does not change any behaviour.
    """
    if isinstance(value, (list, tuple)):
        return all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value)
    return isinstance(value, (bool, int, float, str))


def module_constants():
    """The UPPER_CASE simple constants of every module in CONSTANT_MODULES."""
    import importlib

    constants = {}
    for module_name in CONSTANT_MODULES:
        module = importlib.import_module(module_name)
        constants[module_name] = {
            name: _plain(value) for name, value in vars(module).items()
            if name.isupper() and _is_simple(value)
        }
    return constants


def _vector(v):
    return [v.x, v.y, v.z]


def chrono_setup(scenario_path, seed=0):
    """What Chrono is built with after ``reset()``: settings, bodies, car.

    Only things that do not depend on time integration (masses, materials,
    where the fixed bodies stand), so the result is the same on every OS.
    """
    from flycar.env.core import FlyCarEnv

    env = FlyCarEnv(REPO_ROOT / scenario_path, render_mode=None, eye_backend="raycast")
    env.reset(seed=seed)
    system = env.system

    bodies = []
    for body in system.GetBodies():
        entry = {
            "tag": body.GetTag(),
            "fixed": body.IsFixed(),
            "mass": body.GetMass(),
            "inertia_xx": _vector(body.GetInertiaXX()),
            "collision": body.IsCollisionEnabled(),
        }
        # Bodies Chrono names itself ("_obj_67") get a counter that depends on
        # how many objects were created before; only keep real names.
        if not body.GetName().startswith("_obj_"):
            entry["name"] = body.GetName()
        if body.IsFixed():
            entry["position"] = _vector(body.GetPos())
        model = body.GetCollisionModel()
        if model is not None:
            entry["shapes"] = []
            for i in range(model.GetNumShapes()):
                shape = model.GetShapeInstance(i).shape
                material = shape.GetMaterial()
                entry["shapes"].append({
                    "type": int(shape.GetType()),
                    "friction": material.GetStaticFriction(),
                    "restitution": material.GetRestitution(),
                })
        bodies.append(entry)

    vehicle = env.vehicle.vehicle
    tires = []
    for axle in range(vehicle.GetNumberAxles()):
        for side in (0, 1):   # left, right
            tire = vehicle.GetTire(axle, side)
            # (Not GetRadius(): that is the loaded radius, which depends on the state.)
            tires.append({
                "model": tire.GetTemplateName(),
                "width": tire.GetWidth(),
                "mass": tire.GetTireMass(),
                "inertia": _vector(tire.GetTireInertia()),
                "step": tire.GetStepsize(),
            })

    result = {
        "solver_type": int(system.GetSolverType()),
        "solver_max_iterations": system.GetSolver().AsIterative().GetMaxIterations(),
        "timestepper_type": int(system.GetTimestepperType()),
        "gravity": _vector(system.GetGravitationalAcceleration()),
        "threads": [system.GetNumThreadsChrono(), system.GetNumThreadsCollision(), system.GetNumThreadsEigen()],
        "substeps": env.substeps,
        "eye_every": env.eye_every,
        "n_links": system.GetNumLinks(),
        "n_shafts": system.GetNumShafts(),
        "n_other_items": len(system.GetOtherPhysicsItems()),
        "bodies": bodies,
        "vehicle": {
            "mass": vehicle.GetMass(),
            "wheelbase": vehicle.GetWheelbase(),
            "wheeltrack": [vehicle.GetWheeltrack(axle) for axle in range(vehicle.GetNumberAxles())],
            "max_steering_angle": vehicle.GetMaxSteeringAngle(),
            "min_turning_radius": vehicle.GetMinTurningRadius(),
            "tires": tires,
        },
    }
    env.close()
    return _plain(result)


def setup_snapshot():
    """Everything that goes into the simulation, as plain JSON-friendly data."""
    from gymnasium.utils import seeding

    from flycar.config import DEFAULTS, ITEM_DEFAULTS
    from flycar.env.scenario import load_scenario
    from flycar.env.world import sample_layout

    scenarios = {}
    for scenario_path in SETUP_SCENARIOS:
        scenario = load_scenario(REPO_ROOT / scenario_path)
        layouts = {}
        for seed in SETUP_SEEDS:
            rng, _ = seeding.np_random(seed)   # the same generator env.reset(seed) uses
            layouts[str(seed)] = sample_layout(scenario, rng)
        scenarios[scenario_path] = {
            "scenario": scenario,
            "layouts": layouts,
            "chrono": chrono_setup(scenario_path),
        }

    return _plain({
        "defaults": DEFAULTS,
        "item_defaults": ITEM_DEFAULTS,
        "constants": module_constants(),
        "scenarios": scenarios,
    })


def run_maneuver():
    """Drive MANEUVER_PLAN open loop. Returns a dict of arrays."""
    from flycar.env.core import FlyCarEnv

    env = FlyCarEnv(dict(MANEUVER_SCENARIO), render_mode=None, eye_backend="raycast")
    samples = {"step": [], "pose": [], "applied_action": [], "imu": []}

    def record(observation, info):
        samples["step"].append(int(info["step"]))
        samples["pose"].append(np.array(info["pose"], dtype=np.float64))
        samples["applied_action"].append(np.array(info["applied_action"], dtype=np.float64))
        samples["imu"].append(np.array(observation["imu"], dtype=np.float64))

    observation, info = env.reset(seed=0)
    record(observation, info)
    for action, n_steps in MANEUVER_PLAN:
        for _ in range(n_steps):
            observation, _, _, _, info = env.step(action)
            if info["step"] % MANEUVER_RECORD_EVERY == 0:
                record(observation, info)
    env.close()
    return {name: np.array(values) for name, values in samples.items()}


def main():
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    if not sys.platform.startswith("linux"):
        print("Record the baseline on Linux: the full episodes only reproduce on the OS they were recorded on, "
              "and CI runs Linux.")
        return 1

    path = BASELINE_DIR / "setup.json"
    path.write_text(json.dumps(setup_snapshot(), indent=1, sort_keys=True) + "\n")
    print(f"{path.relative_to(REPO_ROOT)}")

    path = BASELINE_DIR / "maneuver.npz"
    np.savez_compressed(path, **run_maneuver())
    print(f"{path.relative_to(REPO_ROOT)}")

    for name, scenario_path, driver_name, seed in BASELINE_RUNS:
        result = run_episode(scenario_path, driver_name, seed)
        path = BASELINE_DIR / f"{name}.npz"
        np.savez_compressed(path, **result)
        print(f"{path.relative_to(REPO_ROOT)}: {int(result['n_steps'])} steps, "
              f"success={bool(result['success'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
