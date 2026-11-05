"""Vehicle & terrain team: the car, its controls, and the ground.

The last part of this file is the A1 handling test suite: it measures how the
car drives (top speed, 0 -> 3 m/s time, braking distance, turning radius and
steering step response) and checks each number against today's baseline.
See the "A1" section below for details.

Run (from the repository root):

    python -m pytest tests/test_vehicle.py -s                  # everything + A1 table
    python -m pytest tests/test_vehicle.py -s -m "not slow"    # quick tests only (skips the long drives)

or press the VS Code play button on this file.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

# Lets the file also run with VS Code's "Run Python File" button: add the repo
# root to the import path so "import flycar" works. (pytest already does this.)
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from conftest import SCENARIO_FILES, small_scenario  # noqa: E402
from flycar.env import spaces  # noqa: E402
from flycar.env.scenario import ScenarioError, load_scenario  # noqa: E402
from flycar.env.spaces import BRAKE, IMU_SPEED, IMU_YAW_RATE, STEERING, THROTTLE  # noqa: E402

CONTROL_PERIOD_S = 0.01


def _empty_arena_vehicle_scenario(**vehicle_overrides):
    path = next(path for path in SCENARIO_FILES if path.stem == "empty_arena")
    scenario = load_scenario(path)
    scenario["spawn"].update({"x": 0.0, "y": 0.0, "yaw_deg": 0.0})
    scenario["vehicle"].update(vehicle_overrides)
    return scenario


def test_respawn_matches_the_sampled_pose(make_env):
    path = SCENARIO_FILES[[p.stem for p in SCENARIO_FILES].index("light_seek")]
    env = make_env(path)
    _, info = env.reset(seed=3)
    spawn_x, spawn_y, spawn_yaw = info["world"]["spawn"]
    x, y, yaw = info["pose"]
    assert math.hypot(x - spawn_x, y - spawn_y) < 0.05
    assert abs(yaw - spawn_yaw) < 0.02


def test_actions_are_saturated_and_rate_limited(make_env):
    env = make_env(small_scenario())
    env.reset(seed=0)
    rates = env.scenario["vehicle"]
    previous = np.zeros(3)
    for _ in range(60):
        _, _, _, _, info = env.step([5.0, -9.0, float("nan")])   # nonsense on purpose
        applied = info["applied_action"]
        assert env.action_space.contains(applied)
        change = np.abs(applied - previous)
        assert change[spaces.THROTTLE] <= rates["throttle_rate_per_s"] * 0.01 + 1e-6
        assert change[spaces.STEERING] <= rates["steering_rate_per_s"] * 0.01 + 1e-6
        previous = applied.copy()
    # After 0.6 s the commands have arrived at the limits (NaN brake became 0).
    np.testing.assert_allclose(applied, [1.0, -1.0, 0.0], atol=1e-6)


def test_extra_info_exposes_finite_wheel_and_powertrain_data(make_env):
    env = make_env(small_scenario())
    env.reset(seed=0)
    for _ in range(50):
        _, _, _, _, info = env.step([1.0, 0.0, 0.0])
    extras = info["extras"]["vehicle"]["vehicle"]

    assert set(extras["wheels"]) == {"front_left", "front_right", "rear_left", "rear_right"}
    for wheel in extras["wheels"].values():
        assert set(wheel) == {"omega_radps", "slip", "normal_force_n"}
        assert all(math.isfinite(value) for value in wheel.values())
        assert wheel["omega_radps"] > 0.0        # rolling forwards
        assert wheel["normal_force_n"] > 0.0     # pressed onto the ground
    total_load = sum(wheel["normal_force_n"] for wheel in extras["wheels"].values())
    assert 5000.0 < total_load < 15000.0         # roughly the car's weight
    assert 0.0 <= extras["governor_scale"] <= 1.0
    assert math.isfinite(extras["engine_speed_radps"])
    assert isinstance(extras["gear"], int)


def test_positive_steering_turns_left_and_speed_is_governed(make_env):
    env = make_env(small_scenario(arena={"size_m": [60.0, 60.0], "walls": True}))
    env.reset(seed=0)
    for _ in range(400):
        observation, _, _, _, info = env.step([1.0, 0.5, 0.0])
    assert observation["imu"][spaces.IMU_YAW_RATE] > 0.1      # turning left
    assert info["pose"][1] > 0.5                              # ended up on the left (+y)
    top_speed = env.scenario["vehicle"]["max_speed_mps"]
    assert 0.8 * top_speed < observation["imu"][spaces.IMU_SPEED] < top_speed + 0.2


@pytest.mark.slow
def test_lower_steering_gain_reduces_yaw_rate(make_env):
    yaw_rates = []
    for gain in (1.0, 0.5):
        env = make_env(_empty_arena_vehicle_scenario(steering_gain=gain))
        observation, _ = env.reset(seed=0)
        for _ in range(250):
            observation, _, _, _, _ = env.step([0.2, 1.0, 0.0])
        yaw_rates.append(abs(float(observation["imu"][spaces.IMU_YAW_RATE])))

    assert yaw_rates[1] < yaw_rates[0] * 0.9


@pytest.mark.slow
def test_rigid_tires_change_braking_distance(make_env):
    distances = []
    for tire_model in ("tmeasy", "rigid"):
        scenario = _empty_arena_vehicle_scenario(tire_model=tire_model)
        scenario["arena"].update({"size_m": [200.0, 200.0], "walls": False})
        env = make_env(scenario)
        observation, _ = env.reset(seed=0)
        for _ in range(1000):
            observation, _, _, _, info = env.step([1.0, 0.0, 0.0])
            if observation["imu"][spaces.IMU_SPEED] >= 3.9:
                break
        assert observation["imu"][spaces.IMU_SPEED] >= 3.9
        start_x = float(info["pose"][0])
        for _ in range(500):
            observation, _, _, _, info = env.step([0.0, 0.0, 1.0])
            if observation["imu"][spaces.IMU_SPEED] <= 0.05:
                break
        assert observation["imu"][spaces.IMU_SPEED] <= 0.05
        distances.append(float(info["pose"][0]) - start_x)

    assert abs(distances[1] - distances[0]) > max(0.05, distances[0] * 0.02)


@pytest.mark.slow
def test_larger_throttle_exponent_slows_partial_throttle_acceleration(make_env):
    times = []
    for exponent in (1.0, 2.0):
        env = make_env(_empty_arena_vehicle_scenario(throttle_curve_exponent=exponent))
        observation, _ = env.reset(seed=0)
        for step in range(1500):
            observation, _, _, _, _ = env.step([0.5, 0.0, 0.0])
            if observation["imu"][spaces.IMU_SPEED] >= 3.0:
                times.append((step + 1) * CONTROL_PERIOD_S)
                break

    assert len(times) == 2, "both throttle settings must reach 3 m/s"
    assert times[1] > times[0] * 1.1


def test_vehicle_tuning_defaults_preserve_the_current_behaviour():
    """Check that the default values in config/vehicle.py reproduce the current behaviour.

    This is a regression test for the vehicle & terrain team. If you change the defaults,
    you must update this test to match the new behaviour.
    """
    scenario = load_scenario({})
    vehicle = scenario["vehicle"]
    assert vehicle["tire_model"] == "tmeasy"
    assert vehicle["steering_gain"] == 1.0
    assert vehicle["throttle_curve_exponent"] == 1.0


@pytest.mark.parametrize("gain", (1.0, 3.0))
def test_steering_gain_accepts_values_above_one(gain):
    scenario = load_scenario({"vehicle": {"steering_gain": gain}})

    assert scenario["vehicle"]["steering_gain"] == gain


def test_steering_gain_rejects_negative_values():
    with pytest.raises(ScenarioError, match=r"vehicle\.steering_gain must be non-negative"):
        load_scenario({"vehicle": {"steering_gain": -1.01}})


def test_scaled_steering_is_clamped_before_reaching_chrono(make_env):
    scenario = _empty_arena_vehicle_scenario(steering_gain=3.0, steering_rate_per_s=1000.0)
    env = make_env(scenario)
    env.reset(seed=0)

    env.vehicle.set_action([0.0, 1.0, 0.0], CONTROL_PERIOD_S)
    assert env.vehicle.inputs.m_steering == 1.0
    env.vehicle.set_action([0.0, -1.0, 0.0], CONTROL_PERIOD_S)
    assert env.vehicle.inputs.m_steering == -1.0
    env.vehicle.set_action([0.0, 0.2, 0.0], CONTROL_PERIOD_S)
    assert env.vehicle.inputs.m_steering == pytest.approx(0.6)


# ----------------------------------------------------------------------------
# Terrain
# ----------------------------------------------------------------------------

def test_terrain_defaults_reproduce_the_flat_rigid_ground():
    assert load_scenario({})["terrain"] == {"type": "rigid_flat", "friction": 0.9, "patches": []}


def test_unimplemented_terrain_type_is_rejected():
    with pytest.raises(ScenarioError, match=r"terrain\.type: scm is not implemented yet"):
        load_scenario({"terrain": {"type": "scm"}})


# ---- B1: friction patches -------------------------------------------------

ICE = {"x": 5.0, "y": 0.0, "size_m": [4.0, 4.0], "friction": 0.2}
MUD = {"x": -5.0, "y": 0.0, "size_m": [4.0, 4.0], "friction": 0.5}


def _patches(*patches):
    return {"type": "rigid_patches", "patches": [dict(p) for p in patches]}


def test_valid_friction_patches_are_accepted():
    touching = dict(ICE, x=9.0)                      # shares an edge with ICE: allowed
    terrain = load_scenario({"terrain": _patches(ICE, MUD, touching)})["terrain"]
    assert terrain["type"] == "rigid_patches"
    assert len(terrain["patches"]) == 3


@pytest.mark.parametrize(
    "terrain, message",
    [
        ({"patches": [ICE]}, r"terrain\.patches needs terrain\.type: rigid_patches"),
        (_patches(dict(ICE, z=1.0)), r"unknown setting 'z' in terrain\.patches\[0\]"),
        (_patches({"x": 5.0, "y": 0.0, "size_m": [4.0, 4.0]}), r"missing 'friction'"),
        (_patches(dict(ICE, x=[1.0, 2.0])), r"x must be a number"),
        (_patches(dict(ICE, size_m=[4.0, -1.0])), r"size_m must be two positive numbers"),
        (_patches(dict(ICE, size_m=[4.0])), r"size_m must be two positive numbers"),
        (_patches(dict(ICE, friction=0.0)), r"friction must be positive"),
        (_patches(dict(ICE, x=19.0)), r"reaches outside"),
        (_patches(ICE, dict(MUD, x=7.0, y=1.0)), r"terrain\.patches\[1\] overlaps terrain\.patches\[0\]"),
        ({"type": "rigid_patches", "patches": "ice"}, r"terrain\.patches must be a list"),
    ],
)
def test_bad_friction_patches_are_rejected(terrain, message):
    with pytest.raises(ScenarioError, match=message):
        load_scenario({"terrain": terrain})


def test_patch_colour_goes_from_icy_blue_to_dark_brown():
    from flycar.env import terrain

    assert terrain.patch_colour(0.05) == pytest.approx(terrain.ICE_COLOUR)
    assert terrain.patch_colour(2.0) == pytest.approx(terrain.GRIP_COLOUR)
    blues = [terrain.patch_colour(f)[2] for f in (0.1, 0.4, 0.8, 1.2)]
    assert blues == sorted(blues, reverse=True)       # less blue as the grip goes up


def test_rigid_flat_still_builds_one_ground_patch(make_env):
    env = make_env(small_scenario())
    env.reset(seed=0)
    assert env.terrain.friction_patches == []
    assert len(env.terrain.chrono_terrain.GetPatches()) == 1


def test_friction_patches_set_the_friction_under_them_and_stay_level(make_env):
    import pychrono as chrono

    env = make_env(small_scenario(terrain=_patches(ICE, MUD)))
    env.reset(seed=0)
    ground = env.terrain.chrono_terrain
    assert len(ground.GetPatches()) == 3              # two patches + the ground

    def at(x, y):
        return chrono.ChVector3d(x, y, 0.0)

    for point, friction in [(at(5.0, 0.0), 0.2), (at(6.9, 1.9), 0.2),    # on the ice
                            (at(-5.0, 0.0), 0.5),                         # on the mud
                            (at(0.0, 0.0), 0.9), (at(7.5, 0.0), 0.9)]:    # plain ground
        assert ground.GetCoefficientFriction(point) == pytest.approx(friction, abs=1e-6)
        assert ground.GetHeight(point) == pytest.approx(0.0, abs=1e-9)    # no step at the patches


# Driving across ice: a patch from x = 4 m to x = 24 m, right in the car's path.
ICE_STRIP = {"x": 14.0, "y": 0.0, "size_m": [20.0, 10.0], "friction": 0.1}
WHEEL_MARGIN_M = 1.5    # the axles are about 0.97 m in front of/behind the chassis origin


def _mean_wheel_slip_over_strip(make_env, patches):
    """Full throttle along +x; mean |slip| of the 4 wheels while all are over the strip's x range."""
    env = make_env(small_scenario(
        arena={"size_m": [60.0, 20.0], "walls": True},
        episode={"max_time_s": 20.0},
        vehicle={"max_speed_mps": 10.0},       # above the Gator's own top speed: the governor stays out
        terrain={"type": "rigid_patches", "patches": patches},
    ))
    env.reset(seed=0)
    start = ICE_STRIP["x"] - ICE_STRIP["size_m"][0] / 2.0 + WHEEL_MARGIN_M
    end = ICE_STRIP["x"] + ICE_STRIP["size_m"][0] / 2.0 - WHEEL_MARGIN_M
    slips = []
    for _ in range(1500):
        _, _, terminated, truncated, info = env.step([1.0, 0.0, 0.0])
        x = float(info["pose"][0])
        if x > end:
            break
        if x >= start:
            wheels = info["extras"]["vehicle"]["vehicle"]["wheels"].values()
            slips.append(sum(abs(wheel["slip"]) for wheel in wheels) / 4.0)
        assert not (terminated or truncated), "the car did not get across the strip"
    assert slips, "the car never reached the strip"
    return sum(slips) / len(slips)


@pytest.mark.slow
def test_wheels_slip_more_on_a_low_friction_patch(make_env):
    """B1 'done when': full throttle across ice gives more wheel slip than the plain ground."""
    on_ground = _mean_wheel_slip_over_strip(make_env, [])
    on_ice = _mean_wheel_slip_over_strip(make_env, [ICE_STRIP])
    print(f"\n  mean |slip| over x = 5.5..22.5 m: ground {on_ground:.4f}, ice (friction 0.1) {on_ice:.4f}")
    assert on_ice > 1.5 * on_ground


def test_terrain_is_built_from_its_config_and_is_flat_at_z0(make_env):
    import pychrono as chrono

    env = make_env(small_scenario(terrain={"friction": 0.6}))
    env.reset(seed=0)
    centre = chrono.ChVector3d(0.0, 0.0, 0.0)
    assert env.terrain.chrono_terrain.GetHeight(centre) == pytest.approx(0.0)
    assert env.terrain.chrono_terrain.GetCoefficientFriction(centre) == pytest.approx(0.6)


# ============================================================================
# A1: handling test suite
#
# Measures, through the public interface only (env.reset(), env.step(),
# info["pose"] and obs["imu"]):
#
#     top_speed     speed after a long full-throttle run          [m/s]
#     t_0_to_3      time from rest to 3 m/s at full throttle        [s]
#     brake_dist    stopping distance from top speed, full brake    [m]
#     turn_radius   circle fitted to the pose trace, full steering  [m]
#     t90_steer     steering step -> 90 % of the final yaw rate     [s]
#
# Each value is checked against BASELINE ("today's numbers") within a
# tolerance band, so the suite fails if someone changes the car by accident.
# With -s a summary table is printed after the tests in this file.
#
# Scenario: scenarios/empty_arena.yaml with four changes, so the numbers are
# repeatable and the car never reaches a wall:
#     * spawn fixed at the origin facing +x (the file randomizes it),
#     * arena 200 m x 200 m instead of 40 m x 40 m,
#     * episode long enough for the longest measurement,
#     * vehicle.max_speed_mps set by each test.
#
# Baseline recorded on 2026-10-15 (Windows) with the default car
# (max_speed_mps = 4.0). t90_steer is mostly set by the steering rate limit
# (steering_rate_per_s = 4.0 per second, so a 0.5 step takes 0.125 s).
# ============================================================================

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

EMPTY_ARENA = REPO_ROOT / "scenarios" / "empty_arena.yaml"

# One env.step() advances this much simulated time [s]. It is written into the
# scenario below (timing.control_period_s), so the tests never need to read it
# back from the environment. 0.01 s is the project default.
DT = 0.01

# Today's numbers: name -> (value, relative tolerance, unit).
BASELINE = {
    "top_speed": (3.974, 0.05, "m/s"),
    "t_0_to_3": (0.520, 0.10, "s"),
    "brake_dist": (2.081, 0.10, "m"),
    "turn_radius": (2.806, 0.05, "m"),
    "t90_steer": (0.140, 0.15, "s"),
}

# Every measurement is added here, and printed as one table at the end.
RESULTS = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def scenario(max_speed_mps=4.0, max_time_s=60.0):
    """Load scenarios/empty_arena.yaml and apply the four changes listed at the top."""
    with open(EMPTY_ARENA, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw["timing"] = {"control_period_s": DT}
    raw["episode"] = {"max_time_s": max_time_s}
    raw["arena"] = {"size_m": [200.0, 200.0], "walls": True}
    raw["spawn"] = {"x": 0.0, "y": 0.0, "yaw_deg": 0.0}
    raw["vehicle"] = {"max_speed_mps": max_speed_mps}
    return raw


def act(throttle=0.0, steering=0.0, brake=0.0):
    """Build one action [throttle, steering, brake]."""
    a = np.zeros(3, dtype=np.float32)
    a[THROTTLE], a[STEERING], a[BRAKE] = throttle, steering, brake
    return a


def speed(obs):
    """Forward speed from the IMU [m/s]."""
    return float(obs["imu"][IMU_SPEED])


def yaw_rate(obs):
    """Turning rate from the IMU [rad/s], positive = left."""
    return float(obs["imu"][IMU_YAW_RATE])


def steps(seconds):
    """How many env.step() calls make `seconds` of simulated time."""
    return int(round(seconds / DT))


def run(env, action, seconds):
    """Hold one action for `seconds`, recording speed, yaw rate and position.

    Returns (obs, info, speeds, yaw_rates, xy) where the last three are arrays
    with one entry per step.
    """
    speeds, rates, xys = [], [], []
    obs = info = None
    for _ in range(steps(seconds)):
        obs, _, terminated, truncated, info = env.step(action)
        speeds.append(speed(obs))
        rates.append(yaw_rate(obs))
        xys.append(np.array(info["pose"][:2]))
        if terminated or truncated:
            break
    return obs, info, np.array(speeds), np.array(rates), np.array(xys)


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------

def measure_top_speed(env, seconds=15.0):
    """Full throttle for `seconds`, then the average speed over the last second."""
    env.reset(seed=0)
    _, _, speeds, _, _ = run(env, act(throttle=1.0), seconds)
    return float(speeds[-steps(1.0):].mean())


def measure_0_to_3(env, target=3.0, timeout=10.0):
    """Time from rest until the speed first reaches `target` at full throttle."""
    env.reset(seed=0)
    for k in range(1, steps(timeout) + 1):
        obs, *_ = env.step(act(throttle=1.0))
        if speed(obs) >= target:
            return k * DT
    return float("nan")


def measure_braking_distance(env):
    """Reach top speed, then full brake: distance travelled until stopped."""
    env.reset(seed=0)
    _, info, _, _, _ = run(env, act(throttle=1.0), 15.0)
    start = np.array(info["pose"][:2])
    stop_xy = start
    for _ in range(steps(10.0)):
        obs, _, _, _, info = env.step(act(brake=1.0))
        stop_xy = np.array(info["pose"][:2])
        if abs(speed(obs)) < 0.02:
            break
    return float(np.linalg.norm(stop_xy - start))


def fit_circle_radius(xy):
    """Least-squares circle through points: x^2 + y^2 = 2ax + 2by + c.

    (a, b) is the centre and the radius is sqrt(c + a^2 + b^2).
    """
    x, y = xy[:, 0], xy[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    (a, b, c), *_ = np.linalg.lstsq(A, x**2 + y**2, rcond=None)
    return math.sqrt(c + a * a + b * b)


def measure_turn_radius(env):
    """Full throttle + full left steering at low speed; fit a circle to the path.

    The scenario's max_speed_mps (1.5) keeps the car slow.
    """
    env.reset(seed=0)
    run(env, act(throttle=1.0, steering=1.0), 4.0)                  # let the turn settle
    _, _, _, _, xy = run(env, act(throttle=1.0, steering=1.0), 8.0)
    return fit_circle_radius(xy)


def measure_t90_steer(env, steer=0.5):
    """Drive straight, step the steering to `steer`, time to 90 % of final yaw rate."""
    env.reset(seed=0)
    run(env, act(throttle=1.0), 6.0)                                 # steady straight run
    _, _, _, rates, _ = run(env, act(throttle=1.0, steering=steer), 6.0)
    final = rates[-steps(1.0):].mean()
    assert abs(final) > 0.05, "car did not turn; cannot measure step response"
    reached = np.abs(rates) >= 0.9 * abs(final)
    return (int(np.argmax(reached)) + 1) * DT


# ---------------------------------------------------------------------------
# Checking and reporting
# ---------------------------------------------------------------------------

def record(label, value, unit, baseline="-", status="info"):
    """Add one row to the summary table."""
    RESULTS.append((label, value, unit, baseline, status))


def check(name, value):
    """Compare a measurement with BASELINE[name] and fail if it is outside the band."""
    expected, tol, unit = BASELINE[name]
    inside = math.isfinite(value) and value == pytest.approx(expected, rel=tol)
    record(name, value, unit, f"{expected:g} +-{tol:.0%}", "ok" if inside else "FAIL")
    assert math.isfinite(value), f"{name} is not finite"
    assert inside, f"{name} = {value:.3f} {unit}, expected {expected} +-{tol:.0%}"


@pytest.fixture(scope="module", autouse=True)
def summary_table():
    """After all tests in this file have run, print the results as one table."""
    RESULTS.clear()
    yield
    if not RESULTS:
        return
    print("\n\nVehicle handling (A1)")
    print(f"{'measurement':<24}{'value':>10}  {'unit':<5}{'baseline':<16}{'result'}")
    print("-" * 64)
    for label, value, unit, baseline, status in RESULTS:
        print(f"{label:<24}{value:>10.3f}  {unit:<5}{baseline:<16}{status}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_top_speed(make_env):
    """Full throttle settles near vehicle.max_speed_mps, and matches today's number."""
    value = measure_top_speed(make_env(scenario(max_speed_mps=4.0)))
    assert value == pytest.approx(4.0, rel=0.10), "top speed is not near max_speed_mps"
    check("top_speed", value)


@pytest.mark.slow
def test_top_speed_follows_max_speed_setting(make_env):
    """'Done when': changing max_speed_mps moves the top-speed measurement."""
    slow = measure_top_speed(make_env(scenario(max_speed_mps=2.0)), seconds=12.0)
    fast = measure_top_speed(make_env(scenario(max_speed_mps=5.0)), seconds=15.0)
    slow_ok = slow == pytest.approx(2.0, rel=0.10)
    fast_ok = fast == pytest.approx(5.0, rel=0.10)
    record("top_speed @ max 2.0", slow, "m/s", "2 +-10%", "ok" if slow_ok else "FAIL")
    record("top_speed @ max 5.0", fast, "m/s", "5 +-10%", "ok" if fast_ok else "FAIL")
    assert slow_ok, f"max_speed_mps = 2.0 gave top speed {slow:.3f} m/s"
    assert fast_ok, f"max_speed_mps = 5.0 gave top speed {fast:.3f} m/s"
    assert fast > slow + 2.0


def test_zero_to_three(make_env):
    check("t_0_to_3", measure_0_to_3(make_env(scenario())))


@pytest.mark.slow
def test_braking_distance(make_env):
    check("brake_dist", measure_braking_distance(make_env(scenario())))


@pytest.mark.slow
def test_turning_radius(make_env):
    check("turn_radius", measure_turn_radius(make_env(scenario(max_speed_mps=1.5))))


@pytest.mark.slow
def test_steering_step_response(make_env):
    check("t90_steer", measure_t90_steer(make_env(scenario(max_speed_mps=2.0))))


def test_circle_fit_recovers_known_radius():
    """The circle fit itself is right (no simulation needed)."""
    t = np.linspace(0.3, 5.0, 200)
    xy = np.column_stack([2.0 + 3.0 * np.cos(t), -1.0 + 3.0 * np.sin(t)])
    assert fit_circle_radius(xy) == pytest.approx(3.0, rel=1e-6)


if __name__ == "__main__":
    # Running this file directly (e.g. the VS Code play button) runs the
    # tests through pytest, the same as `python -m pytest tests/test_vehicle.py -s`.
    sys.exit(pytest.main([__file__, "-s"]))

