"""Vision team: the compound eyes."""

import math

import numpy as np
import pytest

from conftest import small_scenario
from flycar.env.scenario import ScenarioError, load_scenario


def test_eye_benchmark_excludes_warmup_and_closes_each_seed(monkeypatch):
    from scripts import benchmark_eyes

    created = []

    class FakeEnv:
        def __init__(self, scenario, eye_backend):
            assert scenario.name == "light_seek.yaml"
            assert eye_backend == "raycast"
            self.eyes = self
            self.rows = self.cols = 20
            self.renders = 0
            self.closed = False
            created.append(self)

        def reset(self, seed):
            self.seed = seed
            self.render()

        def render(self):
            self.renders += 1

        def extra_info(self):
            return {"render_ms": float(self.renders), "primary_rays": 800}

        def close(self):
            self.closed = True

    monkeypatch.setattr(benchmark_eyes, "FlyCarEnv", FakeEnv)
    report = benchmark_eyes.main()["eyes_benchmark"]
    assert report["warmup_renders"] == 20
    assert report["measured_renders"] == 200
    assert [env.seed for env in created] == [0, 1, 2]
    assert all(env.closed and env.renders == 221 for env in created)
    for result in report["results"]:
        assert result["metrics"]["render_ms"]["median"] == pytest.approx(121.5)
        assert result["metrics"]["render_ms"]["p95"] == pytest.approx(211.05)
        assert result["metrics"]["wall_ms"]["median"] >= 0.0
        assert result["metrics"]["primary_rays"] == 800


def test_eye_benchmark_closes_environment_on_failure(monkeypatch):
    from scripts import benchmark_eyes

    class FakeEnv:
        rows = cols = 20
        closed = False

        def reset(self, seed):
            pass

        def render(self):
            raise RuntimeError("render failed")

        def close(self):
            self.closed = True

    env = FakeEnv()
    env.eyes = env
    monkeypatch.setattr(benchmark_eyes, "FlyCarEnv", lambda *args, **kwargs: env)
    with pytest.raises(RuntimeError, match="render failed"):
        benchmark_eyes.main()
    assert env.closed


@pytest.mark.parametrize("shadows", [False, True])
def test_raycast_render_metrics(make_env, monkeypatch, shadows):
    env = make_env(small_scenario(
        eyes={"resolution": [20, 20], "shadows": shadows},
        lights=[{"x": 5.0, "y": 6.0, "z": 1.5, "target": True}],
    ))
    observation, info = env.reset(seed=0)
    reported = info["extras"]["vision"]["eyes"]
    assert reported == env.eyes.extra_info()

    secondary_calls = []
    ray_hits_something = env.eyes._ray_hits_something

    def count_secondary(start, end):
        secondary_calls.append(1)
        return ray_hits_something(start, end)

    monkeypatch.setattr(env.eyes, "_ray_hits_something", count_secondary)
    left, right = env.eyes.render()
    metrics = env.eyes.extra_info()

    for key in ("render_ms", "primary_ms", "shadow_ms", "lamp_glow_ms"):
        assert math.isfinite(metrics[key]) and metrics[key] >= 0.0
    assert metrics["render_ms"] >= sum(
        metrics[key] for key in ("primary_ms", "shadow_ms", "lamp_glow_ms")
    )
    assert metrics["primary_rays"] == 800
    assert metrics["lamp_glow_rays"] == 2
    assert metrics["shadow_rays"] == len(secondary_calls) - 2
    assert (metrics["shadow_rays"] > 0) == shadows
    assert metrics["rays_cast"] == 800 + len(secondary_calls)
    np.testing.assert_array_equal(left, observation["eye_left"])
    np.testing.assert_array_equal(right, observation["eye_right"])
    assert reported["primary_rays"] == 800
    metrics["render_ms"] = -1.0
    assert env.eyes.extra_info()["render_ms"] >= 0.0


def test_raycast_looks_up_collision_system_for_each_ray_batch(make_env, monkeypatch):
    from types import SimpleNamespace

    env = make_env(small_scenario(lights=[{"x": 5.0, "y": 6.0, "target": True}]))
    observation, _ = env.reset(seed=0)
    collision_system = env.system.GetCollisionSystem()
    lookups = []
    queries = []

    def ray_hit(start, end, result):
        queries.append(1)
        return collision_system.RayHit(start, end, result)

    def get_collision_system():
        lookups.append(1)
        return SimpleNamespace(RayHit=ray_hit)

    monkeypatch.setattr(env.eyes, "system", SimpleNamespace(GetCollisionSystem=get_collision_system))
    for _ in range(2):
        before_queries = len(queries)
        before_lookups = len(lookups)
        left, right = env.eyes.render()
        metrics = env.eyes.extra_info()
        assert len(lookups) - before_lookups == 2 + metrics["shadow_rays"] + metrics["lamp_glow_rays"]
        assert len(queries) - before_queries == metrics["rays_cast"]
        np.testing.assert_array_equal(left, observation["eye_left"])
        np.testing.assert_array_equal(right, observation["eye_right"])


def test_shadow_rays_reuse_lamp_coordinate_lists(make_env, monkeypatch):
    env = make_env(small_scenario(lights=[{"x": 5.0, "y": 6.0, "target": True}]))
    observation, _ = env.reset(seed=0)
    endpoints = []
    ray_hits_something = env.eyes._ray_hits_something

    def record_endpoint(start, end):
        endpoints.append(end)
        return ray_hits_something(start, end)

    monkeypatch.setattr(env.eyes, "_ray_hits_something", record_endpoint)
    left, right = env.eyes.render()
    assert env.eyes.extra_info()["shadow_rays"] > 4
    assert len({id(endpoint) for endpoint in endpoints}) <= 4
    np.testing.assert_array_equal(left, observation["eye_left"])
    np.testing.assert_array_equal(right, observation["eye_right"])


def test_primary_ray_distances_and_normals_match_individual_queries(make_env):
    from flycar.env.sensing.eyes import chrono

    env = make_env(small_scenario(lights=[]))
    env.reset(seed=0)
    origin = np.array([3.0, 0.0, 1.0])
    directions = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    expected_distance = np.full(3, np.inf)
    expected_normal = np.zeros((3, 3))
    start = chrono.ChVector3d(*origin.tolist())
    result = chrono.ChRayhitResult()
    collision_system = env.system.GetCollisionSystem()
    for index, direction in enumerate(directions):
        end = chrono.ChVector3d(*(origin + direction * env.eyes.max_range).tolist())
        if collision_system.RayHit(start, end, result):
            expected_distance[index] = result.dist_factor * env.eyes.max_range
            normal = result.abs_hitNormal
            expected_normal[index] = (normal.x, normal.y, normal.z)

    assert np.isfinite(expected_distance[:2]).all()
    assert np.isinf(expected_distance[2])
    assert not np.array_equal(expected_normal[0], expected_normal[1])
    distance, normal = env.eyes._cast_rays(origin, directions)
    np.testing.assert_array_equal(distance, expected_distance)
    np.testing.assert_array_equal(normal, expected_normal)
    empty_distance, empty_normal = env.eyes._cast_rays(origin, np.empty((0, 3)))
    assert empty_distance.shape == (0,)
    assert empty_normal.shape == (0, 3)


def test_raycast_sky_without_surface_hits(make_env, monkeypatch):
    env = make_env(small_scenario(lights=[]))
    env.reset(seed=0)

    def miss_all_rays(origin, directions):
        return np.full(len(directions), np.inf), np.zeros((len(directions), 3))

    monkeypatch.setattr(env.eyes, "_cast_rays", miss_all_rays)
    for image in env.eyes.render():
        np.testing.assert_array_equal(
            image, np.full((20, 20), env.eyes.sky, dtype=np.float32)
        )


def test_a_lamp_on_the_left_is_brighter_in_the_left_eye(make_env):
    lamp_left = small_scenario(lights=[{"x": 5.0, "y": 6.0, "target": True}],
                               lighting={"ambient": 0.05, "sky": 0.02})
    env = make_env(lamp_left)
    observation, info = env.reset(seed=0)
    assert info["target_bearing"] > 0.5                       # the lamp is to the left
    assert observation["eye_left"].max() > observation["eye_right"].max() + 0.1


def test_an_obstacle_hides_the_lamp(make_env):
    lights = [{"x": 9.0, "y": 0.0, "z": 1.0, "target": True}]
    lighting = {"ambient": 0.05, "sky": 0.02}
    visible = make_env(small_scenario(lights=lights, lighting=lighting))
    bright, _ = visible.reset(seed=0)

    # The same scene with a tall, wide box between the car and the lamp.
    box = [{"shape": "box", "size_m": [1.0, 6.0, 4.0], "x": 5.0, "y": 0.0, "clearance_m": 0.0}]
    hidden = make_env(small_scenario(lights=lights, lighting=lighting, obstacles=box))
    dark, _ = hidden.reset(seed=0)

    assert bright["eye_left"].max() > 0.3
    assert dark["eye_left"].max() < 0.5 * bright["eye_left"].max()


def test_raycast_eyes_follow_the_lights_the_world_reports(make_env):
    """Eyes ask World.current_lights() on every picture, so a moved lamp is seen moved."""
    scenario = small_scenario(lights=[{"x": 5.0, "y": 6.0, "target": True}],
                              lighting={"ambient": 0.05, "sky": 0.02})
    env = make_env(scenario)
    env.reset(seed=0)
    left, right = env.eyes.render()
    assert left.max() > right.max() + 0.1                   # lamp on the left

    # Pretend World moved the lamp to the right-hand side.
    moved = [dict(light, y=-6.0) for light in env.world.layout["lights"]]
    env.world.current_lights = lambda: moved
    left, right = env.eyes.render()
    assert right.max() > left.max() + 0.1


def test_sensor_eyes_move_their_scene_lights_when_the_world_moves_a_lamp():
    """SensorEyes.update() logic, without a GPU: a stand-in sensor manager records the calls."""
    from flycar.env.sensing.eyes import SensorEyes

    class FakeScene:
        def __init__(self):
            self.modified = []

        def ModifyPointLight(self, light_id, position, color, max_range):
            self.modified.append((light_id, position.x, position.y))

    class FakeManager:
        def __init__(self):
            self.scene = FakeScene()
            self.updates = 0

        def Update(self):
            self.updates += 1

    class FakeWorld:
        lights = [{"x": 1.0, "y": 2.0, "z": 1.5, "intensity": 1.0, "falloff_m": 6.0, "target": True}]

        def current_lights(self):
            return [dict(light) for light in self.lights]

    eyes = SensorEyes.__new__(SensorEyes)   # skip __init__: it needs a GPU
    eyes.world = FakeWorld()
    eyes.manager = FakeManager()
    eyes.light_ids = [7]
    eyes.light_positions = [(1.0, 2.0, 1.5)]

    eyes.update()
    assert eyes.manager.scene.modified == []                 # nothing moved
    eyes.world.lights[0]["y"] = -3.0
    eyes.update()
    assert eyes.manager.scene.modified == [(7, 1.0, -3.0)]
    assert eyes.manager.updates == 2


@pytest.mark.parametrize("settings, key", [
    ({"eyes": {"noise_std": 0.1}}, "eyes.noise_std"),
    ({"eyes": {"motion_blur": True}}, "eyes.motion_blur"),
])
def test_eye_noise_and_blur_are_not_implemented_yet(settings, key):
    eyes = load_scenario({})["eyes"]
    assert eyes["noise_std"] == 0.0 and eyes["motion_blur"] is False
    with pytest.raises(ScenarioError, match=key.replace(".", r"\.") + ".*not implemented yet"):
        load_scenario(settings)


def test_eye_update_period_must_be_a_multiple_of_the_control_period():
    with pytest.raises(ScenarioError, match="update_period_s"):
        load_scenario({"eyes": {"update_period_s": 0.015}})


def test_flycar_eye_backend_variable_overrides_the_requested_backend(monkeypatch):
    from flycar.env.sensing import eyes

    monkeypatch.setenv("FLYCAR_EYE_BACKEND", "raycast")
    # "sensor" would raise on a machine without a GPU; the variable wins.
    assert eyes.choose_backend("sensor") == ("raycast", "set by FLYCAR_EYE_BACKEND")

    monkeypatch.setenv("FLYCAR_EYE_BACKEND", "lidar")
    with pytest.raises(ValueError, match="FLYCAR_EYE_BACKEND"):
        eyes.choose_backend("auto")

    monkeypatch.delenv("FLYCAR_EYE_BACKEND")
    assert eyes.choose_backend("raycast") == ("raycast", "requested by the scenario")
