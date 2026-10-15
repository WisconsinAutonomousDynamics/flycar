"""Vision team: the compound eyes."""

import pytest

from conftest import small_scenario
from flycar.env.scenario import ScenarioError, load_scenario


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
