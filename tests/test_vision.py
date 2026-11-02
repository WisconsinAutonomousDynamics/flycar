"""Vision team: the compound eyes."""

import math
import os

import numpy as np
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


# ---- Backend parity (A2) ---------------------------------------------------

# The sensor settings that make the GPU eyes match the raycast eyes. How they
# were measured, and the numbers, are in docs/teams/vision.md.
SENSOR_PARITY = {
    "sensor_linearize": True,
    "sensor_light_falloff": True,
    "sensor_light_range": 5.0,
    "sensor_lamp_glow": True,
}


def test_sensor_calibration_settings_default_to_todays_images():
    eyes = load_scenario({})["eyes"]
    assert eyes["sensor_linearize"] is False
    assert eyes["sensor_light_falloff"] is False
    assert eyes["sensor_lamp_glow"] is False
    assert eyes["sensor_brightness_scale"] == 1.0
    assert eyes["sensor_light_range"] == 10.0
    load_scenario({"eyes": SENSOR_PARITY})   # the calibrated values are accepted


@pytest.mark.parametrize("settings, key", [
    ({"sensor_brightness_scale": 0.0}, "sensor_brightness_scale"),
    ({"sensor_brightness_scale": -1.0}, "sensor_brightness_scale"),
    ({"sensor_light_range": "far"}, "sensor_light_range"),
    ({"sensor_linearize": "yes"}, "sensor_linearize"),
    ({"sensor_light_falloff": 1}, "sensor_light_falloff"),
    ({"sensor_lamp_glow": None}, "sensor_lamp_glow"),
])
def test_sensor_calibration_settings_are_checked(settings, key):
    with pytest.raises(ScenarioError, match=key):
        load_scenario({"eyes": settings})


def test_srgb_to_linear_matches_the_standard():
    from flycar.env.sensing.eyes import srgb_to_linear

    values = srgb_to_linear(np.array([0.0, 0.04045, 0.5, 1.0]))
    np.testing.assert_allclose(values, [0.0, 0.04045 / 12.92, 0.21404, 1.0], atol=1e-5)


def test_lamp_glow_peaks_at_the_lamp_and_vanishes_when_hidden():
    from flycar.env.sensing.eyes import lamp_glow, pinhole_ray_directions

    directions = pinhole_ray_directions(20, 20, 120.0, 60.0, 0.0)
    lamp = {"x": 10.0, "y": 0.0, "z": 0.0, "intensity": 1.0}
    pixel_angle = math.radians(120.0) / 20

    glow = lamp_glow(np.zeros(3), directions, [lamp], pixel_angle, lambda origin, lamp: False)
    row, col = np.unravel_index(np.argmax(glow), (20, 20))
    assert row in (9, 10) and col in (9, 10)                 # straight ahead: the centre pixels
    assert 0.0 <= glow.min() and glow.max() <= 1.0

    hidden = lamp_glow(np.zeros(3), directions, [lamp], pixel_angle, lambda origin, lamp: True)
    assert not hidden.any()


def test_sensor_eyes_ask_chrono_for_dimming_lights_only_when_enabled():
    from flycar.env.sensing.eyes import SensorEyes

    eyes = SensorEyes.__new__(SensorEyes)   # skip __init__: it needs a GPU
    assert eyes._light_flags() == ()          # Chrono's default: constant brightness
    eyes.light_falloff = True
    assert eyes._light_flags() == (False,)    # const_color=False


def test_sensor_eyes_render_applies_the_calibration_to_the_camera_image():
    from flycar.env.sensing.eyes import SensorEyes, srgb_to_linear

    class FakeBuffer:
        def HasData(self):
            return True

        def GetRGBA8Data(self):
            return np.full((4, 6, 4), 128, dtype=np.uint8)   # 4 x 6 camera pixels, mid grey

    class FakeCamera:
        def GetMostRecentRGBA8Buffer(self):
            return FakeBuffer()

    def render(**settings):
        eyes = SensorEyes.__new__(SensorEyes)   # skip __init__: it needs a GPU
        eyes.cameras = {"left": FakeCamera(), "right": FakeCamera()}
        eyes._raw = {"left": None, "right": None}
        eyes.rows, eyes.cols, eyes.block_h, eyes.block_w = 2, 3, 2, 2
        eyes.linearize, eyes.brightness_scale, eyes.glow = False, 1.0, False
        for name, value in settings.items():
            setattr(eyes, name, value)
        left, _ = eyes.render()
        assert left.shape == (2, 3) and left.dtype == np.float32
        return left

    grey = 128 / 255   # the Rec. 709 weights add up to 1
    np.testing.assert_allclose(render(), grey, atol=1e-6)
    np.testing.assert_allclose(render(linearize=True), srgb_to_linear(grey), atol=1e-6)
    np.testing.assert_allclose(render(brightness_scale=0.5), 0.5 * grey, atol=1e-6)


PARITY_HIDER = {"shape": "box", "size_m": [1.0, 6.0, 4.0], "x": 5.0, "y": 0.0, "clearance_m": 0.0}
PARITY_SCENES = {
    "lamp_ahead": ([{"x": 8.0, "y": 0.0, "target": True}], []),
    "lamp_left": ([{"x": 5.0, "y": 6.0, "target": True}], []),
    "lamp_right": ([{"x": 5.0, "y": -6.0, "target": True}], []),
    "lamp_close": ([{"x": 4.0, "y": 1.0, "target": True}], []),
    "lamp_hidden": ([{"x": 9.0, "y": 0.0, "z": 1.0, "target": True}], [PARITY_HIDER]),
    "no_lamp": ([], []),
}

# Tolerances for the calibrated sensor eyes (see docs/teams/vision.md).
PARITY_MEAN_ERROR = 0.08      # mean |sensor - raycast| over both eyes (worst measured: 0.058)
PARITY_STEERING_ERROR = 0.1   # Braitenberg steering command, sensor vs raycast (worst measured: 0.00)


def _braitenberg_steering(left, right):
    from flycar.drivers.braitenberg import BraitenbergDriver

    return float(BraitenbergDriver().act({"eye_left": left, "eye_right": right})[1])


def _rendered_eyes(make_env, backend, lights, obstacles):
    scenario = small_scenario(lights=lights, obstacles=obstacles,
                              lighting={"ambient": 0.08, "sky": 0.04}, eyes=dict(SENSOR_PARITY))
    env = make_env(scenario, eye_backend=backend)
    env.reset(seed=0)
    for _ in range(10):                      # brake on, so the cameras have fresh frames
        _, _, _, _, info = env.step([0.0, 0.0, 1.0])
    assert info["eye_backend"] == backend
    return env.eyes.render()


@pytest.mark.parametrize("scene", sorted(PARITY_SCENES))
def test_calibrated_sensor_eyes_match_the_raycast_eyes(make_env, scene):
    """GPU only: skipped in CI (FLYCAR_EYE_BACKEND=raycast) and on machines without OptiX."""
    from flycar.env.sensing import eyes

    if os.environ.get(eyes.BACKEND_ENV_VAR, "").strip().lower() == "raycast":
        pytest.skip(f"{eyes.BACKEND_ENV_VAR} forces the raycast backend")
    works, reason = eyes.sensor_backend_works()
    if not works:
        pytest.skip(f"sensor backend unavailable: {reason}")

    lights, obstacles = PARITY_SCENES[scene]
    ray_left, ray_right = _rendered_eyes(make_env, "raycast", lights, obstacles)
    gpu_left, gpu_right = _rendered_eyes(make_env, "sensor", lights, obstacles)

    # Overall brightness agrees.
    mean_error = np.abs(np.stack([gpu_left, gpu_right]) - np.stack([ray_left, ray_right])).mean()
    assert mean_error <= PARITY_MEAN_ERROR

    # A driver tuned on one backend steers the same way on the other.
    ray_steering = _braitenberg_steering(ray_left, ray_right)
    gpu_steering = _braitenberg_steering(gpu_left, gpu_right)
    assert abs(gpu_steering - ray_steering) <= PARITY_STEERING_ERROR

    # A visible lamp appears in the same place (within one pixel).
    for ray_image, gpu_image in ((ray_left, gpu_left), (ray_right, gpu_right)):
        if ray_image.max() >= 0.5:
            ray_spot = np.array(np.unravel_index(np.argmax(ray_image), ray_image.shape))
            gpu_spot = np.array(np.unravel_index(np.argmax(gpu_image), gpu_image.shape))
            assert np.abs(ray_spot - gpu_spot).max() <= 1
