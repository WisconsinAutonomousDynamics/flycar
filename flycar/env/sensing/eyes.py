"""Compound-eye vision.

The car has two eyes on the front of the chassis, one turned to the left and
one to the right. Each eye produces a small picture (20 x 20 by default) of
luminance values between 0 (black) and 1 (saturated).

Two backends produce that picture. They share the same camera model and the
same interface, so the rest of the code does not care which one is running.

``RaycastEyes``  (CPU, works on any laptop)
    Shoots one ray per pixel using Chrono's collision ray query, then shades
    the point each ray hits: how much light reaches it, and is it in shadow?

``SensorEyes``  (needs an NVIDIA GPU with OptiX)
    One ``ChCameraSensor`` per eye from the Chrono sensor module. The camera
    image is averaged down to the compound-eye grid.

Camera model (both backends): a pinhole camera. Row 0 is the top of the view,
column 0 its left edge. The horizontal and vertical fields of view come from
the scenario (``eyes.fov_h_deg`` and ``eyes.fov_v_deg``).
"""

import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pychrono as chrono

from flycar.env.world import LAMP_RADIUS_M

# How much of the light falling on a surface is reflected towards the eye.
# Ground, walls, obstacles and the car's own body all use the same value, so
# an unlit wall is exactly as dark as unlit ground: only lamps create contrast.
ALBEDO = 0.5

# Shadow rays start this far off the surface, so that a surface does not
# shadow itself because of rounding errors [m].
SHADOW_OFFSET_M = 0.02


# ----------------------------------------------------------------------------
# Choosing a backend
# ----------------------------------------------------------------------------

_sensor_probe_result = None   # cached, because the probe takes about a second

# If this environment variable is set (auto, sensor or raycast), it overrides
# the backend asked for by the scenario or by run.py --eyes. CI sets it to
# "raycast" so every run is reproducible on any machine.
BACKEND_ENV_VAR = "FLYCAR_EYE_BACKEND"
BACKENDS = ("auto", "sensor", "raycast")


def sensor_backend_works():
    """Return (works, reason). Tests the sensor module in a separate process.

    See sensor_probe.py for why this cannot be a simple ``import``.
    """
    global _sensor_probe_result
    if _sensor_probe_result is not None:
        return _sensor_probe_result

    probe_script = Path(__file__).with_name("sensor_probe.py")
    try:
        finished = subprocess.run(
            [sys.executable, str(probe_script)],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        _sensor_probe_result = (False, "the sensor probe timed out")
        return _sensor_probe_result

    if finished.returncode == 0:
        _sensor_probe_result = (True, "the sensor probe rendered an image")
    else:
        # Keep the start of the last line of output: it usually names the
        # real problem. (Run sensor_probe.py by hand for the full message.)
        output = (finished.stdout + finished.stderr).strip().splitlines()
        last_line = output[-1].strip() if output else f"exit code {finished.returncode}"
        if len(last_line) > 70:
            last_line = last_line[:70] + "..."
        _sensor_probe_result = (False, last_line)
    return _sensor_probe_result


def choose_backend(requested):
    """Decide which eye backend to use. Returns (name, reason).

    requested = "raycast" -> always raycast.
    requested = "sensor"  -> sensor, or an error if it does not work here.
    requested = "auto"    -> sensor if it works, otherwise raycast.

    The environment variable FLYCAR_EYE_BACKEND, when set, replaces ``requested``.
    """
    source = "requested by the scenario"
    forced = os.environ.get(BACKEND_ENV_VAR, "").strip().lower()
    if forced:
        if forced not in BACKENDS:
            raise ValueError(f"{BACKEND_ENV_VAR} must be one of {', '.join(BACKENDS)}, got {forced!r}.")
        requested = forced
        source = f"set by {BACKEND_ENV_VAR}"

    if requested == "raycast":
        return "raycast", source

    works, reason = sensor_backend_works()
    if works:
        return "sensor", reason
    if requested == "sensor":
        raise RuntimeError(
            "The scenario asks for the 'sensor' eye backend, but it does not work on this machine "
            f"({reason}). It needs an NVIDIA GPU with OptiX. Use 'raycast' or 'auto' instead."
        )
    return "raycast", f"sensor backend unavailable ({reason})"


def make_eyes(backend, system, vehicle, world, config):
    """Create the eye object for a backend name returned by choose_backend.

    ``world`` is the ``World`` of this episode. The eyes ask it where the
    lights are (``world.current_lights()``) every time they need to know, and
    read the ambient and sky brightness from ``world.layout``.
    """
    if backend == "sensor":
        return SensorEyes(system, vehicle, world, config)
    return RaycastEyes(system, vehicle, world, config)


# ----------------------------------------------------------------------------
# Geometry shared by both backends
# ----------------------------------------------------------------------------

def eye_mounts(config):
    """Where each eye sits and which way it faces, in the chassis frame.

    Returns {"left": (position, yaw), "right": (position, yaw)}.
    The right eye is the mirror image of the left one.
    """
    forward, sideways, up = config["mount_m"]
    yaw = math.radians(config["yaw_offset_deg"])
    return {
        "left": (np.array([forward, sideways, up]), yaw),
        "right": (np.array([forward, -sideways, up]), -yaw),
    }


def pinhole_ray_directions(rows, cols, fov_h_deg, fov_v_deg, yaw):
    """Unit direction of the ray through the centre of every pixel.

    Directions are in the chassis frame (x forward, y left, z up), for an eye
    turned by ``yaw`` radians about the up axis. The result has shape
    (rows * cols, 3), ordered row by row from the top-left pixel.
    """
    half_width = math.tan(math.radians(fov_h_deg) / 2.0)
    half_height = math.tan(math.radians(fov_v_deg) / 2.0)

    directions = np.zeros((rows * cols, 3))
    for row in range(rows):
        # +1 at the top edge of the image, -1 at the bottom edge.
        vertical = 1.0 - 2.0 * (row + 0.5) / rows
        for col in range(cols):
            # +1 at the left edge of the image, -1 at the right edge.
            horizontal = 1.0 - 2.0 * (col + 0.5) / cols

            # A point on an imaginary screen 1 m in front of the eye.
            # (y is "left" in the chassis frame, so left pixels have y > 0.)
            x = 1.0
            y = horizontal * half_width
            z = vertical * half_height

            # Turn the eye outwards: rotate (x, y) about the up axis.
            x_turned = x * math.cos(yaw) - y * math.sin(yaw)
            y_turned = x * math.sin(yaw) + y * math.cos(yaw)

            length = math.sqrt(x_turned**2 + y_turned**2 + z**2)
            directions[row * cols + col] = (x_turned / length, y_turned / length, z / length)
    return directions


# ----------------------------------------------------------------------------
# Backend 1: ray casting on the CPU
# ----------------------------------------------------------------------------

class RaycastEyes:
    """CPU eyes built on ``ChCollisionSystem.RayHit``.

    For every pixel:

    1. Shoot a ray from the eye. If it hits nothing, the pixel shows sky.
    2. Otherwise look at the point it hit. The point receives the ambient
       light, plus light from each lamp. A lamp contributes more when it is
       close and when it shines squarely onto the surface, and nothing at all
       when something stands between the lamp and the point (a shadow ray
       checks that).
    3. If the pixel looks towards a lamp that is not hidden, add its glow.

    Rays hit *collision* shapes, so the eyes see exactly what the car can bump
    into (plus the ground, and the car's own bumper box at the bottom edge).
    """

    name = "raycast"

    def __init__(self, system, vehicle, world, config):
        self.system = system
        self.vehicle = vehicle
        self.world = world
        self.ambient = world.layout["lighting"]["ambient"]
        self.sky = world.layout["lighting"]["sky"]
        self.rows, self.cols = int(config["resolution"][0]), int(config["resolution"][1])
        self.max_range = float(config["max_range_m"])
        self.shadows = bool(config["shadows"])

        # Roughly the angle one pixel covers [rad]. Used to size the lamp glow.
        self.pixel_angle = math.radians(config["fov_h_deg"]) / self.cols

        # Ray directions never change in the chassis frame, so compute them once.
        self.mounts = eye_mounts(config)
        self.directions = {}
        for side, (_, yaw) in self.mounts.items():
            self.directions[side] = pinhole_ray_directions(
                self.rows, self.cols, config["fov_h_deg"], config["fov_v_deg"], yaw
            )

        # Reused for every ray, to avoid creating thousands of objects per frame.
        self._ray_start = chrono.ChVector3d(0, 0, 0)
        self._ray_end = chrono.ChVector3d(0, 0, 0)
        self._ray_result = chrono.ChRayhitResult()

    def update(self):
        """Nothing to do between frames (the sensor backend needs this hook)."""

    def render(self):
        """Take a picture with both eyes. Returns (left, right), float32 (rows, cols)."""
        chassis_position = self.vehicle.position()
        chassis_rotation = self.vehicle.rotation_matrix()

        # Ask the world where the lights are now (they may move one day).
        lights = self.world.current_lights()

        images = {}
        for side, (mount_position, _) in self.mounts.items():
            # Chassis frame -> world frame.
            origin = chassis_position + chassis_rotation @ mount_position
            directions = self.directions[side] @ chassis_rotation.T
            images[side] = self._render_one_eye(origin, directions, lights)
        return images["left"], images["right"]

    def raw_images(self):
        """This backend has no camera image."""
        return None, None

    # ---- internals ----------------------------------------------------------

    def _ray_hits_something(self, start, end):
        """True if anything lies on the straight segment from start to end."""
        self._ray_start.x, self._ray_start.y, self._ray_start.z = start
        self._ray_end.x, self._ray_end.y, self._ray_end.z = end
        return self.system.GetCollisionSystem().RayHit(self._ray_start, self._ray_end, self._ray_result)

    def _cast_rays(self, origin, directions):
        """Cast one ray per direction.

        Returns (distance, normal): the distance to the first hit along each
        ray (infinity for a miss) and the surface normal at that hit.
        """
        count = len(directions)
        distance = np.full(count, np.inf)
        normal = np.zeros((count, 3))

        collision_system = self.system.GetCollisionSystem()
        start = self._ray_start
        end = self._ray_end
        result = self._ray_result
        start.x, start.y, start.z = origin.tolist()
        end_points = (origin + directions * self.max_range).tolist()

        for i in range(count):
            end.x, end.y, end.z = end_points[i]
            if collision_system.RayHit(start, end, result):
                # dist_factor is the fraction of the segment travelled before the hit.
                distance[i] = result.dist_factor * self.max_range
                hit_normal = result.abs_hitNormal
                normal[i] = (hit_normal.x, hit_normal.y, hit_normal.z)
        return distance, normal

    def _render_one_eye(self, origin, directions, lights):
        distance, normal = self._cast_rays(origin, directions)
        hit = np.isfinite(distance)

        # Start with sky everywhere, then fill in the pixels that hit a surface.
        luminance = np.full(len(directions), self.sky)

        if hit.any():
            points = origin + directions[hit] * distance[hit, None]
            normals = normal[hit]

            # Light arriving at each point: ambient light, plus every lamp.
            received = np.full(len(points), self.ambient)
            for light in lights:
                lamp = np.array([light["x"], light["y"], light["z"]])
                to_lamp = lamp - points
                lamp_distance = np.linalg.norm(to_lamp, axis=1)
                to_lamp_unit = to_lamp / lamp_distance[:, None]

                # Lambert's cosine law: light landing squarely on a surface is
                # brighter than light skimming along it. Surfaces facing away
                # from the lamp (negative cosine) get nothing.
                facing = np.clip(np.sum(normals * to_lamp_unit, axis=1), 0.0, 1.0)

                # Light gets weaker with distance (half strength at falloff_m).
                strength = light["intensity"] / (1.0 + (lamp_distance / light["falloff_m"]) ** 2)
                contribution = strength * facing

                if self.shadows:
                    # Shadow rays: is the path from this point to the lamp clear?
                    lifted = points + normals * SHADOW_OFFSET_M
                    for i in np.nonzero(contribution > 1e-3)[0]:
                        if self._ray_hits_something(lifted[i].tolist(), lamp.tolist()):
                            contribution[i] = 0.0
                received += contribution

            luminance[hit] = ALBEDO * received

        # The lamps themselves. A lamp far away is smaller than one pixel, so
        # it shows up as a soft glow that gets brighter as the car gets closer.
        for light in lights:
            lamp = np.array([light["x"], light["y"], light["z"]])
            to_lamp = lamp - origin
            lamp_distance = float(np.linalg.norm(to_lamp))
            if lamp_distance < 1e-6:
                continue
            if self._ray_hits_something(origin.tolist(), lamp.tolist()):
                continue   # hidden behind a wall or an obstacle

            # Angle between each pixel's viewing direction and the lamp.
            cosine = np.clip(directions @ (to_lamp / lamp_distance), -1.0, 1.0)
            angle = np.arccos(cosine)

            # How big the lamp looks from here, compared with one pixel.
            lamp_angle = math.atan2(LAMP_RADIUS_M, lamp_distance)
            fill = min(1.0, (lamp_angle / (0.5 * self.pixel_angle)) ** 2)
            spread = max(self.pixel_angle, lamp_angle)
            luminance += light["intensity"] * fill * np.exp(-0.5 * (angle / spread) ** 2)

        luminance = np.clip(luminance, 0.0, 1.0)
        return luminance.reshape(self.rows, self.cols).astype(np.float32)


# ----------------------------------------------------------------------------
# Backend 2: Chrono sensor-module cameras (GPU)
# ----------------------------------------------------------------------------

class SensorEyes:
    """GPU eyes: one ``ChCameraSensor`` per eye, averaged down to the eye grid.

    NOTE: this backend needs an NVIDIA GPU with OptiX. It follows the bundled
    PyChrono sensor demos (demo_SEN_camera.py), but brightness is not calibrated
    against the raycast backend: expect similar pictures, not identical numbers.
    """

    name = "sensor"

    def __init__(self, system, vehicle, world, config):
        import pychrono.sensor as sens   # imported here: it only exists on some machines

        self.world = world

        self.rows, self.cols = int(config["resolution"][0]), int(config["resolution"][1])

        # Raw image size: a whole number of camera pixels per eye pixel, with
        # the aspect ratio that gives the requested vertical field of view.
        self.block_w = max(1, round(config["camera_width_px"] / self.cols))
        aspect = math.tan(math.radians(config["fov_v_deg"]) / 2.0) / math.tan(math.radians(config["fov_h_deg"]) / 2.0)
        self.block_h = max(1, round(self.block_w * self.cols * aspect / self.rows))
        self.width = self.block_w * self.cols
        self.height = self.block_h * self.rows

        # ---- Scene lighting -------------------------------------------------
        self.manager = sens.ChSensorManager(system)
        ambient = world.layout["lighting"]["ambient"]
        sky = world.layout["lighting"]["sky"]
        self.manager.scene.SetAmbientLight(chrono.ChVector3f(ambient, ambient, ambient))
        background = sens.Background()
        background.mode = sens.BackgroundMode_SOLID_COLOR
        background.color_zenith = chrono.ChVector3f(sky, sky, sky)
        self.manager.scene.SetBackground(background)
        # One point light per lamp. Remember each one's id and position, so
        # update() can move it if the world moves the lamp.
        self.light_ids = []
        self.light_positions = []
        for light in world.current_lights():
            position, color, max_range = self._point_light(light)
            self.light_ids.append(self.manager.scene.AddPointLight(position, color, max_range))
            self.light_positions.append((light["x"], light["y"], light["z"]))

        # ---- One camera per eye ---------------------------------------------
        update_rate_hz = 1.0 / config["update_period_s"]
        self.cameras = {}
        for side, (mount_position, yaw) in eye_mounts(config).items():
            offset = chrono.ChFramed(
                chrono.ChVector3d(float(mount_position[0]), float(mount_position[1]), float(mount_position[2])),
                chrono.QuatFromAngleZ(yaw),
            )
            camera = sens.ChCameraSensor(
                vehicle.chassis_body,                    # body the camera rides on
                update_rate_hz,                          # pictures per second
                offset,                                  # pose on that body
                self.width,
                self.height,
                math.radians(config["fov_h_deg"]),       # horizontal field of view
            )
            camera.SetName(f"eye_{side}")
            camera.SetLag(0.0)
            camera.SetCollectionWindow(0.0)
            camera.PushFilter(sens.ChFilterRGBA8Access())   # lets Python read the pixels
            self.manager.AddSensor(camera)
            self.cameras[side] = camera

        self._raw = {"left": None, "right": None}

    @staticmethod
    def _point_light(light):
        """(position, color, max_range) of the scene light for one lamp."""
        # Just above the lamp ball, so the ball does not block its own light.
        position = chrono.ChVector3f(light["x"], light["y"], light["z"] + LAMP_RADIUS_M + 0.1)
        color = chrono.ChColor(light["intensity"], light["intensity"], light["intensity"])
        return position, color, 10.0 * light["falloff_m"]

    def update(self):
        """Let the sensor manager render if a camera is due. Call every physics step."""
        # Move any scene light whose lamp has moved since the last step.
        for index, light in enumerate(self.world.current_lights()):
            position = (light["x"], light["y"], light["z"])
            if position != self.light_positions[index]:
                point, color, max_range = self._point_light(light)
                self.manager.scene.ModifyPointLight(self.light_ids[index], point, color, max_range)
                self.light_positions[index] = position
        self.manager.Update()

    def render(self):
        """Read both cameras. Returns (left, right), float32 (rows, cols)."""
        images = {}
        for side, camera in self.cameras.items():
            buffer = camera.GetMostRecentRGBA8Buffer()
            if buffer.HasData():
                # The buffer's first row is the BOTTOM of the picture: flip it.
                rgba = np.asarray(buffer.GetRGBA8Data())[::-1]
                self._raw[side] = np.ascontiguousarray(rgba[:, :, :3])

            raw = self._raw[side]
            if raw is None:
                # Rendering runs in the background; the first frame may not be ready.
                images[side] = np.zeros((self.rows, self.cols), dtype=np.float32)
                continue

            # RGB -> luminance in [0, 1] (standard Rec. 709 weights).
            rgb = raw.astype(np.float32) / 255.0
            gray = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]

            # Average each block of camera pixels into one eye pixel.
            blocks = gray.reshape(self.rows, self.block_h, self.cols, self.block_w)
            images[side] = np.clip(blocks.mean(axis=(1, 3)), 0.0, 1.0).astype(np.float32)
        return images["left"], images["right"]

    def raw_images(self):
        """The latest full-resolution RGB images (uint8), or None before the first frame."""
        return self._raw["left"], self._raw["right"]


def save_eye_figure(env, path):
    """Save both eye images, which pixels are in shadow, and the ray hits on a top-down map."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch

    from flycar.tools.plotting import draw_arena

    eyes = env.eyes
    left, right = eyes.render()
    rotation = env.vehicle.rotation_matrix()
    position = env.vehicle.position()
    lights = env.world.current_lights()
    views = {}
    for side, (mount, _) in eyes.mounts.items():
        origin = position + rotation @ mount
        directions = eyes.directions[side] @ rotation.T
        distance, normal = eyes._cast_rays(origin, directions)
        hit = np.isfinite(distance)
        shadow = np.zeros(eyes.rows * eyes.cols, dtype=bool)
        xy = np.full((eyes.rows * eyes.cols, 2), np.nan)
        if hit.any():
            points = origin + directions[hit] * distance[hit, None]
            normals = normal[hit]
            xy[hit, 0] = points[:, 0]
            xy[hit, 1] = points[:, 1]
            blocked = np.zeros(len(points), dtype=bool)
            if eyes.shadows:
                for light in lights:
                    lamp = np.array([light["x"], light["y"], light["z"]])
                    to_lamp = lamp - points
                    lamp_distance = np.linalg.norm(to_lamp, axis=1)
                    facing = np.clip(np.sum(normals * (to_lamp / lamp_distance[:, None]), axis=1), 0.0, 1.0)
                    contribution = facing * light["intensity"] / (1.0 + (lamp_distance / light["falloff_m"]) ** 2)
                    lifted = points + normals * SHADOW_OFFSET_M
                    for index in np.nonzero(contribution > 1e-3)[0]:
                        if eyes._ray_hits_something(lifted[index].tolist(), lamp.tolist()):
                            blocked[index] = True
            shadow[hit] = blocked
        views[side] = {"xy": xy, "shadow": shadow.reshape(eyes.rows, eyes.cols)}

    figure = Figure(figsize=(12, 6), dpi=100)
    FigureCanvasAgg(figure)
    grid = figure.add_gridspec(2, 3)
    for column, image, side in ((0, left, "left"), (1, right, "right")):
        eye_axis = figure.add_subplot(grid[0, column])
        eye_axis.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        eye_axis.set_title(f"{side} eye")
        eye_axis.set_axis_off()

        shadow_axis = figure.add_subplot(grid[1, column])
        shadow_axis.imshow(views[side]["shadow"], cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
        shadow_axis.set_title(f"{side} eye, pixels in shadow")
        shadow_axis.set_xticks([])
        shadow_axis.set_yticks([])
        shadow_axis.legend(
            handles=[Patch(facecolor="black", edgecolor="gray", label="in shadow")],
            loc="lower right", fontsize=7,
        )

    map_axis = figure.add_subplot(grid[:, 2])
    draw_arena(map_axis, env.layout)
    for side, color in (("left", "C0"), ("right", "C1")):
        xy = views[side]["xy"]
        hit = np.isfinite(xy[:, 0])
        map_axis.plot(xy[hit, 0], xy[hit, 1], ".", color=color, markersize=3, label=f"{side} eye hits")
    map_axis.set_title("ray hit points")
    map_axis.legend(fontsize=7)
    figure.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    print(path)
    return left, right


def show_fly_eyes(left, right):
    """Open a window with both eye images. Returns a function to keep that window alive."""
    import matplotlib.pyplot as plt

    plt.ion()
    figure, axes = plt.subplots(1, 2, figsize=(6, 3.2))
    for axis, image, title in ((axes[0], left, "left eye"), (axes[1], right, "right eye")):
        axis.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axis.set_title(title)
        axis.set_axis_off()
    figure.canvas.manager.set_window_title("What the fly sees")
    figure.tight_layout()
    figure.canvas.draw()
    figure.canvas.flush_events()

    def pump():
        if plt.fignum_exists(figure.number):
            figure.canvas.flush_events()

    return pump

# python -m flycar.env.sensing.eyes --x -5 --y 0 --yaw-deg 0 --eye right
#
# Runs this file. The car is parked at x = -5 m, y = 0 m, facing +x (yaw 0),
# in scenarios/team_vision.yaml with seed 0. The eyes are the raycast backend.
# It writes outputs/eyes_view.png: both eye images, which pixels are in shadow (black
# in the bottom row), and the ray hit points on a top-down map. A pixel is in
# shadow when something blocks the lamp from the spot it sees, so the lamp is
# drawn on the map too. Then two windows open. The 3D
# window looks through the right eye. The other window shows both eye pictures.
# Close the 3D window to quit.

if __name__ == "__main__":
    import argparse

    import pychrono.vehicle as veh

    from flycar.env.core import FlyCarEnv
    from flycar.env.scenario import load_scenario

    parser = argparse.ArgumentParser(description="Park the car and look through one eye at the 3D arena.")
    parser.add_argument("--scenario", default=str(Path(__file__).resolve().parents[3] / "scenarios" / "team_vision.yaml"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--x", type=float, required=True)
    parser.add_argument("--y", type=float, required=True)
    parser.add_argument("--yaw-deg", type=float, required=True)
    parser.add_argument("--eye", choices=("left", "right"), default="left")
    parser.add_argument(
        "-o", "--output",
        default=str(Path(__file__).resolve().parents[3] / "outputs" / "eyes_view.png"),
    )
    args = parser.parse_args()

    os.environ[BACKEND_ENV_VAR] = "raycast"
    scenario = load_scenario(args.scenario)
    scenario["spawn"]["x"] = args.x
    scenario["spawn"]["y"] = args.y
    scenario["spawn"]["yaw_deg"] = args.yaw_deg
    scenario["spawn"]["min_target_distance_m"] = 0.0
    scenario["eyes"]["backend"] = "raycast"

    env = FlyCarEnv(scenario, render_mode="human", eye_backend="raycast")
    try:
        env.reset(seed=args.seed)
        left, right = save_eye_figure(env, args.output)
        mount, yaw = env.eyes.mounts[args.eye]
        rotation = env.vehicle.rotation_matrix()
        origin = env.vehicle.position() + rotation @ mount
        forward = rotation @ np.array([math.cos(yaw), math.sin(yaw), 0.0])
        target = origin + forward * 20.0
        position = chrono.ChVector3d(float(origin[0]), float(origin[1]), float(origin[2]))
        look_at = chrono.ChVector3d(float(target[0]), float(target[1]), float(target[2]))

        window = env.window
        window.SetWindowTitle(f"Fly Car {args.eye} eye")
        window.SetChaseCameraPosition(position, look_at)
        window.SetChaseCameraState(veh.ChChaseCamera.Fixed)
        camera = window.GetActiveCamera()
        if camera is not None:
            camera.setFOV(math.radians(scenario["eyes"]["fov_v_deg"]))

        print(f"{args.eye} eye in the 3D window. The other window is what both eyes see. Close the 3D window to quit.")
        pump_eyes = show_fly_eyes(left, right)
        while window.Run():
            window.SetChaseCameraPosition(position, look_at)
            window.SetCameraPosition(position)
            window.SetCameraTarget(look_at)
            window.BeginScene()
            window.Render()
            window.EndScene()
            pump_eyes()
    finally:
        env.close()

