"""FlyCarEnv: the Gymnasium environment that ties everything together.

This file owns the Chrono ``ChSystem`` and the time loop. The pieces it uses:

    scenario.py   reads the YAML experiment description
    world.py      terrain, walls, obstacles, lights
    vehicle.py    the car, and action -> driver inputs
    terrain.py    the ground
    sensing/      eyes, IMU, contact
    spaces.py     what an action and an observation look like

Typical use::

    from flycar.env.core import FlyCarEnv

    env = FlyCarEnv("scenarios/light_seek.yaml")
    obs, info = env.reset(seed=0)
    for _ in range(1000):
        action = [0.5, 0.0, 0.0]            # [throttle, steering, brake]
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    env.close()

Two clocks
----------
* The **control period** (10 ms by default) is how often a controller gets to
  act. One ``step()`` call is one control period.
* The **physics step** (1 ms by default) is how finely Chrono integrates.

So each ``step()`` holds the action constant and advances Chrono several
physics steps (10 by default).
"""

import math

import gymnasium
import numpy as np
import pychrono as chrono

from flycar.env.scenario import load_scenario
from flycar.env.sensing.contact import ContactSensor
from flycar.env.sensing.eyes import choose_backend, make_eyes
from flycar.env.sensing.imu import Imu
from flycar.env.spaces import make_action_space, make_observation_space
from flycar.env.terrain import Terrain
from flycar.env.vehicle import FlyCarVehicle
from flycar.env.world import World, sample_layout

# After a reset the car is dropped onto its suspension and held on the brakes
# for this long before the episode clock starts [s].
SETTLE_TIME_S = 0.5

# The 3D window is redrawn at most this often, in simulated time [s].
RENDER_PERIOD_S = 0.02


class FlyCarEnv(gymnasium.Env):
    """A car in a walled arena, with compound eyes, an IMU and touch sensors.

    Parameters
    ----------
    scenario : str, Path or dict
        A scenario YAML file (see ``scenarios/``) or an already-loaded dict.
    render_mode : None or "human"
        ``"human"`` opens a 3D Irrlicht window that follows the car.
        ``None`` (the default) runs headless, which is faster and deterministic.
    eye_backend : None, "auto", "sensor" or "raycast"
        Overrides ``eyes.backend`` from the scenario.
    """

    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, scenario, render_mode=None, eye_backend=None):
        super().__init__()
        self.scenario = load_scenario(scenario)
        if eye_backend is not None:
            self.scenario["eyes"]["backend"] = eye_backend

        if render_mode not in (None, "human"):
            raise ValueError(f"render_mode must be None or 'human', got {render_mode!r}")
        self.render_mode = render_mode

        # ---- Timing ---------------------------------------------------------
        timing = self.scenario["timing"]
        self.control_period = float(timing["control_period_s"])
        self.physics_step = float(timing["physics_step_s"])
        self.substeps = int(round(self.control_period / self.physics_step))
        self.eye_every = int(round(self.scenario["eyes"]["update_period_s"] / self.control_period))
        self.render_every = max(1, int(round(RENDER_PERIOD_S / self.control_period)))

        # ---- Spaces (defined once, in spaces.py) ----------------------------
        self.action_space = make_action_space()
        self.observation_space = make_observation_space(self.scenario["eyes"]["resolution"])

        # ---- Eye backend ----------------------------------------------------
        self.eye_backend, reason = choose_backend(self.scenario["eyes"]["backend"])
        print(f"[flycar] eye backend: {self.eye_backend} ({reason})")

        # Everything below is (re)built by reset().
        self.system = None
        self.vehicle = None
        self.terrain = None
        self.world = None
        self.eyes = None
        self.imu = None
        self.contact = None
        self.layout = None
        self.window = None
        self.window_closed = False
        self.step_count = 0
        self._eye_left = None
        self._eye_right = None

    # ------------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        """Start a new episode. Returns ``(observation, info)``.

        The whole Chrono system is thrown away and rebuilt. That is the
        simplest way to guarantee that nothing leaks from one episode into the
        next, and it only takes a few milliseconds.

        With a ``seed`` the episode layout (spawn pose, light and obstacle
        positions) is reproducible. Without one, the random stream simply
        continues, so successive resets give different layouts.
        """
        super().reset(seed=seed)   # seeds self.np_random
        self._close_window()

        # 1) Roll the dice: where is everything in this episode?
        self.layout = sample_layout(self.scenario, self.np_random)
        spawn_x, spawn_y, spawn_yaw = self.layout["spawn"]

        # 2) Build the physics system and everything in it.
        #    Visual shapes are only needed for a window or for GPU cameras.
        needs_visuals = self.render_mode == "human" or self.eye_backend == "sensor"
        self.system = self._make_system()
        self.vehicle = FlyCarVehicle(
            self.system, spawn_x, spawn_y, spawn_yaw,
            self.scenario["vehicle"], self.physics_step, visualize=needs_visuals,
        )
        # The terrain is built before the world, so Chrono receives its bodies
        # in the same order as always (the ground patch, then the walls).
        self.terrain = Terrain(self.system, self.layout, self.scenario["terrain"], visualize=needs_visuals)
        self.world = World(self.system, self.layout, visualize=needs_visuals)

        # Create the collision system LAST, so that it picks up every collision
        # shape added above (the bundled Chrono vehicle demos do the same).
        self.system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)

        # 3) Sensors.
        self.eyes = make_eyes(self.eye_backend, self.system, self.vehicle, self.layout, self.scenario["eyes"])
        self.imu = Imu(self.vehicle)
        self.contact = ContactSensor(self.system, self.vehicle)

        if self.render_mode == "human":
            self._open_window()

        # 4) Let the car settle on its suspension with the brakes on, so the
        #    episode starts from rest instead of from a small drop.
        self.vehicle.hold_brake()
        for _ in range(int(round(SETTLE_TIME_S / self.physics_step))):
            self._physics_step()
        self.vehicle.release_controls()

        # 5) Start the episode clock.
        self.step_count = 0
        self.imu.reset()
        self.contact.clear()
        self._eye_left, self._eye_right = self.eyes.render()

        observation = self._observation()
        info = self._info()
        info["world"] = self.layout
        return observation, info

    def step(self, action):
        """Advance one control period.

        Returns ``(observation, reward, terminated, truncated, info)``.

        * ``reward`` is always 0.0 for now (kept for Gymnasium compatibility).
        * ``terminated`` becomes True when the car reaches the target light.
        * ``truncated`` becomes True when the episode runs out of time.
        """
        if self.system is None:
            raise RuntimeError("Call reset() before step().")

        # The action is held constant for the whole control period.
        self.vehicle.set_action(action, self.control_period)
        self.contact.clear()

        # Advance Chrono by several small physics steps.
        for _ in range(self.substeps):
            self._physics_step()
        self.step_count += 1

        # The eyes take a new picture every few control steps. In between,
        # the observation repeats the last picture (like a real camera).
        if self.step_count % self.eye_every == 0:
            self._eye_left, self._eye_right = self.eyes.render()

        observation = self._observation()
        info = self._info()

        terminated = bool(info["success"])
        out_of_time = info["time"] >= self.scenario["episode"]["max_time_s"] - 1e-9
        truncated = bool(out_of_time and not terminated)

        if self.render_mode == "human" and self.step_count % self.render_every == 0:
            self.render()

        return observation, 0.0, terminated, truncated, info

    def render(self):
        """Redraw the 3D window (only does something with render_mode="human")."""
        if self.window is None:
            return
        if not self.window.Run():      # the user closed the window
            self.window_closed = True
            return
        self.window.BeginScene()
        self.window.Render()
        self.window.EndScene()

    def close(self):
        """Release the window and the Chrono objects."""
        self._close_window()
        self.eyes = None
        self.imu = None
        self.contact = None
        self.world = None
        self.terrain = None
        self.vehicle = None
        self.system = None

    # ------------------------------------------------------------------------
    # The physics loop
    # ------------------------------------------------------------------------

    def _make_system(self):
        """Create an empty Chrono system set up the way vehicles need it."""
        system = chrono.ChSystemNSC()   # NSC = non-smooth (hard) contacts

        # Chrono's default "up" is +y. Vehicles use z up, so gravity is -z.
        system.SetGravitationalAcceleration(chrono.ChVector3d(0.0, 0.0, -9.81))

        # The same solver settings Chrono picks when a vehicle creates its own system.
        system.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)
        system.GetSolver().AsIterative().SetMaxIterations(150)
        system.SetMaxPenetrationRecoverySpeed(4.0)

        # One thread: multi-threaded runs can add floating-point numbers in a
        # different order from run to run, which breaks determinism.
        system.SetNumThreads(1, 1, 1)
        return system

    def _physics_step(self):
        """Advance everything by one physics step."""
        time = self.system.GetChTime()
        step = self.physics_step

        # Phase 1, "Synchronize": every subsystem reads the current time and
        # its inputs (the car reads the driver inputs and the terrain).
        # This order is a frozen contract (docs/CONTRACTS.md).
        self.terrain.synchronize(time)
        self.world.synchronize(time)
        self.vehicle.synchronize(time, self.terrain.chrono_terrain)
        if self.window is not None:
            self.window.Synchronize(time, self.vehicle.inputs)

        # Phase 2, "Advance": every subsystem moves forward in time.
        self.terrain.advance(step)
        self.world.advance(step)
        self.vehicle.advance(step)
        if self.window is not None:
            self.window.Advance(step)

        # We created the ChSystem ourselves, so we are the ones who step it.
        # (A vehicle only steps the system when it created that system itself.)
        self.system.DoStepDynamics(step)

        # Sensors that need to look at every physics step.
        self.eyes.update()
        self.contact.poll()

    # ------------------------------------------------------------------------
    # Observation and info
    # ------------------------------------------------------------------------

    def _time(self):
        """Episode time [s]. Counting steps avoids floating-point drift."""
        return self.step_count * self.control_period

    def _observation(self):
        """Assemble the observation dictionary described in spaces.py."""
        return {
            "eye_left": self._eye_left,
            "eye_right": self._eye_right,
            "imu": self.imu.read(self.control_period),
            "contact": self.contact.read(),
            "time": np.array([self._time()], dtype=np.float32),
        }

    def _info(self):
        """Ground truth and debugging data. See INFO_FIELDS in spaces.py."""
        pose = self.vehicle.pose()
        x, y, yaw = pose

        target = self.layout["target"]
        if target is None:
            target_distance = float("nan")
            target_bearing = float("nan")
            success = False
        else:
            dx = target[0] - x
            dy = target[1] - y
            target_distance = math.hypot(dx, dy)
            # Angle to the target relative to the car's heading, wrapped to (-pi, pi].
            bearing = math.atan2(dy, dx) - yaw
            target_bearing = math.atan2(math.sin(bearing), math.cos(bearing))
            success = target_distance <= self.scenario["episode"]["success_radius_m"]

        camera_left, camera_right = self.eyes.raw_images()
        return {
            "pose": pose,
            "target_distance": target_distance,
            "target_bearing": target_bearing,
            "eye_backend": self.eye_backend,
            "camera_left": camera_left,
            "camera_right": camera_right,
            "applied_action": self.vehicle.applied_action.copy(),
            "time": self._time(),
            "step": self.step_count,
            "success": success,
        }

    # ------------------------------------------------------------------------
    # The optional 3D window
    # ------------------------------------------------------------------------

    def _open_window(self):
        """Open an Irrlicht window with a camera that chases the car."""
        import pychrono.vehicle as veh   # only needed here

        window = veh.ChWheeledVehicleVisualSystemIrrlicht()
        window.SetWindowTitle(f"Fly Car - {self.scenario['name']}")
        window.SetWindowSize(1000, 700)
        # Camera: looks at a point above the chassis, from 8 m behind and 1.5 m up.
        window.SetChaseCamera(chrono.ChVector3d(0.0, 0.0, 1.0), 8.0, 1.5)
        window.Initialize()
        window.AddLightDirectional()
        window.AddSkyBox()
        window.AttachVehicle(self.vehicle.vehicle)
        self.window = window
        self.window_closed = False

    def _close_window(self):
        if self.window is not None:
            try:
                self.window.GetDevice().closeDevice()
            except Exception:
                pass   # the window may already be gone
            self.window = None
