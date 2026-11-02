"""The car: building it, and turning an action into Chrono driver inputs.

We use the stock **Gator** from ``pychrono.vehicle``, a small utility vehicle.
Measured in this simulation it has a 1.94 m wheelbase, turns in a circle of
about 2.9 m radius and tops out near 8 m/s. A speed governor (below) holds it
to a slower, fly-like pace.

Frames and signs (the same as Chrono's vehicle convention):

* The chassis frame has **x forward, y left, z up**. Its origin sits midway
  between the axles, about 0.275 m above the ground when the car is at rest.
* Steering is positive to the **left**.

Rate limiting and saturation of the action happen HERE, not in the drivers.
A driver can ask for anything; the car only does what is physically sensible.
"""

import math

import numpy as np
import pychrono as chrono
import pychrono.vehicle as veh

from flycar.env.spaces import BRAKE, STEERING, THROTTLE

# Height of the chassis frame above flat ground when the car is at rest [m].
RIDE_HEIGHT_M = 0.275

# The collision box ("bumper") we attach to the chassis, in the chassis frame.
#
# The Gator's own collision shape is a 1.0 m x 0.5 m box, much smaller than the
# car, so with it the wheels sink deep into a wall before anything touches.
# This box covers the whole visible body: the Gator's 3D mesh spans x = -2.2 m
# (the cargo bed overhangs the rear axle) to x = +1.26 m, and y = +-0.75 m.
BUMPER_X_MIN_M = -2.2   # rear face
BUMPER_X_MAX_M = 1.3    # front face
BUMPER_WIDTH_M = 1.5    # along y: from -0.75 m to +0.75 m
BUMPER_HEIGHT_M = 0.5   # along z: from 0 to 0.5 m above the chassis frame origin

# Tag stored on the chassis body so the contact sensor can recognise it.
CHASSIS_TAG = 500

# The speed governor fades the throttle to zero over this band below max speed.
GOVERNOR_BAND_MPS = 0.5


def limit_action(requested, previous, max_rates, dt):
    """Saturate an action and limit how fast it may change.

    requested : what the driver asked for, [throttle, steering, brake]
    previous  : what was applied on the previous control step
    max_rates : largest allowed change per second for each channel
    dt        : control period in seconds

    Returns the action that should actually be applied (float32, shape (3,)).
    """
    requested = np.asarray(requested, dtype=np.float64).reshape(3)

    # A NaN from a buggy driver must not reach the physics engine.
    requested = np.nan_to_num(requested, nan=0.0, posinf=1.0, neginf=-1.0)

    # 1) Saturation: keep every channel inside its legal range.
    low = np.array([0.0, -1.0, 0.0])
    high = np.array([1.0, 1.0, 1.0])
    target = np.clip(requested, low, high)

    # 2) Rate limiting: move from the previous value towards the target, but
    #    by no more than (rate * dt) in one control step.
    max_change = np.asarray(max_rates, dtype=np.float64) * dt
    change = np.clip(target - np.asarray(previous, dtype=np.float64), -max_change, max_change)
    return (np.asarray(previous, dtype=np.float64) + change).astype(np.float32)


class FlyCarVehicle:
    """A Gator placed in an existing Chrono system.

    Parameters
    ----------
    system : chrono.ChSystemNSC
        The physics system (owned by FlyCarEnv).
    x, y, yaw : float
        Spawn pose in the world frame [m, m, rad].
    config : dict
        The ``vehicle`` section of the scenario.
    physics_step : float
        Chrono step size [s], also used for the tire model.
    visualize : bool
        Load the 3D meshes. Only needed when a window is open.
    """

    def __init__(self, system, x, y, yaw, config, physics_step, visualize=False):
        self.config = config
        self.physics_step = physics_step

        # ---- Build the Gator ------------------------------------------------
        gator = veh.Gator(system)
        gator.SetContactMethod(chrono.ChContactMethod_NSC)
        gator.SetChassisCollisionType(veh.CollisionType_NONE)  # we add our own box below
        gator.SetChassisFixed(False)
        gator.SetBrakeType(veh.BrakeType_SHAFTS)
        gator.SetTireType(veh.TireModelType_TMEASY)
        gator.SetTireStepSize(physics_step)
        gator.SetInitFwdVel(0.0)

        # Spawn a couple of centimetres above the rest height; reset() lets the
        # car settle onto its suspension before the episode starts.
        spawn_position = chrono.ChVector3d(x, y, RIDE_HEIGHT_M + 0.02)
        spawn_rotation = chrono.QuatFromAngleZ(yaw)
        gator.SetInitPosition(chrono.ChCoordsysd(spawn_position, spawn_rotation))
        gator.Initialize()

        if visualize:
            gator.SetChassisVisualizationType(chrono.VisualizationType_MESH)
            gator.SetSuspensionVisualizationType(chrono.VisualizationType_PRIMITIVES)
            gator.SetSteeringVisualizationType(chrono.VisualizationType_PRIMITIVES)
            gator.SetWheelVisualizationType(chrono.VisualizationType_NONE)
            gator.SetTireVisualizationType(chrono.VisualizationType_MESH)
        else:
            gator.SetChassisVisualizationType(chrono.VisualizationType_NONE)
            gator.SetSuspensionVisualizationType(chrono.VisualizationType_NONE)
            gator.SetSteeringVisualizationType(chrono.VisualizationType_NONE)
            gator.SetWheelVisualizationType(chrono.VisualizationType_NONE)
            gator.SetTireVisualizationType(chrono.VisualizationType_NONE)

        self.gator = gator
        self.vehicle = gator.GetVehicle()          # the generic ChWheeledVehicle
        self.chassis_body = gator.GetChassisBody()
        self.chassis_body.SetTag(CHASSIS_TAG)

        # ---- Attach the bumper collision box --------------------------------
        bumper_material = chrono.ChContactMaterialNSC()
        bumper_material.SetFriction(0.5)
        bumper_material.SetRestitution(0.0)   # no bounce: the car stops dead against a wall
        bumper_length = BUMPER_X_MAX_M - BUMPER_X_MIN_M
        bumper_shape = chrono.ChCollisionShapeBox(
            bumper_material, bumper_length, BUMPER_WIDTH_M, BUMPER_HEIGHT_M
        )
        bumper_centre = chrono.ChVector3d(
            (BUMPER_X_MAX_M + BUMPER_X_MIN_M) / 2.0, 0.0, BUMPER_HEIGHT_M / 2.0
        )
        self.chassis_body.AddCollisionShape(bumper_shape, chrono.ChFramed(bumper_centre, chrono.QUNIT))
        self.chassis_body.EnableCollision(True)

        # ---- Control state ---------------------------------------------------
        self.inputs = veh.DriverInputs()   # what Chrono's vehicle reads each step
        self.applied_action = np.zeros(3, dtype=np.float32)
        self.governor_scale = 1.0          # last throttle factor of the speed governor [0, 1]
        self.terrain = None                # the ChTerrain seen in the last synchronize()
        self.max_rates = np.array(
            [
                config["throttle_rate_per_s"],
                config["steering_rate_per_s"],
                config["brake_rate_per_s"],
            ]
        )

    # ------------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------------

    def set_action(self, action, control_period):
        """Convert an action into Chrono driver inputs. Call once per control step.

        Returns the action that is actually applied after saturation and rate
        limiting (this is what ends up in ``info["applied_action"]``).
        """
        self.applied_action = limit_action(action, self.applied_action, self.max_rates, control_period)

        throttle = float(self.applied_action[THROTTLE])
        steering = float(self.applied_action[STEERING])
        brake = float(self.applied_action[BRAKE])

        # Speed governor: fade the throttle out as the car nears its top speed.
        # This is what makes the car "slow". It never brakes, it only stops
        # pushing, so the car can still roll faster than this downhill.
        max_speed = self.config["max_speed_mps"]
        headroom = (max_speed - self.forward_speed()) / GOVERNOR_BAND_MPS
        self.governor_scale = min(1.0, max(0.0, headroom))
        throttle = throttle * self.governor_scale

        self.inputs.m_throttle = throttle
        self.inputs.m_steering = steering
        self.inputs.m_braking = brake
        return self.applied_action

    def hold_brake(self):
        """Full brake, no throttle. Used while the car settles after a reset."""
        self.applied_action = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self.governor_scale = 1.0
        self.inputs.m_throttle = 0.0
        self.inputs.m_steering = 0.0
        self.inputs.m_braking = 1.0

    def release_controls(self):
        """Forget the control history, so the first action starts from zero."""
        self.applied_action = np.zeros(3, dtype=np.float32)
        self.governor_scale = 1.0
        self.inputs.m_throttle = 0.0
        self.inputs.m_steering = 0.0
        self.inputs.m_braking = 0.0

    # ------------------------------------------------------------------------
    # Chrono's two-phase update. See tutorials/03_vehicle_figure8.py.
    # ------------------------------------------------------------------------

    def synchronize(self, time, terrain):
        """Phase 1: tell the vehicle the current time, inputs and terrain."""
        self.terrain = terrain   # kept for extra_info(): tire forces are reported against it
        self.gator.Synchronize(time, self.inputs, terrain)

    def advance(self, step):
        """Phase 2: integrate the vehicle (and the whole system) by one step."""
        self.gator.Advance(step)

    # ------------------------------------------------------------------------
    # State, as plain NumPy / Python numbers (no Chrono types leak out)
    # ------------------------------------------------------------------------

    def position(self):
        """World position of the chassis frame origin, shape (3,) [m]."""
        p = self.chassis_body.GetFrameRefToAbs().GetPos()
        return np.array([p.x, p.y, p.z])

    def rotation_matrix(self):
        """3x3 matrix R whose columns are the chassis x, y, z axes in the world.

        ``R @ v_chassis`` converts a vector from the chassis frame to the world
        frame, and ``R.T @ v_world`` converts the other way.
        """
        q = self.chassis_body.GetFrameRefToAbs().GetRot()
        w, x, y, z = q.e0, q.e1, q.e2, q.e3
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )

    def velocity(self):
        """World-frame velocity of the chassis frame origin, shape (3,) [m/s]."""
        v = self.chassis_body.GetFrameRefToAbs().GetPosDt()
        return np.array([v.x, v.y, v.z])

    def forward_speed(self):
        """Speed along the chassis x axis [m/s]. Negative when rolling backwards."""
        forward_axis = self.rotation_matrix()[:, 0]
        return float(np.dot(self.velocity(), forward_axis))

    def yaw_rate(self):
        """Angular velocity about the chassis z (up) axis [rad/s]. Positive = turning left."""
        return float(self.chassis_body.GetAngVelLocal().z)

    def pose(self):
        """Ground-truth pose [x, y, yaw] in the world frame [m, m, rad]."""
        position = self.position()
        forward_axis = self.rotation_matrix()[:, 0]
        yaw = math.atan2(forward_axis[1], forward_axis[0])
        return np.array([position[0], position[1], yaw])

    # ------------------------------------------------------------------------
    # Debug data (info["extras"]["vehicle"]["vehicle"]); not part of the API
    # ------------------------------------------------------------------------

    def extra_info(self):
        """A look inside the car, as plain Python numbers.

        wheels             : per wheel (front_left, front_right, rear_left, rear_right):
            omega_radps    : wheel angular speed [rad/s], positive when rolling forwards
            slip           : longitudinal slip [-], positive when driving harder than rolling
            normal_force_n : vertical tire force [N]; the world z component, which is
                             the normal force on flat ground. NaN before the first step,
                             and with rigid tires (see below).
        governor_scale     : throttle factor of the speed governor (1 = no cut, 0 = full cut)
        engine_speed_radps : engine speed [rad/s] (NaN if the powertrain has no engine)
        gear               : current transmission gear (0 if there is no transmission)
        """
        wheels = {}
        for axle, axle_name in enumerate(("front", "rear")):
            for side, side_name in ((veh.LEFT, "left"), (veh.RIGHT, "right")):
                tire = self.vehicle.GetTire(axle, side)
                # Rigid tires report a summed contact force that swings by orders of
                # magnitude from step to step even with the car parked, so only the
                # TMeasy force is worth showing. ("tire_model" arrives with task A3.)
                if self.terrain is None or self.config.get("tire_model", "tmeasy") != "tmeasy":
                    normal_force = math.nan
                else:
                    normal_force = float(tire.ReportTireForce(self.terrain).force.z)
                wheels[f"{axle_name}_{side_name}"] = {
                    # Chrono reports forward rolling as negative; flip it.
                    "omega_radps": -float(self.vehicle.GetSpindleOmega(axle, side)),
                    "slip": float(tire.GetLongitudinalSlip()),
                    "normal_force_n": normal_force,
                }

        engine = self.vehicle.GetEngine()
        transmission = self.vehicle.GetTransmission()
        return {
            "wheels": wheels,
            "governor_scale": float(self.governor_scale),
            "engine_speed_radps": float(engine.GetMotorSpeed()) if engine is not None else math.nan,
            "gear": int(transmission.GetCurrentGear()) if transmission is not None else 0,
        }
