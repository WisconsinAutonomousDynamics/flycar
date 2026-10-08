"""Tutorial 2: a two-wheeled cart driven by motors.

    python tutorials/02_motor_cart.py
    python tutorials/02_motor_cart.py --headless     # no windows, saves a PNG

What you learn here
-------------------
* Building a machine out of several **bodies**.
* **Joints** (links) that connect bodies and remove some of their freedom.
* ``ChLinkMotorRotationSpeed``: a joint that also *drives* a rotation at a
  speed you choose.
* Steering by spinning two wheels at different speeds ("differential drive").
  A real car steers differently, but the idea of "commands in, motion out" is
  the same one the Fly Car environment is built on.

The cart
--------
    top view                 side view

        [L wheel]
    +---------------+          +---------------+
    | o  chassis  o |          |    chassis    |
    +---------------+          +-o-----O-----o-+
        [R wheel]             caster wheel caster      ---> +x (forward)

Two driven wheels on the sides, and a sliding ball ("caster") at the front and
at the back to stop the cart tipping over.

Things to try
-------------
1. Change the wheel speeds in ``PHASES``. Can you drive a square?
2. Make the wheels slippery (friction 0.1). Does the cart still follow the plan?
3. Double ``WHEEL_RADIUS``. What happens to the speed? Does the formula agree?
"""

import argparse
import math
from pathlib import Path

import pychrono as chrono

TIME_STEP = 0.002      # s

CHASSIS_SIZE = (0.6, 0.3, 0.1)   # length (x), width (y), height (z) [m]
WHEEL_RADIUS = 0.1               # m
WHEEL_WIDTH = 0.04               # m
TRACK = 0.4                      # m, distance between the two wheels
CASTER_RADIUS = 0.05             # m

# The plan: (duration [s], left wheel speed [rad/s], right wheel speed [rad/s]).
PHASES = [
    (3.0, 5.0, 5.0),    # both wheels the same speed   -> straight ahead
    (4.0, 3.0, 6.0),    # right wheel faster           -> curve to the left
    (3.0, 5.0, 5.0),    # straight again
    (1.0, -3.0, 3.0),   # opposite speeds              -> spin on the spot (about 90 degrees)
    (3.0, 5.0, 5.0),    # straight
]


def build_cart(system):
    """Create the cart. Returns (chassis, left_motor, right_motor)."""
    material = chrono.ChContactMaterialNSC()
    material.SetFriction(0.9)

    # The caster must slide sideways freely, so it gets a slippery material.
    slippery = chrono.ChContactMaterialNSC()
    slippery.SetFriction(0.0)

    axle_height = WHEEL_RADIUS   # the wheels rest on the floor (z = 0)

    # ---- Chassis -------------------------------------------------------------
    # collide=False: the chassis never touches anything, so skip the collision shape.
    chassis = chrono.ChBodyEasyBox(*CHASSIS_SIZE, 500, True, False)
    chassis.SetPos(chrono.ChVector3d(0, 0, axle_height + 0.02))
    system.Add(chassis)

    # ---- Wheels and motors ---------------------------------------------------
    motors = []
    for side in (+1, -1):   # +1 = left (y > 0), -1 = right (y < 0)
        wheel_position = chrono.ChVector3d(0, side * TRACK / 2.0, axle_height)

        # A cylinder whose axis points along y, like an axle.
        wheel = chrono.ChBodyEasyCylinder(chrono.ChAxis_Y, WHEEL_RADIUS, WHEEL_WIDTH, 800, True, True, material)
        wheel.SetPos(wheel_position)
        system.Add(wheel)

        # A rotation motor is a hinge that turns at a commanded speed.
        # It always spins about the z axis of the frame you give it, so we
        # build a frame whose z axis points along the axle (the world y axis).
        # Rotating by -90 degrees about x tips the z axis over onto +y.
        motor_frame = chrono.ChFramed(wheel_position, chrono.QuatFromAngleX(-chrono.CH_PI_2))

        motor = chrono.ChLinkMotorRotationSpeed()
        motor.Initialize(wheel, chassis, motor_frame)        # (driven body, base body, frame)
        motor.SetSpeedFunction(chrono.ChFunctionConst(0.0))  # stand still for now
        system.Add(motor)
        motors.append(motor)

    # ---- Casters -------------------------------------------------------------
    # One at the front and one at the back. With only a front caster, the
    # motors' reaction torque flips the chassis over backwards (try it!).
    for end in (+1, -1):
        caster = chrono.ChBodyEasySphere(CASTER_RADIUS, 800, True, True, slippery)
        caster_position = chrono.ChVector3d(end * (CHASSIS_SIZE[0] / 2.0 - 0.05), 0, CASTER_RADIUS)
        caster.SetPos(caster_position)
        system.Add(caster)

        # ChLinkLockLock welds two bodies together: zero freedom left between
        # them. Together with zero friction this makes a simple sliding skid.
        weld = chrono.ChLinkLockLock()
        weld.Initialize(caster, chassis, chrono.ChFramed(caster_position))
        system.Add(weld)

    left_motor, right_motor = motors
    return chassis, left_motor, right_motor


def heading_of(body):
    """Yaw angle of a body [rad]: the direction its x axis points, seen from above."""
    x_axis = body.GetRot().GetAxisX()
    return math.atan2(x_axis.y, x_axis.x)


def main():
    parser = argparse.ArgumentParser(description="Tutorial 2: a two-wheeled cart driven by motors.")
    parser.add_argument("--headless", action="store_true", help="No 3D window; save the plot to a PNG")
    args = parser.parse_args()

    # ---- System and floor ----------------------------------------------------
    system = chrono.ChSystemNSC()
    system.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -9.81))
    system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)

    floor_material = chrono.ChContactMaterialNSC()
    floor_material.SetFriction(0.9)
    floor = chrono.ChBodyEasyBox(20, 20, 1, 1000, True, True, floor_material)
    floor.SetPos(chrono.ChVector3d(0, 0, -0.5))
    floor.SetFixed(True)
    floor.GetVisualShape(0).SetColor(chrono.ChColor(0.6, 0.6, 0.6))
    system.Add(floor)

    chassis, left_motor, right_motor = build_cart(system)

    # ---- Optional 3D window --------------------------------------------------
    window = None
    if not args.headless:
        import pychrono.irrlicht as chronoirr

        window = chronoirr.ChVisualSystemIrrlicht()
        window.AttachSystem(system)
        window.SetCameraVertical(chrono.CameraVerticalDir_Z)   # tell the camera that z is up
        window.SetWindowSize(1000, 700)
        window.SetWindowTitle("Tutorial 2: motor cart")
        window.Initialize()
        window.AddSkyBox()
        window.AddCamera(chrono.ChVector3d(-3, -5, 4), chrono.ChVector3d(2, 1, 0))
        window.AddTypicalLights()

    # ---- Run the plan --------------------------------------------------------
    path_x, path_y = [], []
    print("phase   wheels L/R [rad/s]   speed [m/s] predicted/measured   turn rate [rad/s] predicted/measured")

    for number, (duration, left_speed, right_speed) in enumerate(PHASES, start=1):
        # Give each motor its new speed command.
        left_motor.SetSpeedFunction(chrono.ChFunctionConst(left_speed))
        right_motor.SetSpeedFunction(chrono.ChFunctionConst(right_speed))

        phase_end = system.GetChTime() + duration
        window_open = True
        while system.GetChTime() < phase_end and window_open:
            system.DoStepDynamics(TIME_STEP)
            path_x.append(chassis.GetPos().x)
            path_y.append(chassis.GetPos().y)

            # Draw at 50 frames per second, not at every physics step.
            if window is not None and len(path_x) % 10 == 0:
                window_open = window.Run()
                window.BeginScene()
                window.Render()
                window.EndScene()
        if not window_open:
            break

        # Differential-drive kinematics, for wheels that roll without slipping:
        #   forward speed = radius * (left + right) / 2
        #   turn rate     = radius * (right - left) / track
        predicted_speed = WHEEL_RADIUS * (left_speed + right_speed) / 2.0
        predicted_turn = WHEEL_RADIUS * (right_speed - left_speed) / TRACK

        velocity = chassis.GetPosDt()
        heading = heading_of(chassis)
        measured_speed = velocity.x * math.cos(heading) + velocity.y * math.sin(heading)
        measured_turn = chassis.GetAngVelParent().z

        print(f"  {number}        {left_speed:5.1f} / {right_speed:4.1f}          "
              f"{predicted_speed:5.2f} / {measured_speed:5.2f}                   "
              f"{predicted_turn:5.2f} / {measured_turn:5.2f}")

    print("\nThe prediction assumes perfect rolling. The small differences come from tire slip.")

    # ---- Plot the path -------------------------------------------------------
    if args.headless:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6.5, 6), constrained_layout=True)
    axis.plot(path_x, path_y, color="#2a78d6", linewidth=2)
    axis.plot(path_x[0], path_y[0], marker="o", markersize=8, color="#0b0b0b", linestyle="none")
    axis.annotate("start", (path_x[0], path_y[0]), textcoords="offset points", xytext=(8, -14), fontsize=9)
    axis.annotate("end", (path_x[-1], path_y[-1]), textcoords="offset points", xytext=(8, 6), fontsize=9)
    axis.set_aspect("equal")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_title("Path of the cart, seen from above", loc="left")
    axis.grid(True, color="#e4e3df", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)

    if args.headless:
        output = Path(__file__).with_suffix(".png")
        figure.savefig(output, dpi=110)
        print(f"Saved the plot to {output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
