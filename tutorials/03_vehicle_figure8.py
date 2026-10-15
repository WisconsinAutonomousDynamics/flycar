"""Tutorial 3: a real vehicle model driving a figure-8.

    python tutorials/03_vehicle_figure8.py
    python tutorials/03_vehicle_figure8.py --headless     # no windows, saves a PNG

What you learn here
-------------------
* ``pychrono.vehicle``: ready-made vehicles with suspension, steering, tires,
  an engine and brakes. We use the **Gator**, the same car as the Fly Car.
* ``DriverInputs``: the three numbers a vehicle understands, namely throttle,
  steering and braking. (The C++ class is called ``ChDriverInputs``; in
  PyChrono 10 it is exposed as ``veh.DriverInputs``.)
* The **Synchronize / Advance** pattern that every Chrono vehicle program uses.

Synchronize / Advance
---------------------
A vehicle simulation has several parts (vehicle, terrain, visualization) that
need to talk to each other. Each time step has two phases:

1. ``Synchronize(time, ...)``: every part reads its inputs for this instant.
   The vehicle reads the driver inputs and asks the terrain for the ground
   height under each tire.
2. ``Advance(step)``: every part moves forward in time by one step.

Things to try
-------------
1. Change ``STEERING``. How does the size of the 8 change?
2. Raise ``TARGET_SPEED`` to 7 m/s. Does the car still hold its line?
3. Replace the script with your own rule, e.g. steer with a sine wave.
"""

import argparse
import math
from pathlib import Path

import pychrono as chrono
import pychrono.vehicle as veh

TIME_STEP = 0.001      # s
TARGET_SPEED = 3.0    # m/s
STEERING = 0.7        # fraction of full lock used in the turns
WARMUP_TIME = 2.0      # s of driving straight before the first circle
MAX_TIME = 60.0        # s, safety stop


def scripted_driver(time, speed, turned):
    """The "driver": decides throttle, steering and braking.

    time   : simulation time [s]
    speed  : current forward speed [m/s]
    turned : how far the car has turned since the warm-up ended [rad]
             (positive = to the left)

    Returns (throttle, steering, braking, finished).
    """
    # Cruise control: push harder the further we are below the target speed.
    throttle = min(1.0, max(0.0, 0.5 * (TARGET_SPEED - speed)))
    braking = 0.0
    finished = False

    if time < WARMUP_TIME:
        steering = 0.0                    # 1) straight
    elif turned < 2.0 * math.pi and not scripted_driver.left_circle_done:
        steering = +STEERING              # 2) one full circle to the left
    else:
        scripted_driver.left_circle_done = True
        if turned > 0.0:
            steering = -STEERING          # 3) one full circle to the right
        else:
            steering = 0.0                # 4) done: stop
            throttle = 0.0
            braking = 1.0
            finished = True
    return throttle, steering, braking, finished


scripted_driver.left_circle_done = False   # the driver's only piece of memory

def sine_driver(time, speed, turned):
    """Steer with a sine wave: left, right, left, right..."""
    throttle = min(1.0, max(0.0, 0.5 * (TARGET_SPEED - speed)))   # same cruise control
    braking = 0.0
    finished = False

    # amplitude 0.6 of full lock; 0.25 Hz means one full wiggle every 4 seconds
    steering = 0.6 * math.sin(2.0 * math.pi * 0.25 * time)

    if time > 12.0:               # stop after 12 s of driving
        throttle = 0.0
        braking = 1.0
        steering = 0.0
        finished = True
    return throttle, steering, braking, finished

    

def main():
    parser = argparse.ArgumentParser(description="Tutorial 3: a Gator driving a figure-8.")
    parser.add_argument("--headless", action="store_true", help="No 3D window; save the plot to a PNG")
    args = parser.parse_args()

    # ---- 1. The vehicle ------------------------------------------------------
    # veh.Gator() bundles the chassis, suspensions, steering, wheels, tires,
    # engine, transmission and brakes. It also creates its own ChSystem.
    gator = veh.Gator()
    gator.SetContactMethod(chrono.ChContactMethod_NSC)
    gator.SetChassisFixed(False)
    gator.SetInitPosition(chrono.ChCoordsysd(chrono.ChVector3d(0, 0, 0.4), chrono.QUNIT))
    gator.SetBrakeType(veh.BrakeType_SHAFTS)
    gator.SetTireType(veh.TireModelType_TMEASY)   # a fast, semi-empirical tire model
    gator.SetTireStepSize(TIME_STEP)
    gator.Initialize()

    # How to draw each part (MESH = detailed 3D model, PRIMITIVES = simple shapes).
    gator.SetChassisVisualizationType(chrono.VisualizationType_MESH)
    gator.SetSuspensionVisualizationType(chrono.VisualizationType_PRIMITIVES)
    gator.SetSteeringVisualizationType(chrono.VisualizationType_PRIMITIVES)
    gator.SetWheelVisualizationType(chrono.VisualizationType_NONE)
    gator.SetTireVisualizationType(chrono.VisualizationType_MESH)

    vehicle = gator.GetVehicle()   # the generic vehicle object inside the Gator
    system = gator.GetSystem()     # the ChSystem the Gator created
    system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)

    # ---- 2. The terrain ------------------------------------------------------
    terrain = veh.RigidTerrain(system)
    ground_material = chrono.ChContactMaterialNSC()
    ground_material.SetFriction(0.9)
    patch = terrain.AddPatch(ground_material, chrono.CSYSNORM, 100, 100)   # 100 m x 100 m, flat
    patch.SetColor(chrono.ChColor(0.8, 0.8, 1.0))
    patch.SetTexture(veh.GetVehicleDataFile("terrain/textures/tile4.jpg"), 100, 100)
    terrain.Initialize()

    # ---- 3. Optional 3D window -----------------------------------------------
    window = None
    if not args.headless:
        window = veh.ChWheeledVehicleVisualSystemIrrlicht()
        window.SetWindowTitle("Tutorial 3: Gator figure-8")
        window.SetWindowSize(1000, 700)
        window.SetChaseCamera(chrono.ChVector3d(0, 0, 1.5), 9.0, 1.5)   # follow the car
        window.Initialize()
        window.AddLightDirectional()
        window.AddSkyBox()
        window.AttachVehicle(vehicle)
        vehicle.EnableRealtime(True)   # do not run faster than real time while we watch

    # ---- 4. The simulation loop ----------------------------------------------
    inputs = veh.DriverInputs()
    path_x, path_y = [], []
    turned = 0.0        # total angle turned since the warm-up [rad]
    step_number = 0

    while True:
        time = system.GetChTime()
        speed = vehicle.GetSpeed()

        # Keep track of how far the car has turned, by adding up the turn rate.
        if time >= WARMUP_TIME:
            turned += vehicle.GetYawRate() * TIME_STEP

        # The driver decides.
        throttle, steering, braking, finished = sine_driver(time, speed, turned)
        inputs.m_throttle = throttle
        inputs.m_steering = steering    # -1 = full right ... +1 = full left
        inputs.m_braking = braking

        # Phase 1: Synchronize. Everyone reads the current time and inputs.
        terrain.Synchronize(time)
        gator.Synchronize(time, inputs, terrain)
        if window is not None:
            window.Synchronize(time, inputs)

        # Phase 2: Advance. Everyone moves forward by one time step.
        # (The Gator owns the ChSystem, so its Advance() also steps the physics.)
        terrain.Advance(TIME_STEP)
        gator.Advance(TIME_STEP)
        if window is not None:
            window.Advance(TIME_STEP)

        # Record and draw every 20th step (50 times per simulated second).
        if step_number % 20 == 0:
            position = vehicle.GetPos()
            path_x.append(position.x)
            path_y.append(position.y)
            if window is not None:
                if not window.Run():
                    break                 # the window was closed
                window.BeginScene()
                window.Render()
                window.EndScene()
        step_number += 1

        if (finished and abs(speed) < 0.05) or time > MAX_TIME:
            break

    print(f"Finished after {system.GetChTime():.1f} s of simulated time.")
    print(f"The car ended {math.hypot(path_x[-1] - path_x[0], path_y[-1] - path_y[0]):.1f} m from where it started.")

    # ---- 5. Plot the path ----------------------------------------------------
    if args.headless:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    axis.plot(path_x, path_y, color="#2a78d6", linewidth=2)
    axis.plot(path_x[0], path_y[0], marker="o", markersize=8, color="#0b0b0b", linestyle="none")
    axis.annotate("start", (path_x[0], path_y[0]), textcoords="offset points", xytext=(8, -14), fontsize=9)
    axis.annotate("end", (path_x[-1], path_y[-1]), textcoords="offset points", xytext=(8, 8), fontsize=9)
    axis.set_aspect("equal")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_title(f"Gator figure-8 at {TARGET_SPEED:g} m/s, steering = {STEERING:g}", loc="left")
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
