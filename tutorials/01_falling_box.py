"""Tutorial 1: a falling box.

    python tutorials/01_falling_box.py
    python tutorials/01_falling_box.py --headless     # no window, saves a PNG

What you learn here
-------------------
* ``ChSystemNSC``  : the container that holds everything and advances time.
* A **body**       : something with mass, a position and a velocity.
* A **floor**      : a body that is fixed in space.
* The **step loop**: ``system.DoStepDynamics(dt)`` over and over.

We drop a box from 5 m and compare its height with the textbook formula for
free fall, ``z(t) = z0 - g t^2 / 2``. They agree until the box lands, and the
small difference between them teaches something about time steps.

Things to try
-------------
1. Change ``TIME_STEP`` to 0.01 and then to 0.0001. How does the error change?
2. Give the box an initial sideways velocity with ``box.SetPosDt(...)``.
3. Set the restitution to 0.8 on both materials. Does the box bounce?
"""

import argparse
from pathlib import Path

import numpy as np
import pychrono as chrono

GRAVITY = 9.81       # m/s^2
TIME_STEP = 0.001    # s
END_TIME = 2.0       # s
DROP_HEIGHT = 5.0    # m, height of the box centre at t = 0
BOX_SIZE = 0.5       # m, edge length of the cube


def simulate():
    """Run the simulation. Returns arrays of time and box height."""

    # ---- 1. The system -------------------------------------------------------
    # NSC means "non-smooth contact": bodies are perfectly hard, and contacts
    # are solved as constraints. (The alternative, SMC, models contacts as stiff
    # springs. NSC tolerates larger time steps, so we use it in this project.)
    system = chrono.ChSystemNSC()

    # Chrono's default "up" is +y. Vehicles use +z up, and so do we, everywhere.
    system.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -GRAVITY))

    # Bullet is the library that finds out which shapes touch each other.
    system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)

    # ---- 2. A material -------------------------------------------------------
    # The contact material says how surfaces behave when they touch.
    material = chrono.ChContactMaterialNSC()
    material.SetFriction(0.5)
    material.SetRestitution(0.8)   # 0 = no bounce, 1 = perfect bounce

    # ---- 3. The floor --------------------------------------------------------
    # ChBodyEasyBox(size_x, size_y, size_z, density, visualize, collide, material)
    # builds a box-shaped body with a matching visual and collision shape.
    floor = chrono.ChBodyEasyBox(10, 10, 1, 1000, True, True, material)
    floor.SetPos(chrono.ChVector3d(0, 0, -0.5))   # its top surface is at z = 0
    floor.SetFixed(True)                          # fixed bodies never move
    system.Add(floor)

    # ---- 4. The box ----------------------------------------------------------
    box = chrono.ChBodyEasyBox(BOX_SIZE, BOX_SIZE, BOX_SIZE, 1000, True, True, material)
    box.SetPos(chrono.ChVector3d(0, 0, DROP_HEIGHT))
    box.SetPosDt(chrono.ChVector3d(2, 0, 0))
    system.Add(box)
    print(f"The box weighs {box.GetMass():.1f} kg (its volume times the density, 1000 kg/m^3).")

    # ---- 5. The step loop ----------------------------------------------------
    times = []
    heights = []
    while system.GetChTime() < END_TIME:
        times.append(system.GetChTime())
        heights.append(box.GetPos().z)
        system.DoStepDynamics(TIME_STEP)   # advance everything by one time step

    print(f"Final box pos: {box.GetPos()}")

    return np.array(times), np.array(heights)


def main():
    parser = argparse.ArgumentParser(description="Tutorial 1: a falling box.")
    parser.add_argument("--headless", action="store_true", help="Save the plot to a PNG instead of opening a window")
    args = parser.parse_args()

    times, heights = simulate()

    # ---- Compare with the analytic solution ---------------------------------
    # Free fall, until the bottom of the box reaches the floor.
    resting_height = BOX_SIZE / 2.0
    analytic = np.maximum(DROP_HEIGHT - 0.5 * GRAVITY * times**2, resting_height)

    landing_time = np.sqrt(2.0 * (DROP_HEIGHT - resting_height) / GRAVITY)
    falling = times < landing_time - 0.01
    error_mm = (heights - analytic) * 1000.0
    worst = np.abs(error_mm[falling]).max()
    print(f"Analytic landing time: {landing_time:.3f} s")
    print(f"Largest difference from the formula while falling: {worst:.2f} mm")
    print(f"Final height of the box centre: {heights[-1]:.4f} m (expected {resting_height:.4f} m)")
    print()
    print("Why is there any difference at all? Chrono moves the box in small straight")
    print("steps of length dt. That is an approximation of the smooth parabola, and the")
    print("error grows with dt. Halve TIME_STEP and the error halves too.")

    # ---- Plot ----------------------------------------------------------------
    if args.headless:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, (ax_height, ax_error) = plt.subplots(2, 1, figsize=(7, 6.5), sharex=True, constrained_layout=True)

    ax_height.plot(times, heights, color="#2a78d6", linewidth=2, label="Chrono simulation")
    ax_height.plot(times, analytic, color="#eb6834", linewidth=2, linestyle="--", label="formula: z0 - g t^2 / 2")
    ax_height.set_ylabel("height of the box centre [m]")
    ax_height.set_title("A box dropped from 5 m", loc="left")
    ax_height.legend(frameon=False)

    ax_error.plot(times[falling], error_mm[falling], color="#2a78d6", linewidth=2)
    ax_error.set_xlabel("time [s]")
    ax_error.set_ylabel("simulation minus formula [mm]")
    ax_error.set_title(f"Error while falling (time step = {TIME_STEP * 1000:g} ms)", loc="left")

    for axis in (ax_height, ax_error):
        axis.grid(True, color="#e4e3df", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)

    if args.headless:
        output = Path(__file__).with_suffix(".png")
        figure.savefig(output, dpi=110)
        print(f"\nSaved the plot to {output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
