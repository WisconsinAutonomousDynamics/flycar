"""Can the Chrono sensor module really render a camera image on this machine?

Importing ``pychrono.sensor`` proves very little. The module imports happily on
a laptop with no NVIDIA GPU and then fails (sometimes by killing the whole
Python process) the moment a camera is created, because it needs OptiX.

So the environment runs THIS script in a separate process and looks at the
exit code. If the render crashes, only the throw-away process dies.

    exit code 0  -> a camera image was rendered: the 'sensor' backend works
    anything else -> use the 'raycast' backend

You can run it by hand to see the real error message:

    python flycar/env/sensing/sensor_probe.py
"""

import sys


def main():
    import pychrono as chrono
    import pychrono.sensor as sens

    # The smallest possible scene: one fixed box and one light.
    system = chrono.ChSystemNSC()
    box = chrono.ChBodyEasyBox(1.0, 1.0, 1.0, 1000.0, True, False)
    box.SetPos(chrono.ChVector3d(3.0, 0.0, 0.0))
    box.SetFixed(True)
    system.Add(box)

    manager = sens.ChSensorManager(system)
    manager.scene.AddPointLight(chrono.ChVector3f(0.0, 0.0, 5.0), chrono.ChColor(1.0, 1.0, 1.0), 100.0)

    update_rate_hz = 100.0
    offset = chrono.ChFramed(chrono.ChVector3d(-3.0, 0.0, 0.0), chrono.QUNIT)
    camera = sens.ChCameraSensor(box, update_rate_hz, offset, 32, 32, 1.0)
    camera.PushFilter(sens.ChFilterRGBA8Access())
    manager.AddSensor(camera)

    # Rendering is asynchronous, so step until an image shows up.
    for _ in range(500):
        manager.Update()
        system.DoStepDynamics(0.005)
        buffer = camera.GetMostRecentRGBA8Buffer()
        if buffer.HasData():
            image = buffer.GetRGBA8Data()
            print(f"sensor probe: OK, rendered a {image.shape[1]}x{image.shape[0]} image")
            return 0

    print("sensor probe: FAILED, the camera never produced an image")
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"sensor probe: FAILED, {type(error).__name__}: {error}")
        sys.exit(1)
