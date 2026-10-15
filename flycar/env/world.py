"""The world: walls, obstacles and lights, built from a scenario.

(The ground is not part of the world: it lives in ``terrain.py``.)

There are two steps, kept separate on purpose:

1. ``sample_layout(scenario, rng)`` rolls the dice. It turns the scenario
   (which may contain ``[min, max]`` ranges) into one concrete *layout*: plain
   numbers saying where everything is in this episode. No Chrono involved.
2. ``World(system, layout)`` creates the Chrono bodies for that layout.

The layout is plain Python data, so the logger, the replay tool and the
dashboard can draw the arena without importing pychrono. It is returned by
``env.reset()`` as ``info["world"]``.
"""

import math

import pychrono as chrono

from flycar.env.scenario import ScenarioError, sample

# Wall and obstacle bodies get tags counting up from here. The contact sensor
# uses the tag to tell "hit a wall or obstacle" apart from "touching the ground".
WORLD_TAG_BASE = 1000

# The car must spawn at least this far inside the arena walls [m]. Its rear
# corner is 2.3 m from the chassis origin, so anything less could start it
# inside a wall.
SPAWN_WALL_MARGIN_M = 3.0

# How many times to re-draw a random position before giving up.
MAX_SAMPLING_TRIES = 200

# A lamp is drawn as a glowing ball of this radius [m]. (The eyes use the same
# number to work out how big a lamp looks from a distance.)
LAMP_RADIUS_M = 0.4


# ----------------------------------------------------------------------------
# Step 1: sample a concrete layout (no Chrono)
# ----------------------------------------------------------------------------

def _wall_boxes(arena):
    """Four boxes around the arena. Their inner faces sit on the arena edge."""
    size_x, size_y = arena["size_m"]
    height = arena["wall_height_m"]
    thickness = arena["wall_thickness_m"]
    half_x = size_x / 2.0 + thickness / 2.0
    half_y = size_y / 2.0 + thickness / 2.0
    long_x = size_x + 2.0 * thickness
    return [
        {"shape": "box", "x": half_x, "y": 0.0, "yaw": 0.0, "size_m": [thickness, size_y, height]},    # east
        {"shape": "box", "x": -half_x, "y": 0.0, "yaw": 0.0, "size_m": [thickness, size_y, height]},   # west
        {"shape": "box", "x": 0.0, "y": half_y, "yaw": 0.0, "size_m": [long_x, thickness, height]},    # north
        {"shape": "box", "x": 0.0, "y": -half_y, "yaw": 0.0, "size_m": [long_x, thickness, height]},   # south
    ]


def _bounding_radius(obstacle):
    """Radius of the smallest circle (seen from above) that contains the obstacle."""
    if obstacle["shape"] == "cylinder":
        return float(obstacle["size_m"][0])
    length, width = obstacle["size_m"][0], obstacle["size_m"][1]
    return 0.5 * math.hypot(length, width)


def sample_layout(scenario, rng):
    """Draw one concrete episode layout from a scenario.

    ``rng`` is a ``numpy.random.Generator``. The same scenario and the same
    generator state always give the same layout, which is what makes
    ``reset(seed=...)`` reproducible.
    """
    arena = scenario["arena"]
    size_x, size_y = arena["size_m"]

    # ---- Lights -------------------------------------------------------------
    lights = []
    target = None
    for spec in scenario["lights"]:
        light = {
            "x": sample(spec["x"], rng),
            "y": sample(spec["y"], rng),
            "z": sample(spec["z"], rng),
            "intensity": float(spec["intensity"]),
            "falloff_m": float(spec["falloff_m"]),
            "target": bool(spec["target"]),
        }
        lights.append(light)
        if light["target"]:
            target = [light["x"], light["y"]]

    # ---- Spawn pose ---------------------------------------------------------
    spawn_spec = scenario["spawn"]
    limit_x = size_x / 2.0 - SPAWN_WALL_MARGIN_M
    limit_y = size_y / 2.0 - SPAWN_WALL_MARGIN_M
    spawn = None
    for _ in range(MAX_SAMPLING_TRIES):
        x = sample(spawn_spec["x"], rng)
        y = sample(spawn_spec["y"], rng)
        yaw = math.radians(sample(spawn_spec["yaw_deg"], rng))
        inside = abs(x) <= limit_x and abs(y) <= limit_y
        far_enough = True
        if target is not None:
            distance = math.hypot(x - target[0], y - target[1])
            far_enough = distance >= spawn_spec["min_target_distance_m"]
        if inside and far_enough:
            spawn = [x, y, yaw]
            break
    if spawn is None:
        raise ScenarioError(
            f"Could not find a spawn pose inside the arena (keeping {SPAWN_WALL_MARGIN_M} m from the walls) "
            f"and at least {spawn_spec['min_target_distance_m']} m from the target. Check the 'spawn' ranges."
        )

    # ---- Obstacles ----------------------------------------------------------
    obstacles = []
    for spec in scenario["obstacles"]:
        for _ in range(int(spec["count"])):
            placed = None
            for _ in range(MAX_SAMPLING_TRIES):
                candidate = {
                    "shape": spec["shape"],
                    "x": sample(spec["x"], rng),
                    "y": sample(spec["y"], rng),
                    "yaw": math.radians(sample(spec["yaw_deg"], rng)),
                    "size_m": [float(s) for s in spec["size_m"]],
                }
                # Keep the obstacle off the spawn point and off the target.
                keep_out = spec["clearance_m"] + _bounding_radius(candidate)
                clear = math.hypot(candidate["x"] - spawn[0], candidate["y"] - spawn[1]) >= keep_out
                if target is not None:
                    clear = clear and math.hypot(candidate["x"] - target[0], candidate["y"] - target[1]) >= keep_out
                if clear:
                    placed = candidate
                    break
            if placed is None:
                raise ScenarioError(
                    "Could not place an obstacle clear of the spawn point and the target. "
                    "Widen its x/y range or reduce clearance_m."
                )
            obstacles.append(placed)

    return {
        "arena": {
            "size_m": [float(size_x), float(size_y)],
            "walls": bool(arena["walls"]),
            "wall_height_m": float(arena["wall_height_m"]),
            "wall_thickness_m": float(arena["wall_thickness_m"]),
        },
        "walls": _wall_boxes(arena) if arena["walls"] else [],
        "obstacles": obstacles,
        "lights": lights,
        "target": target,
        "spawn": spawn,
        "lighting": {
            "ambient": float(scenario["lighting"]["ambient"]),
            "sky": float(scenario["lighting"]["sky"]),
        },
    }


# ----------------------------------------------------------------------------
# Step 2: build the Chrono bodies for a layout
# ----------------------------------------------------------------------------

class World:
    """Chrono bodies for one episode: walls, obstacles, lamps.

    Parameters
    ----------
    system : chrono.ChSystemNSC
        The physics system to add everything to.
    layout : dict
        The result of ``sample_layout``.
    visualize : bool
        Give the bodies visual shapes. Needed for the 3D window and for the
        ``sensor`` eye backend (cameras render visual shapes, not collision
        shapes). Left off otherwise, because it only costs time.
    """

    def __init__(self, system, layout, visualize=False):
        self.layout = layout
        self.bodies = []   # the wall and obstacle bodies, in creation order

        # One contact material shared by everything solid in the world.
        material = chrono.ChContactMaterialNSC()
        material.SetFriction(0.9)
        material.SetRestitution(0.01)

        # ---- Walls and obstacles --------------------------------------------
        for item in layout["walls"]:
            self._add_solid(system, item, material, visualize, chrono.ChColor(0.75, 0.75, 0.75))
        for item in layout["obstacles"]:
            self._add_solid(system, item, material, visualize, chrono.ChColor(0.55, 0.35, 0.25))

        # ---- Lamps ----------------------------------------------------------
        # A light has no collision shape (the car can drive through it). The
        # yellow ball is only there so people and cameras can see where it is.
        if visualize:
            for light in layout["lights"]:
                lamp = chrono.ChBodyEasySphere(LAMP_RADIUS_M, 1000.0, True, False)
                lamp.SetPos(chrono.ChVector3d(light["x"], light["y"], light["z"]))
                lamp.SetFixed(True)
                lamp.GetVisualShape(0).SetColor(chrono.ChColor(1.0, 0.9, 0.2))
                system.Add(lamp)

    def _add_solid(self, system, item, material, visualize, color):
        """Add one fixed box or cylinder that the car can collide with."""
        if item["shape"] == "box":
            length, width, height = item["size_m"]
            body = chrono.ChBodyEasyBox(length, width, height, 1000.0, visualize, True, material)
        else:
            radius, height = item["size_m"]
            body = chrono.ChBodyEasyCylinder(chrono.ChAxis_Z, radius, height, 1000.0, visualize, True, material)

        # Bodies are positioned by their centre, so lift them by half their
        # height to make them stand on the ground.
        body.SetPos(chrono.ChVector3d(item["x"], item["y"], height / 2.0))
        body.SetRot(chrono.QuatFromAngleZ(item["yaw"]))
        body.SetFixed(True)
        body.SetTag(WORLD_TAG_BASE + len(self.bodies))
        if visualize:
            body.GetVisualShape(0).SetColor(color)
        system.Add(body)
        self.bodies.append(body)

    def current_lights(self):
        """Where the lights are right now, in the world frame.

        Returns a list of dicts with the same keys as ``layout["lights"]``
        entries: x, y, z, intensity, falloff_m, target. The eyes call this every
        time they take a picture, so if lights ever move, only this method has
        to change. Today nothing moves, so it returns the layout lights.
        """
        return [dict(light) for light in self.layout["lights"]]

    # The world uses the same two-phase update as the vehicle and the terrain.
    # Nothing in it moves yet, so both phases are empty. (They are where moving
    # obstacles and lights will be updated.)

    def synchronize(self, time):
        """Phase 1: tell the world the current time."""

    def advance(self, step):
        """Phase 2: move the world forward in time."""
