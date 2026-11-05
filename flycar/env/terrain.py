"""The ground the car drives on.

Today this is one flat, rigid patch, a bit larger than the arena, with its top
surface at z = 0. It uses Chrono's ``veh.RigidTerrain``, which is what the
vehicle's tires query for the ground height, normal and friction under each
wheel.

The terrain uses the same two-phase update as the vehicle (see
``tutorials/03_vehicle_figure8.py``): ``synchronize(time)`` then
``advance(step)``, once per physics step, called by ``FlyCarEnv``.

Settings come from the ``terrain`` section of the scenario
(``flycar/config/vehicle.py``). Planned: ramps, and SCM (deformable soil).

Friction patches (``terrain.type: rigid_patches``)
---------------------------------------------------
Extra flat patches ("ice", "mud") with their own friction, each a
``RigidTerrain`` box patch with its own ``ChContactMaterialNSC``. Their top
surface is at z = 0, exactly level with the ground, so there is no step for a
tire to bump over.

Two patches at the same height overlap, so which friction does a tire get?
``RigidTerrain::FindPoint`` (Chrono 10.0, src/chrono_vehicle/terrain/
RigidTerrain.cpp) loops over the patches in the order they were added and only
replaces its answer when a patch is strictly HIGHER. So at equal height the
patch added FIRST wins. That is why the friction patches are added before the
ground patch. (The validator forbids patches overlapping each other.)

Notes:
* TMeasy tires (the default) read friction this way, so they see the patches.
  Rigid tires (``vehicle.tire_model: rigid``) touch the patch and the ground
  boxes through the collision system instead, so a patch only partly applies.
* The patches are drawn as thin coloured slabs 2 mm above the ground, so they
  do not flicker against the ground surface. That only moves the picture; the
  collision boxes stay level. The raycast eyes see every surface with the same
  albedo, so they cannot tell a patch from the ground.
"""

import pychrono as chrono
import pychrono.vehicle as veh

# Bounciness of the ground. (The walls use the same value; see world.py.)
TERRAIN_RESTITUTION = 0.01

# Friction patches are drawn as slabs this thick, lifted this much above the
# ground, so that their colour does not flicker against the ground surface.
PATCH_VISUAL_THICKNESS_M = 0.002
PATCH_VISUAL_LIFT_M = 0.002

# Patch colours: icy blue for low friction, dark brown for high friction.
ICE_COLOUR = (0.75, 0.90, 1.00)
GRIP_COLOUR = (0.35, 0.22, 0.10)
ICE_FRICTION = 0.1    # at or below this: fully icy blue
GRIP_FRICTION = 1.2   # at or above this: fully dark brown


def patch_colour(friction):
    """RGB colour (0..1) for a patch: blue for slippery, brown for grippy."""
    t = (float(friction) - ICE_FRICTION) / (GRIP_FRICTION - ICE_FRICTION)
    t = min(1.0, max(0.0, t))
    return tuple(ice + t * (grip - ice) for ice, grip in zip(ICE_COLOUR, GRIP_COLOUR))


class Terrain:
    """Chrono terrain for one episode.

    Parameters
    ----------
    system : chrono.ChSystemNSC
        The physics system to add the terrain to.
    layout : dict
        The result of ``sample_layout``. Only the arena size is used.
    terrain_config : dict
        The ``terrain`` section of the scenario.
    visualize : bool
        Give the ground a colour and texture (3D window and GPU cameras only).
    """

    def __init__(self, system, layout, terrain_config, visualize=False):
        self.config = terrain_config

        material = chrono.ChContactMaterialNSC()
        material.SetFriction(float(terrain_config["friction"]))
        material.SetRestitution(TERRAIN_RESTITUTION)

        # ---- Rigid, flat terrain --------------------------------------------
        # A patch a bit larger than the arena, with its top surface at z = 0.
        size_x, size_y = layout["arena"]["size_m"]
        margin = 2.0 * layout["arena"]["wall_thickness_m"] + 10.0
        self.chrono_terrain = veh.RigidTerrain(system)

        # Friction patches go in BEFORE the ground: at equal height the patch
        # added first wins (see the module docstring).
        self.friction_patches = []
        if terrain_config["type"] == "rigid_patches":
            for item in terrain_config["patches"]:
                self.friction_patches.append(self._add_friction_patch(item, visualize))

        patch = self.chrono_terrain.AddPatch(material, chrono.CSYSNORM, size_x + margin, size_y + margin)
        if visualize:
            patch.SetColor(chrono.ChColor(0.45, 0.45, 0.42))
            patch.SetTexture(veh.GetVehicleDataFile("terrain/textures/tile4.jpg"), size_x / 4.0, size_y / 4.0)
        self.chrono_terrain.Initialize()

    def _add_friction_patch(self, item, visualize):
        """One flat box patch, top surface at z = 0, with its own friction."""
        material = chrono.ChContactMaterialNSC()
        material.SetFriction(float(item["friction"]))
        material.SetRestitution(TERRAIN_RESTITUTION)

        size_x, size_y = (float(v) for v in item["size_m"])
        position = chrono.ChCoordsysd(chrono.ChVector3d(float(item["x"]), float(item["y"]), 0.0), chrono.QUNIT)
        # Arguments: material, position, length, width, thickness, tiled,
        # max_tile_size, visualization. Chrono's own box is not drawn: it would
        # sit exactly in the ground's surface and flicker.
        patch = self.chrono_terrain.AddPatch(material, position, size_x, size_y, 1.0, False, 1.0, False)

        if visualize:
            slab = chrono.ChVisualShapeBox(size_x, size_y, PATCH_VISUAL_THICKNESS_M)
            slab.SetColor(chrono.ChColor(*patch_colour(item["friction"])))
            lift = PATCH_VISUAL_LIFT_M - PATCH_VISUAL_THICKNESS_M / 2.0   # top of the slab at the lift height
            patch.GetGroundBody().AddVisualShape(slab, chrono.ChFramed(chrono.ChVector3d(0.0, 0.0, lift), chrono.QUNIT))
        return patch

    # The terrain uses the same two-phase update as the vehicle.

    def synchronize(self, time):
        """Phase 1: tell the terrain the current time."""
        self.chrono_terrain.Synchronize(time)

    def advance(self, step):
        """Phase 2: move the terrain forward in time."""
        self.chrono_terrain.Advance(step)
