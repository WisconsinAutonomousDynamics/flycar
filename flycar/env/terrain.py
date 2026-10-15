"""The ground the car drives on.

Today this is one flat, rigid patch, a bit larger than the arena, with its top
surface at z = 0. It uses Chrono's ``veh.RigidTerrain``, which is what the
vehicle's tires query for the ground height, normal and friction under each
wheel.

The terrain uses the same two-phase update as the vehicle (see
``tutorials/03_vehicle_figure8.py``): ``synchronize(time)`` then
``advance(step)``, once per physics step, called by ``FlyCarEnv``.

Settings come from the ``terrain`` section of the scenario
(``flycar/config/vehicle.py``). Planned: several patches, ramps, and SCM
(deformable soil).
"""

import pychrono as chrono
import pychrono.vehicle as veh

# Bounciness of the ground. (The walls use the same value; see world.py.)
TERRAIN_RESTITUTION = 0.01


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
        patch = self.chrono_terrain.AddPatch(material, chrono.CSYSNORM, size_x + margin, size_y + margin)
        if visualize:
            patch.SetColor(chrono.ChColor(0.45, 0.45, 0.42))
            patch.SetTexture(veh.GetVehicleDataFile("terrain/textures/tile4.jpg"), size_x / 4.0, size_y / 4.0)
        self.chrono_terrain.Initialize()

    # The terrain uses the same two-phase update as the vehicle.

    def synchronize(self, time):
        """Phase 1: tell the terrain the current time."""
        self.chrono_terrain.Synchronize(time)

    def advance(self, step):
        """Phase 2: move the terrain forward in time."""
        self.chrono_terrain.Advance(step)
