"""Touch sensing: did the chassis bump into a wall or an obstacle?

Chrono finds every contact in the system on every physics step. We ask it to
report them to a small callback object, keep the ones between the chassis and
a wall or obstacle, and sort them by where on the car they happened.

Output: float32 ``[left, front, right]``, each 0 or 1. A flag is 1 if that part
of the car touched something at any time during the control step.

Zones (chassis frame: x forward, y left):

* front : the contact is on the front face of the bumper box
* left  : anywhere else with y > 0
* right : anywhere else with y < 0

A contact at the very back therefore counts as left or right. There is no rear
sensor, because the car cannot drive backwards.
"""

import numpy as np
import pychrono as chrono

from flycar.env.spaces import CONTACT_FRONT, CONTACT_LEFT, CONTACT_RIGHT
from flycar.env.vehicle import BUMPER_X_MAX_M, CHASSIS_TAG
from flycar.env.world import WORLD_TAG_BASE

# Contacts within this distance of the front face count as "front" [m].
FRONT_ZONE_DEPTH_M = 0.25


class _ContactCollector(chrono.ReportContactCallback):
    """Chrono calls OnReportContact once for every contact in the system."""

    def __init__(self):
        super().__init__()
        self.points = []   # world-frame contact points on the chassis

    def OnReportContact(self, point_a, point_b, plane, distance, radius, force, torque, object_a, object_b, offset):
        tag_a = chrono.CastToChBody(object_a).GetTag()
        tag_b = chrono.CastToChBody(object_b).GetTag()

        # Keep only chassis-versus-world contacts. (Walls and obstacles have
        # tags >= WORLD_TAG_BASE; the ground does not, so it is ignored.)
        if tag_a == CHASSIS_TAG and tag_b >= WORLD_TAG_BASE:
            self.points.append((point_a.x, point_a.y, point_a.z))
        elif tag_b == CHASSIS_TAG and tag_a >= WORLD_TAG_BASE:
            self.points.append((point_b.x, point_b.y, point_b.z))
        return True   # True = keep reporting the remaining contacts


class ContactSensor:
    """Left / front / right touch flags for the chassis."""

    def __init__(self, system, vehicle):
        self.system = system
        self.vehicle = vehicle
        self.collector = _ContactCollector()
        self.flags = np.zeros(3, dtype=np.float32)

    def clear(self):
        """Forget old touches. Call at the start of every control step."""
        self.flags = np.zeros(3, dtype=np.float32)

    def poll(self):
        """Look at the current contacts. Call after every physics step."""
        if self.system.GetNumContacts() == 0:
            return   # the common case: nothing is touching, nothing to do

        self.collector.points = []
        self.system.GetContactContainer().ReportAllContacts(self.collector)
        if not self.collector.points:
            return

        # World frame -> chassis frame.
        chassis_position = self.vehicle.position()
        chassis_rotation = self.vehicle.rotation_matrix()
        for world_point in self.collector.points:
            local = chassis_rotation.T @ (np.array(world_point) - chassis_position)
            if local[0] > BUMPER_X_MAX_M - FRONT_ZONE_DEPTH_M:
                self.flags[CONTACT_FRONT] = 1.0
            elif local[1] > 0.0:
                self.flags[CONTACT_LEFT] = 1.0
            else:
                self.flags[CONTACT_RIGHT] = 1.0

    def read(self):
        """Return the contact observation gathered since the last clear()."""
        return self.flags.copy()
