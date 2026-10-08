"""Controllers. They use only the public API and never import pychrono.

``make_driver`` turns the name given to ``run.py --driver`` into a driver object.
"""

import importlib

from flycar.drivers.base import Driver

# Short names for the built-in drivers: name -> (module, class).
BUILTIN_DRIVERS = {
    "random": ("flycar.drivers.random_driver", "RandomDriver"),
    "braitenberg": ("flycar.drivers.braitenberg", "BraitenbergDriver"),
    "keyboard": ("flycar.drivers.keyboard", "KeyboardDriver"),
}


def make_driver(name):
    """Create a driver from its name.

    ``name`` is either a built-in name (``random``, ``braitenberg``,
    ``keyboard``) or ``package.module:ClassName`` for your own driver, e.g.
    ``python run.py --driver my_drivers.wall_follower:WallFollower``.
    """
    if name in BUILTIN_DRIVERS:
        module_name, class_name = BUILTIN_DRIVERS[name]
    elif ":" in name:
        module_name, class_name = name.split(":", 1)
    else:
        known = ", ".join(sorted(BUILTIN_DRIVERS))
        raise ValueError(f"Unknown driver '{name}'. Use one of: {known}, or 'package.module:ClassName'.")

    module = importlib.import_module(module_name)
    driver = getattr(module, class_name)()

    if not isinstance(driver, Driver):
        raise TypeError(f"{class_name} is not a Driver: it needs reset() and act(observation) methods.")
    return driver
