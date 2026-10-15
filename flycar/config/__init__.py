"""Scenario settings, split by the team that owns them.

Every setting a scenario file can contain is declared in exactly one module
here, next to the checks for it:

    flycar/config/core.py      episode, timing, contact, termination
    flycar/config/vehicle.py   vehicle, spawn, terrain
    flycar/config/world.py     arena, lighting, lights, obstacles, motion
    flycar/config/vision.py    eyes

**To add a setting, edit only your own team's module.** This file collects
them; ``flycar/env/scenario.py`` uses the collection to load scenario files.

Each team module exports

* ``SECTIONS``: section name -> default values. A list section (like
  ``lights``) has an empty list here, and the defaults for ONE entry of it in
* ``ITEM_DEFAULTS``: section name -> default values for one list entry.
* ``validate(scenario, where)``: raises ``ScenarioError`` if that team's
  sections contain something that cannot work.

Nothing in this package may import pychrono (the tools and tests read
scenarios too).

This file itself is frozen: only the project leads change it.
"""


class ScenarioError(ValueError):
    """Raised when a scenario file contains something we do not understand."""


def require_defaults(values, defaults, section, where):
    """Reject any setting in ``values`` that differs from its default.

    For settings that are reserved for a planned feature but not implemented
    yet: the default (today's behaviour) is accepted, anything else is an
    error rather than being silently ignored.
    """
    for key, default in defaults.items():
        if values[key] != default:
            raise ScenarioError(
                f"{where}: {section}.{key}: {values[key]} is not implemented yet "
                f"(only the default, {default}, works today)."
            )


# Imported AFTER ScenarioError exists, because every team module imports it.
from flycar.config import core, vehicle, vision, world  # noqa: E402

# The team modules, in a fixed order. Validators run in this order, which is
# the order the checks ran in before they were split up (timing, eyes, lights
# and obstacles), so a file with several mistakes reports the same one first.
TEAM_MODULES = [core, vision, world, vehicle]

# Top-level settings that belong to no section.
TOP_LEVEL_DEFAULTS = {
    "name": "unnamed",
    "description": "",
}


def _collect(modules):
    """Merge the team modules into (DEFAULTS, ITEM_DEFAULTS, VALIDATORS).

    Raises at import time if two modules declare the same section, so a clash
    between teams shows up immediately instead of one silently winning.
    """
    defaults = dict(TOP_LEVEL_DEFAULTS)
    item_defaults = {}
    validators = []
    owner = {name: "flycar.config" for name in TOP_LEVEL_DEFAULTS}

    for module in modules:
        for section, values in module.SECTIONS.items():
            if section in owner:
                raise ImportError(
                    f"Scenario section '{section}' is declared twice: in {owner[section]} and in {module.__name__}."
                )
            owner[section] = module.__name__
            defaults[section] = values

        for section, values in getattr(module, "ITEM_DEFAULTS", {}).items():
            if section not in module.SECTIONS or not isinstance(module.SECTIONS[section], list):
                raise ImportError(
                    f"{module.__name__} gives ITEM_DEFAULTS for '{section}', "
                    "which is not one of its own list sections."
                )
            item_defaults[section] = values

        validators.append(module.validate)

    return defaults, item_defaults, validators


# Every setting a scenario can contain, with its default value.
DEFAULTS, ITEM_DEFAULTS, VALIDATORS = _collect(TEAM_MODULES)
