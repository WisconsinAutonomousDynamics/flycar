"""Scenario files: load a YAML experiment description and fill in defaults.

A scenario is a plain YAML file (see ``scenarios/``). It describes the arena,
the obstacles, the lights, where the car spawns and how long an episode lasts.
Running a new experiment means writing a new YAML file, not editing code.

Every available setting, its default and the checks on it live in
``flycar/config/``, one module per team:

    flycar/config/core.py      episode, timing, contact, termination
    flycar/config/vehicle.py   vehicle, spawn, terrain
    flycar/config/world.py     arena, lighting, lights, obstacles, motion
    flycar/config/vision.py    eyes

This file only loads a scenario and fills in those defaults.

Randomization rule
------------------
Anywhere a scenario asks for a position or an angle (``x``, ``y``, ``z``,
``yaw_deg``) you may write either

* a single number  -> that exact value every episode, or
* ``[min, max]``   -> a new uniform random value on every ``reset()``.

This file must NOT import pychrono (the tools and tests read scenarios too).
"""

import copy
from pathlib import Path

import yaml

from flycar import config
from flycar.config import ScenarioError  # noqa: F401  (re-exported: "from flycar.env.scenario import ScenarioError" keeps working)


def _merge(defaults, overrides, where):
    """Return ``defaults`` updated with ``overrides``, checking the key names.

    A misspelled key (say ``max_tim_s``) would otherwise be silently ignored
    and the default used instead, which is a miserable bug to track down.
    """
    result = copy.deepcopy(defaults)
    for key, value in overrides.items():
        if key not in defaults:
            known = ", ".join(sorted(defaults))
            raise ScenarioError(f"Unknown setting '{key}' in {where}. Known settings: {known}")
        if isinstance(defaults[key], dict):
            if not isinstance(value, dict):
                raise ScenarioError(f"'{key}' in {where} must be a mapping (key: value pairs).")
            result[key] = _merge(defaults[key], value, f"{where}.{key}")
        else:
            result[key] = value
    return result


def load_scenario(source):
    """Load a scenario and fill in every default.

    ``source`` is a path to a YAML file, or an already-loaded dictionary.
    Returns a plain nested dictionary with every key from
    ``flycar.config.DEFAULTS`` present.
    """
    if isinstance(source, dict):
        raw = source
        where = "scenario"
    else:
        path = Path(source)
        if not path.exists():
            raise ScenarioError(f"Scenario file not found: {path}")
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        where = path.name
        raw.setdefault("name", path.stem)

    scenario = _merge(config.DEFAULTS, raw, where)

    # The list entries (lights, obstacles) have their own defaults.
    for section, item_defaults in config.ITEM_DEFAULTS.items():
        scenario[section] = [
            _merge(item_defaults, item, f"{where}.{section}[{i}]")
            for i, item in enumerate(scenario[section] or [])
        ]

    # Each team checks its own sections (see flycar/config/).
    for validate in config.VALIDATORS:
        validate(scenario, where)
    return scenario


def sample(value, rng):
    """Turn a scenario value into a number.

    A number is returned unchanged. A ``[min, max]`` pair gives a uniform
    random number drawn from ``rng`` (a ``numpy.random.Generator``).
    """
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ScenarioError(f"A random range must be [min, max], got {value}.")
        low, high = float(value[0]), float(value[1])
        return float(rng.uniform(low, high))
    return float(value)
