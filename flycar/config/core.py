"""Scenario settings owned by the core & body sensing team.

Sections: ``episode``, ``timing``, ``contact``, ``termination``.

Add a setting by adding a key (with a comment saying what it does and its
units) to a section below, and a check for it in ``validate``. Defaults must
reproduce the current behaviour exactly.
"""
import time, math
from flycar.config import ScenarioError, require_defaults

SECTIONS = {
    "episode": {
        "max_time_s": 30.0,        # the episode is truncated after this much simulated time
        "success_radius_m": 2.0,   # reaching the target light within this distance ends the episode
    },
    "timing": {
        "control_period_s": 0.01,  # one env.step() = this much simulated time
        "physics_step_s": 0.001,
        "profile": False,  # enable per-phase wall-clock profiling   # Chrono integration step (must divide the control period)
    },
    "contact": {
        "mode": "flag",          # flag: 0/1 touch flags. (Planned: force, in newtons.)
    },
    # Extra reasons to end an episode early. All off: today an episode ends
    # only on success or when max_time_s runs out. (Planned, not implemented.)
    "termination": {
        "flipped": False,        # end the episode if the car rolls over
        "stuck_s": None,         # end it if the car has not moved for this long [s]; null = never
        "left_arena": False,     # end it if the car leaves the arena (no walls)
    },
}

# Settings reserved for planned features: only these values work today.
NOT_YET_IMPLEMENTED = {
    "contact": {"mode": "flag"},
    "termination": {"flipped": False, "stuck_s": None, "left_arena": False},
}


def validate(scenario, where):
    """Catch the mistakes that would otherwise fail deep inside Chrono."""
    if type(scenario["timing"]["profile"]) is not bool:
        raise ScenarioError(f"{where}: timing.profile must be a boolean.")
    control = scenario["timing"]["control_period_s"]
    physics = scenario["timing"]["physics_step_s"]
    substeps = control / physics
    if physics <= 0 or control <= 0:
        raise ScenarioError(f"{where}: timing values must be positive.")
    if abs(substeps - round(substeps)) > 1e-6 or round(substeps) < 1:
        raise ScenarioError(
            f"{where}: control_period_s ({control}) must be a whole multiple of physics_step_s ({physics})."
        )

    for section, defaults in NOT_YET_IMPLEMENTED.items():
        require_defaults(scenario[section], defaults, section, where)
