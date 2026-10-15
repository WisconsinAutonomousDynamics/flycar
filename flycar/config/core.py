"""Scenario settings owned by the core & body sensing team.

Sections: ``episode``, ``timing``, ``contact``.

Add a setting by adding a key (with a comment saying what it does and its
units) to a section below, and a check for it in ``validate``. Defaults must
reproduce the current behaviour exactly.
"""

from flycar.config import ScenarioError

SECTIONS = {
    "episode": {
        "max_time_s": 30.0,        # the episode is truncated after this much simulated time
        "success_radius_m": 2.0,   # reaching the target light within this distance ends the episode
    },
    "timing": {
        "control_period_s": 0.01,  # one env.step() = this much simulated time
        "physics_step_s": 0.001,   # Chrono integration step (must divide the control period)
    },
    "contact": {},
}


def validate(scenario, where):
    """Catch the mistakes that would otherwise fail deep inside Chrono."""
    control = scenario["timing"]["control_period_s"]
    physics = scenario["timing"]["physics_step_s"]
    substeps = control / physics
    if physics <= 0 or control <= 0:
        raise ScenarioError(f"{where}: timing values must be positive.")
    if abs(substeps - round(substeps)) > 1e-6 or round(substeps) < 1:
        raise ScenarioError(
            f"{where}: control_period_s ({control}) must be a whole multiple of physics_step_s ({physics})."
        )
