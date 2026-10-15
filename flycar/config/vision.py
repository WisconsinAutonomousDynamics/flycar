"""Scenario settings owned by the vision team.

Section: ``eyes``.

Add a setting by adding a key (with a comment saying what it does and its
units) to the section below, and a check for it in ``validate``. Defaults must
reproduce the current behaviour exactly.
"""

from flycar.config import ScenarioError, require_defaults

SECTIONS = {
    "eyes": {
        "backend": "auto",         # auto | sensor | raycast
        "resolution": [20, 20],    # rows, columns of each compound eye
        "fov_h_deg": 120.0,        # horizontal field of view of one eye
        "fov_v_deg": 60.0,         # vertical field of view of one eye
        "yaw_offset_deg": 50.0,    # each eye is turned outwards by this angle
        "update_period_s": 0.02,   # how often the eyes take a new picture
        "max_range_m": 80.0,       # rays that hit nothing within this distance see sky
        "shadows": True,           # raycast backend: test whether each surface point can see each light
        "camera_width_px": 240,    # sensor backend: raw image width (height follows from the FOVs)
        "mount_m": [1.0, 0.3, 0.75],   # eye position on the chassis: forward, sideways (mirrored), up
        "noise_std": 0.0,          # (planned) standard deviation of pixel noise; 0 = no noise
        "motion_blur": False,      # (planned) blur the picture while the car turns
    },
}

# Settings reserved for planned features: only these values work today.
NOT_YET_IMPLEMENTED = {
    "eyes": {"noise_std": 0.0, "motion_blur": False},
}


def validate(scenario, where):
    """Checks for the eyes section.

    The eye update period depends on ``timing`` (owned by core), so the check
    that ties them together lives here, with the setting that depends on it.
    """
    control = scenario["timing"]["control_period_s"]
    eye_period = scenario["eyes"]["update_period_s"]
    eye_steps = eye_period / control
    if abs(eye_steps - round(eye_steps)) > 1e-6 or round(eye_steps) < 1:
        raise ScenarioError(
            f"{where}: eyes.update_period_s ({eye_period}) must be a whole multiple of control_period_s ({control})."
        )

    if scenario["eyes"]["backend"] not in ("auto", "sensor", "raycast"):
        raise ScenarioError(f"{where}: eyes.backend must be auto, sensor or raycast.")

    for section, defaults in NOT_YET_IMPLEMENTED.items():
        require_defaults(scenario[section], defaults, section, where)
