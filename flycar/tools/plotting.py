"""Plot helpers shared by the replay tool and the live dashboard.

Everything here works from the plain-data world layout (``info["world"]`` from
``env.reset()``), so no pychrono is needed to draw the arena.
"""

import math

import matplotlib.patches as patches
from matplotlib.transforms import Affine2D

# One colour per control signal, used the same way in every plot.
COLOR_THROTTLE = "#2a78d6"   # blue
COLOR_STEERING = "#eb6834"   # orange
COLOR_BRAKE = "#1baf7a"      # aqua

COLOR_PATH = "#2a78d6"
COLOR_LAMP = "#eda100"
COLOR_WALL = "#52514e"
COLOR_OBSTACLE = "#8a8983"
COLOR_GRID = "#e4e3df"
COLOR_TEXT_MUTED = "#52514e"

# Outline of the car seen from above, in the chassis frame [m] (see vehicle.py).
CAR_X_MIN, CAR_X_MAX, CAR_HALF_WIDTH = -2.2, 1.3, 0.75


def style_axis(ax):
    """Quiet axes: thin grey frame and grid, so the data stands out."""
    ax.grid(True, color=COLOR_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLOR_GRID)
    ax.tick_params(colors=COLOR_TEXT_MUTED, labelsize=8)


def draw_arena(ax, world):
    """Draw the arena seen from above: walls, obstacles and lamps."""
    size_x, size_y = world["arena"]["size_m"]

    for item in world["walls"] + world["obstacles"]:
        is_wall = item in world["walls"]
        color = COLOR_WALL if is_wall else COLOR_OBSTACLE
        if item["shape"] == "box":
            length, width = item["size_m"][0], item["size_m"][1]
            box = patches.Rectangle((-length / 2.0, -width / 2.0), length, width, color=color, linewidth=0)
            # Rotate about the box centre, then move it into place.
            box.set_transform(Affine2D().rotate(item["yaw"]).translate(item["x"], item["y"]) + ax.transData)
            ax.add_patch(box)
        else:
            radius = item["size_m"][0]
            ax.add_patch(patches.Circle((item["x"], item["y"]), radius, color=color, linewidth=0))

    for light in world["lights"]:
        ax.plot(light["x"], light["y"], marker="*", markersize=14, color=COLOR_LAMP,
                markeredgecolor="white", markeredgewidth=0.8, linestyle="none", zorder=5)
        label = "target lamp" if light["target"] else "lamp"
        ax.annotate(label, (light["x"], light["y"]), textcoords="offset points", xytext=(8, 8),
                    fontsize=8, color=COLOR_TEXT_MUTED)

    margin = 2.0
    ax.set_xlim(-size_x / 2.0 - margin, size_x / 2.0 + margin)
    ax.set_ylim(-size_y / 2.0 - margin, size_y / 2.0 + margin)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]", fontsize=8)
    ax.set_ylabel("y [m]", fontsize=8)
    style_axis(ax)


def car_outline(x, y, yaw):
    """Corner coordinates (xs, ys) of the car's outline at a pose, as a closed loop."""
    corners = [
        (CAR_X_MAX, CAR_HALF_WIDTH),
        (CAR_X_MAX, -CAR_HALF_WIDTH),
        (CAR_X_MIN, -CAR_HALF_WIDTH),
        (CAR_X_MIN, CAR_HALF_WIDTH),
        (CAR_X_MAX, CAR_HALF_WIDTH),
    ]
    xs = [x + cx * math.cos(yaw) - cy * math.sin(yaw) for cx, cy in corners]
    ys = [y + cx * math.sin(yaw) + cy * math.cos(yaw) for cx, cy in corners]
    return xs, ys
