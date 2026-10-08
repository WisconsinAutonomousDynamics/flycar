"""Look at a finished run: where the car went, what it did, what it saw.

    python -m flycar.tools.replay logs/light_seek_braitenberg_seed0.parquet
    python -m flycar.tools.replay LOG --eyes-at 1 5 10         # eye frames at these times [s]
    python -m flycar.tools.replay LOG --save replay.png        # write a picture, open no window

The figure has three parts:

* the trajectory drawn over the arena map,
* the control signals and the speed over time,
* the two eye images at a few chosen moments.
"""

import argparse

import numpy as np

from flycar.tools.logger import load_log
from flycar.tools.plotting import (
    COLOR_BRAKE, COLOR_PATH, COLOR_STEERING, COLOR_TEXT_MUTED, COLOR_THROTTLE,
    car_outline, draw_arena, style_axis,
)

COLOR_CONTACT = "#e34948"


def eye_frame_at(table, time, eye_shape):
    """Return (left, right, actual_time) for the stored eye frame closest to ``time``."""
    stored = table[table["eye_left"].notna()]
    if len(stored) == 0:
        return None
    nearest = (stored["time"] - time).abs().idxmin()
    row = stored.loc[nearest]
    left = np.asarray(row["eye_left"], dtype=np.float32).reshape(eye_shape)
    right = np.asarray(row["eye_right"], dtype=np.float32).reshape(eye_shape)
    return left, right, float(row["time"])


def make_figure(table, world, details, eye_times=None):
    """Build the replay figure and return it."""
    import matplotlib.pyplot as plt

    eye_shape = tuple(details.get("eye_shape", (20, 20)))
    end_time = float(table["time"].iloc[-1])
    if not eye_times:
        # Four moments spread over the run.
        eye_times = [end_time * fraction for fraction in (0.05, 0.35, 0.65, 0.95)]

    figure = plt.figure(figsize=(13, 8.5), constrained_layout=True)
    grid = figure.add_gridspec(3, 2, width_ratios=[1.15, 1.0], height_ratios=[1.0, 1.0, 0.95])

    # ---- Trajectory over the arena map --------------------------------------
    ax_map = figure.add_subplot(grid[0:2, 0])
    draw_arena(ax_map, world)
    ax_map.plot(table["x"], table["y"], color=COLOR_PATH, linewidth=2, label="path")

    touching = table[(table["contact_left"] + table["contact_front"] + table["contact_right"]) > 0]
    if len(touching) > 0:
        ax_map.plot(touching["x"], touching["y"], linestyle="none", marker="x", markersize=8,
                    color=COLOR_CONTACT, label="contact")

    # The car's outline at the start and at the end, labelled directly.
    for row, label in ((table.iloc[0], "start"), (table.iloc[-1], "end")):
        xs, ys = car_outline(row["x"], row["y"], row["yaw"])
        ax_map.plot(xs, ys, color="#0b0b0b", linewidth=1.2)
        ax_map.annotate(label, (row["x"], row["y"]), textcoords="offset points", xytext=(8, -14),
                        fontsize=8, color=COLOR_TEXT_MUTED)
    ax_map.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), fontsize=8, frameon=False, ncol=2)

    outcome = "reached the target" if bool(table["success"].iloc[-1]) else "did not reach the target"
    if world["target"] is None:
        outcome = "no target in this scenario"
    title = f"{details.get('scenario', 'run')} / {details.get('driver', '?')} / seed {details.get('seed', '?')}"
    ax_map.set_title(f"{title}\n{end_time:.1f} s, {outcome}", fontsize=10, loc="left")

    # ---- Controls over time -------------------------------------------------
    # All three are fractions of full scale, so they share one axis.
    ax_controls = figure.add_subplot(grid[0, 1])
    ax_controls.plot(table["time"], table["applied_throttle"], color=COLOR_THROTTLE, linewidth=2, label="throttle")
    ax_controls.plot(table["time"], table["applied_steering"], color=COLOR_STEERING, linewidth=2, label="steering (+ = left)")
    ax_controls.plot(table["time"], table["applied_brake"], color=COLOR_BRAKE, linewidth=2, label="brake")
    ax_controls.set_ylim(-1.05, 1.05)
    ax_controls.set_ylabel("fraction of full scale", fontsize=8)
    ax_controls.set_title("Controls applied to the car", fontsize=10, loc="left")
    # The legend sits above the plot, where no trace can run through it.
    ax_controls.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), fontsize=8, frameon=False, ncol=3)
    style_axis(ax_controls)

    # ---- Speed over time ----------------------------------------------------
    ax_speed = figure.add_subplot(grid[1, 1], sharex=ax_controls)
    ax_speed.plot(table["time"], table["speed"], color=COLOR_PATH, linewidth=2)
    ax_speed.set_ylabel("m/s", fontsize=8)
    ax_speed.set_xlabel("time [s]", fontsize=8)
    ax_speed.set_title("Forward speed", fontsize=10, loc="left")
    style_axis(ax_speed)

    # Thin vertical lines show when the eye frames below were taken.
    for time in eye_times:
        for axis in (ax_controls, ax_speed):
            axis.axvline(time, color=COLOR_TEXT_MUTED, linewidth=0.8, linestyle=":")

    # ---- Eye frames ---------------------------------------------------------
    eye_grid = grid[2, :].subgridspec(1, len(eye_times), wspace=0.05)
    for index, time in enumerate(eye_times):
        ax_eye = figure.add_subplot(eye_grid[0, index])
        frame = eye_frame_at(table, time, eye_shape)
        if frame is None:
            ax_eye.text(0.5, 0.5, "no eye frames in this log", ha="center", va="center", fontsize=8)
            ax_eye.set_axis_off()
            continue
        left, right, actual_time = frame
        # Left and right eye side by side, with a thin gap between them.
        gap = np.full((eye_shape[0], 1), np.nan, dtype=np.float32)
        ax_eye.imshow(np.hstack([left, gap, right]), cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        ax_eye.set_title(f"eyes at t = {actual_time:.1f} s   (left | right)", fontsize=9, loc="left")
        ax_eye.set_xticks([])
        ax_eye.set_yticks([])
        ax_eye.set_facecolor("#c8443c")   # colour of the gap, so the two eyes read as separate
    return figure


def main():
    parser = argparse.ArgumentParser(description="Plot a Fly Car log: trajectory, controls and eye frames.")
    parser.add_argument("log", help="Path to a .parquet log written by run.py --log")
    parser.add_argument("--eyes-at", type=float, nargs="+", metavar="T",
                        help="Times [s] at which to show the eye images (default: four spread over the run)")
    parser.add_argument("--save", metavar="FILE", help="Write the figure to this image file instead of opening a window")
    args = parser.parse_args()

    if args.save:
        import matplotlib
        matplotlib.use("Agg")   # draw without a window
    import matplotlib.pyplot as plt

    table, world, details = load_log(args.log)
    figure = make_figure(table, world, details, args.eyes_at)

    if args.save:
        figure.savefig(args.save, dpi=120)
        print(f"Saved {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
