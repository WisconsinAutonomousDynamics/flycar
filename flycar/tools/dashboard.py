"""A live view of a run: map, compound eyes, control traces, camera images.

    python run.py --scenario scenarios/light_seek.yaml --driver braitenberg --dashboard

It is a matplotlib window, so it works on any machine that can show a plot.
It is entirely optional: without ``--dashboard`` nothing here is even imported,
which is what lets runs work on a headless machine.

The dashboard also forwards key presses to drivers that want them (the
keyboard driver), so it doubles as the "steering wheel" window.

Why the code talks about "blitting"
-----------------------------------
Redrawing a whole matplotlib figure takes about 0.1 s, far too slow for a live
view. Almost everything in the figure never changes, though (axes, labels, the
arena). So we draw that static background once, keep a copy of it, and on each
update only paste the copy back and draw the few moving things on top. That is
called blitting, and it is roughly ten times faster.
"""

from collections import deque

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np

from flycar.env.spaces import BRAKE, STEERING, THROTTLE
from flycar.tools.plotting import (
    COLOR_BRAKE, COLOR_PATH, COLOR_STEERING, COLOR_THROTTLE,
    car_outline, draw_arena, style_axis,
)


class Dashboard:
    """Live matplotlib window.

    Parameters
    ----------
    world : dict
        ``info["world"]`` from ``env.reset()``.
    eye_shape : (rows, cols)
        Resolution of one eye.
    show_cameras : bool
        Add two panels for the raw camera images (``sensor`` eye backend only).
    driver : object or None
        If it has ``on_key_press`` / ``on_key_release`` methods, key events from
        this window are forwarded to it.
    refresh_every : int
        Redraw once every this many control steps.
    history_s : float
        How many seconds of control history the trace plot shows.
    """

    def __init__(self, world, eye_shape, show_cameras=False, driver=None, refresh_every=5, history_s=10.0):
        self.refresh_every = int(refresh_every)
        self.history_s = float(history_s)
        self.closed = False
        self.updates = 0

        # Matplotlib uses keys such as "s" (save) and the arrows (history) as
        # shortcuts. Turn those off so they can be used for driving instead.
        for name in list(plt.rcParams):
            if name.startswith("keymap."):
                plt.rcParams[name] = []

        plt.ion()   # interactive mode: drawing does not block the program
        self.figure = plt.figure(figsize=(11, 6.5), dpi=80, constrained_layout=True)
        self.figure.canvas.manager.set_window_title("Fly Car dashboard")
        grid = self.figure.add_gridspec(2, 4, height_ratios=[1.5, 1.0])

        # ---- Map ------------------------------------------------------------
        self.ax_map = self.figure.add_subplot(grid[0, 0:2])
        draw_arena(self.ax_map, world)
        self.ax_map.set_title("Arena (ground truth)", fontsize=10, loc="left")
        (self.path_line,) = self.ax_map.plot([], [], color=COLOR_PATH, linewidth=2)
        (self.car_line,) = self.ax_map.plot([], [], color="#0b0b0b", linewidth=1.5)
        self.path_x = []
        self.path_y = []

        # ---- Compound eyes --------------------------------------------------
        blank = np.zeros(eye_shape, dtype=np.float32)
        self.ax_left = self.figure.add_subplot(grid[0, 2])
        self.ax_right = self.figure.add_subplot(grid[0, 3])
        self.image_left = self.ax_left.imshow(blank, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        self.image_right = self.ax_right.imshow(blank, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        self.ax_left.set_title("Left eye", fontsize=10, loc="left")
        self.ax_right.set_title("Right eye", fontsize=10, loc="left")
        for axis in (self.ax_left, self.ax_right):
            axis.set_xticks([])
            axis.set_yticks([])

        # ---- Control traces -------------------------------------------------
        # The x axis is "seconds before now", so the axis itself never changes
        # and only the three lines need redrawing.
        trace_columns = grid[1, 0:2] if show_cameras else grid[1, :]
        self.ax_controls = self.figure.add_subplot(trace_columns)
        self.times = deque()
        self.traces = {"throttle": deque(), "steering": deque(), "brake": deque()}
        (self.line_throttle,) = self.ax_controls.plot([], [], color=COLOR_THROTTLE, linewidth=2, label="throttle")
        (self.line_steering,) = self.ax_controls.plot([], [], color=COLOR_STEERING, linewidth=2, label="steering (+ = left)")
        (self.line_brake,) = self.ax_controls.plot([], [], color=COLOR_BRAKE, linewidth=2, label="brake")
        self.ax_controls.set_xlim(-self.history_s, 0.0)
        self.ax_controls.set_ylim(-1.05, 1.05)
        self.ax_controls.set_xlabel("seconds before now", fontsize=8)
        self.ax_controls.set_ylabel("fraction of full scale", fontsize=8)
        self.ax_controls.set_title("Controls applied to the car", fontsize=10, loc="left")
        # The legend sits above the plot, where no trace can run through it.
        self.ax_controls.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), fontsize=8, frameon=False, ncol=3)
        style_axis(self.ax_controls)

        # ---- Raw camera images (sensor backend only) ------------------------
        self.camera_axes = []
        self.camera_images = []
        if show_cameras:
            for column, title in ((2, "Left camera (raw)"), (3, "Right camera (raw)")):
                axis = self.figure.add_subplot(grid[1, column])
                axis.set_title(title, fontsize=10, loc="left")
                axis.set_xticks([])
                axis.set_yticks([])
                self.camera_axes.append(axis)
                self.camera_images.append(None)   # created when the first frame arrives

        # ---- Blitting -------------------------------------------------------
        # The artists that move. "Animated" artists are skipped by a normal
        # draw, so the saved background contains everything except them.
        self.moving = [
            self.path_line, self.car_line, self.image_left, self.image_right,
            self.line_throttle, self.line_steering, self.line_brake,
        ]
        self.use_blit = bool(getattr(self.figure.canvas, "supports_blit", False))
        self.background = None
        if self.use_blit:
            for artist in self.moving:
                artist.set_animated(True)

        # ---- Window events --------------------------------------------------
        self.figure.canvas.mpl_connect("draw_event", self._on_full_draw)
        self.figure.canvas.mpl_connect("close_event", self._on_close)
        if driver is not None and hasattr(driver, "on_key_press"):
            self.figure.canvas.mpl_connect("key_press_event", lambda event: driver.on_key_press(event.key))
            self.figure.canvas.mpl_connect("key_release_event", lambda event: driver.on_key_release(event.key))

        self.figure.canvas.draw()   # first full draw (also fills self.background)

    def _on_close(self, event):
        self.closed = True

    def _on_full_draw(self, event):
        """After every full draw (the first one, or a window resize): save the background."""
        if not self.use_blit:
            return
        self.background = self.figure.canvas.copy_from_bbox(self.figure.bbox)
        for artist in self.moving:
            self.figure.draw_artist(artist)

    def update(self, observation, info):
        """Feed in one control step. Returns False once the window has been closed."""
        if self.closed:
            return False

        # Always record the data, even on steps where we do not redraw.
        time = info["time"]
        applied = info["applied_action"]
        self.times.append(time)
        self.traces["throttle"].append(float(applied[THROTTLE]))
        self.traces["steering"].append(float(applied[STEERING]))
        self.traces["brake"].append(float(applied[BRAKE]))
        while self.times and self.times[0] < time - self.history_s:
            self.times.popleft()
            for trace in self.traces.values():
                trace.popleft()

        x, y, yaw = info["pose"]
        self.path_x.append(x)
        self.path_y.append(y)

        self.updates += 1
        if self.updates % self.refresh_every != 0:
            return True

        # ---- Update the moving artists --------------------------------------
        self.path_line.set_data(self.path_x, self.path_y)
        self.car_line.set_data(*car_outline(x, y, yaw))

        self.image_left.set_data(observation["eye_left"])
        self.image_right.set_data(observation["eye_right"])

        seconds_ago = [t - time for t in self.times]
        self.line_throttle.set_data(seconds_ago, list(self.traces["throttle"]))
        self.line_steering.set_data(seconds_ago, list(self.traces["steering"]))
        self.line_brake.set_data(seconds_ago, list(self.traces["brake"]))

        for index, key in enumerate(("camera_left", "camera_right")[: len(self.camera_axes)]):
            frame = info.get(key)
            if frame is None:
                continue
            if self.camera_images[index] is None:
                image = self.camera_axes[index].imshow(frame, animated=self.use_blit)
                self.camera_images[index] = image
                self.moving.append(image)
            else:
                self.camera_images[index].set_data(frame)

        # ---- Redraw ---------------------------------------------------------
        canvas = self.figure.canvas
        if self.use_blit and self.background is not None:
            canvas.restore_region(self.background)      # paste the static background
            for artist in self.moving:
                self.figure.draw_artist(artist)         # draw the moving parts on top
            canvas.blit(self.figure.bbox)               # show the result
        else:
            canvas.draw_idle()                          # slow path: redraw everything
        canvas.flush_events()   # lets the window process clicks and key presses
        return not self.closed

    def close(self):
        plt.close(self.figure)
        self.closed = True
