"""Leaderboard: compare every driver on every scenario.

Used by the batch runner::

    python -m flycar.tools.batch --scenario all --driver all --seeds 0-19 \\
        --workers 4 --out outputs/ --figure outputs/leaderboard.png

``all`` means every scenario file in ``scenarios/`` and every driver found by
:func:`discover_drivers`: the built-in drivers, plus the teams' drivers in
``flycar/drivers/contrib/``.

Everything here works on plain tables (pandas) and plain names. It needs
neither pychrono nor a window, so it is quick to test.
"""

import importlib
import pkgutil
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

from flycar.drivers import BUILTIN_DRIVERS
from flycar.tools import plotting

REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIO_DIR = REPO_ROOT / "scenarios"
CONTRIB_PACKAGE = "flycar.drivers.contrib"

# Text colour inside the cells: dark on light cells, white on the darkest blue.
COLOR_CELL_TEXT = "#1f1e1c"
WHITE_TEXT_FROM_RATE = 0.95
# Colour of a 0 % cell: pale, but not white, so that it does not look like a gap.
COLOR_LOW_RATE = "#eaf1fb"


# ----------------------------------------------------------------------------
# Finding what to run
# ----------------------------------------------------------------------------


def _looks_like_a_driver(cls):
    """True for a class with ``reset()`` and ``act()`` (the Driver protocol of contract 8)."""
    return callable(getattr(cls, "reset", None)) and callable(getattr(cls, "act", None))


def _needs_a_person(cls):
    """True for drivers that read the keyboard: a batch run has nobody to press keys."""
    return bool(getattr(cls, "needs_key_events", False))


def _package_is_missing(error, package_name):
    """True if a ModuleNotFoundError is about the package itself, not about something it imports."""
    missing = error.name or ""
    return package_name == missing or package_name.startswith(missing + ".")


def discover_drivers(contrib_package=CONTRIB_PACKAGE, include_builtin=True):
    """Find every driver the leaderboard can run.

    Returns ``(names, warnings)``. ``names`` can be passed to ``make_driver()``:
    the built-in short names (``braitenberg``) and, for the teams' drivers,
    ``package.module:ClassName``. ``warnings`` is a list of sentences about
    drivers that were skipped, so one broken driver never stops the others.

    * Built-in drivers that need a person at the keyboard are left out.
    * In ``contrib_package``, every module named ``*_driver`` must contain exactly
      one class with ``reset()`` and ``act()`` (what C2 asks of each team).
    * A missing ``contrib`` folder is fine: it just means no team drivers yet.
    """
    names = []
    warnings = []

    if include_builtin:
        for name, (module_name, class_name) in sorted(BUILTIN_DRIVERS.items()):
            cls = getattr(importlib.import_module(module_name), class_name)
            if not _needs_a_person(cls):
                names.append(name)

    try:
        package = importlib.import_module(contrib_package)
    except ModuleNotFoundError as error:
        if not _package_is_missing(error, contrib_package):
            warnings.append(f"{contrib_package}: could not be imported ({error})")
        return names, warnings
    except Exception as error:
        warnings.append(f"{contrib_package}: could not be imported ({type(error).__name__}: {error})")
        return names, warnings

    search_path = getattr(package, "__path__", None)
    if search_path is None:
        warnings.append(f"{contrib_package}: is not a package, so no drivers were looked for in it")
        return names, warnings

    for module_info in sorted(pkgutil.iter_modules(search_path), key=lambda info: info.name):
        if not module_info.name.endswith("_driver"):
            continue
        module_name = f"{contrib_package}.{module_info.name}"
        try:
            module = importlib.import_module(module_name)
        except Exception as error:
            warnings.append(f"{module_name}: could not be imported ({type(error).__name__}: {error})")
            continue

        classes = [
            value for value in vars(module).values()
            if isinstance(value, type) and value.__module__ == module_name and _looks_like_a_driver(value)
        ]
        if len(classes) != 1:
            warnings.append(f"{module_name}: expected exactly one driver class, found {len(classes)}")
            continue
        if _needs_a_person(classes[0]):
            warnings.append(f"{module_name}: {classes[0].__name__} needs a person at the keyboard, so it was skipped")
            continue
        names.append(f"{module_name}:{classes[0].__name__}")

    return names, warnings


def discover_scenarios(folder=SCENARIO_DIR, include_team=True):
    """Paths of the scenario files in ``folder`` (``*.yaml``), sorted by name.

    ``include_team=False`` leaves out the teams' scratch scenarios (``team_*.yaml``).
    """
    paths = sorted(Path(folder).glob("*.yaml"))
    if not include_team:
        paths = [path for path in paths if not path.name.startswith("team_")]
    return [str(path) for path in paths]


def expand_all(items, discovered):
    """Replace the word ``all`` in a list by ``discovered``, without repeating anything.

    ``expand_all(["x", "all", "y"], ["a", "x"])`` is ``["x", "a", "y"]``.
    """
    expanded = []
    for item in items:
        for value in (discovered if item == "all" else [item]):
            if value not in expanded:
                expanded.append(value)
    return expanded


# ----------------------------------------------------------------------------
# The table
# ----------------------------------------------------------------------------


def short_name(driver):
    """A readable label for a driver name: ``pkg.mod_driver:WorldDriver`` becomes ``WorldDriver``."""
    return driver.split(":")[-1].split(".")[-1]


def leaderboard_table(summary):
    """Success rates with one row per driver and one column per scenario.

    ``summary`` is the table from ``batch.summarize()``. Rows are ordered from the
    best driver to the worst, by mean success rate over the scenarios that have a
    target (ties keep alphabetical order). Cells for scenarios without a target,
    or without any episodes of that driver, are NaN.
    """
    grid = summary.pivot(index="driver", columns="scenario", values="success_rate")
    order = grid.mean(axis=1, skipna=True).sort_values(ascending=False, na_position="last", kind="stable").index
    return grid.loc[order]


def _grid(summary, column, index, columns, default=np.nan):
    """One column of the summary laid out like the leaderboard (same rows and columns)."""
    if column not in summary.columns:
        return pd.DataFrame(default, index=index, columns=columns)
    grid = summary.pivot(index="driver", columns="scenario", values=column)
    return grid.reindex(index=index, columns=columns)


# ----------------------------------------------------------------------------
# The figure
# ----------------------------------------------------------------------------


def make_figure(summary, title="Success rate by driver and scenario"):
    """Draw the leaderboard as a grid: drivers down, scenarios across, one cell per pair.

    Each cell shows the success rate and ``successes/episodes``, shaded from pale blue
    (0 %) to blue (100 %). Cells for a scenario without a target say ``n/a``, and
    cells of a pair that was not run say ``not run``. Crashed episodes are counted
    as failures and are mentioned in the cell (``2 err``).

    Returns a matplotlib ``Figure`` (made without pyplot, so no window is needed).
    """
    if len(summary) == 0:
        raise ValueError("The summary is empty: there is nothing to draw.")

    grid = leaderboard_table(summary)
    successes = _grid(summary, "successes", grid.index, grid.columns)
    episodes = _grid(summary, "episodes", grid.index, grid.columns)
    errors = _grid(summary, "errors", grid.index, grid.columns, default=0.0)

    labels = [short_name(driver) for driver in grid.index]
    if len(set(labels)) < len(labels):          # two drivers with the same class name
        labels = list(grid.index)

    n_rows, n_columns = grid.shape
    figure = Figure(
        figsize=(max(4.5, 1.7 * n_columns + 2.4), max(2.4, 0.85 * n_rows + 1.8)),
        layout="constrained",
    )
    ax = figure.subplots()

    colormap = LinearSegmentedColormap.from_list("flycar_success", [COLOR_LOW_RATE, plotting.COLOR_PATH])
    colormap = colormap.with_extremes(bad=plotting.COLOR_GRID)
    ax.imshow(np.ma.masked_invalid(grid.to_numpy(dtype=float)), cmap=colormap, vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(range(n_columns), labels=list(grid.columns), rotation=30, ha="right")
    ax.set_yticks(range(n_rows), labels=labels)
    ax.tick_params(length=0, labelsize=8, colors=plotting.COLOR_TEXT_MUTED)
    # White lines between the cells, instead of a grid or a frame.
    ax.set_xticks(np.arange(-0.5, n_columns, 1.0), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1.0), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.0)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=10, color=plotting.COLOR_TEXT_MUTED, loc="left")

    explained = {}
    for row in range(n_rows):
        for column in range(n_columns):
            rate = grid.iat[row, column]
            color = COLOR_CELL_TEXT
            if np.isnan(episodes.iat[row, column]):
                text = "not run"
                explained["not run"] = "not run: this driver was not run on this scenario"
                color = plotting.COLOR_TEXT_MUTED
            elif np.isnan(rate):
                text = "n/a"
                explained["n/a"] = "n/a: the scenario has no target lamp"
                color = plotting.COLOR_TEXT_MUTED
            else:
                text = f"{rate:.0%}\n{int(round(successes.iat[row, column]))}/{int(episodes.iat[row, column])}"
                if rate >= WHITE_TEXT_FROM_RATE:
                    color = "white"
            crashed = errors.iat[row, column]
            if crashed > 0:
                text += f"\n{int(crashed)} err"
                explained["err"] = "err: episodes that crashed (counted as failures)"
            ax.text(column, row, text, ha="center", va="center", fontsize=8, color=color)

    if explained:
        figure.supxlabel("    ".join(explained.values()), fontsize=7, color=plotting.COLOR_TEXT_MUTED)
    return figure


# ----------------------------------------------------------------------------
# Output files and reports
# ----------------------------------------------------------------------------


def write_figure(summary, path, dpi=150):
    """Draw the leaderboard and save it as an image (the file type follows the name). Returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    make_figure(summary).savefig(path, dpi=dpi)
    return path


def write_summary_files(summary, out_dir):
    """Write ``summary.csv`` (one row per scenario and driver) and ``leaderboard.csv``
    (the drivers x scenarios grid, best driver first). Returns both paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.csv"
    leaderboard_path = out_dir / "leaderboard.csv"
    summary.to_csv(summary_path, index=False)
    leaderboard_table(summary).to_csv(leaderboard_path)
    return summary_path, leaderboard_path


def error_report(table):
    """One line per (scenario, driver) that had crashed episodes, or an empty list.

    ``table`` is the episode table from ``batch.run_batch``. Each line says how
    many episodes crashed, the first seed that did, and the most common error,
    which is what the owning team needs to reproduce it.
    """
    if "termination_reason" not in table.columns or "error" not in table.columns:
        return []
    totals = table.groupby(["scenario", "driver"]).size()
    crashed = table[table["termination_reason"] == "error"]
    lines = []
    for (scenario, driver), group in crashed.groupby(["scenario", "driver"], sort=True):
        commonest = group["error"].value_counts().index[0]
        lines.append(
            f"{scenario} / {driver}: {len(group)} of {totals[(scenario, driver)]} episodes crashed "
            f"(first at seed {int(group['seed'].min())}); most common error: {commonest}"
        )
    return lines
