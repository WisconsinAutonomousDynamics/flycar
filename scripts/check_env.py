"""Check that the Fly Car development environment works.

Run it first, any time something looks broken:

    python scripts/check_env.py

It tries to import each Chrono module we use and prints what it found.

* ``pychrono`` (core) and ``pychrono.vehicle`` are REQUIRED. Nothing runs
  without them, and the script exits with an error if either is missing.
* ``pychrono.irrlicht`` (3D window) and ``pychrono.sensor`` (GPU cameras) are
  OPTIONAL. Without Irrlicht you cannot open the 3D view. Without the sensor
  module the eyes use the CPU ``raycast`` backend instead.
"""

import importlib
import json
import os
import platform
import shutil
import sys
from pathlib import Path

# (module name, required?, what we use it for)
MODULES = [
    ("pychrono", True, "physics core"),
    ("pychrono.vehicle", True, "the car"),
    ("pychrono.irrlicht", False, "3D window"),
    ("pychrono.sensor", False, "GPU camera eyes"),
]

# Other packages from environment.yml.
PYTHON_PACKAGES = ["numpy", "yaml", "gymnasium", "matplotlib", "pandas", "pyarrow", "pytest"]


def try_import(name):
    """Return (module, None) on success or (None, error text) on failure."""
    try:
        return importlib.import_module(name), None
    except Exception as error:  # ImportError, or an OSError from a missing DLL
        return None, f"{type(error).__name__}: {error}"


def chrono_version(chrono):
    """Find the Chrono version. Where it is stored differs between releases."""
    # Some releases expose it on the module.
    for attribute in ("CHRONO_VERSION", "__version__"):
        version = getattr(chrono, attribute, None)
        if version:
            return str(version)

    # PyChrono 10 does not, but conda keeps a record of every package it
    # installed in <environment>/conda-meta/<name>-<version>-<build>.json.
    for record in sorted(Path(sys.prefix, "conda-meta").glob("pychrono-*.json")):
        with open(record, "r", encoding="utf-8") as handle:
            return str(json.load(handle).get("version", "unknown"))
    return "unknown"


def check_activation():
    """Windows only: make sure the conda environment is activated.

    PyChrono's DLLs live in <environment>\\Library\\bin, which `conda activate`
    puts on PATH. Without it, `import pychrono` does not raise an error, it
    hangs forever. So we check BEFORE importing, say so, and patch PATH for
    this one run so the rest of the report is still useful.
    """
    if os.name != "nt":
        return
    library_bin = os.path.join(sys.prefix, "Library", "bin")
    path_entries = os.environ.get("PATH", "").lower().split(os.pathsep)
    if os.path.isdir(library_bin) and library_bin.lower() not in path_entries:
        print("WARNING: this conda environment is not activated.")
        print("  Without activation, 'import pychrono' hangs forever with no error message.")
        print("  Fix: run 'conda activate flycar' first, then run python.")
        print("  (Continuing with a temporary fix for this check only.)")
        print()
        os.environ["PATH"] = library_bin + os.pathsep + os.environ.get("PATH", "")


def main():
    print(f"Python   {platform.python_version()}  ({sys.executable})", flush=True)
    print(f"Platform {platform.platform()}", flush=True)
    print(flush=True)
    check_activation()

    all_required_ok = True
    loaded = {}

    print("Chrono modules:")
    for name, required, purpose in MODULES:
        module, error = try_import(name)
        loaded[name] = module
        kind = "required" if required else "optional"
        if module is not None:
            print(f"  [ OK ]  {name:<18} ({kind}: {purpose})")
        else:
            print(f"  [FAIL]  {name:<18} ({kind}: {purpose})")
            print(f"            {error}")
            if required:
                all_required_ok = False

    if loaded["pychrono"] is not None:
        print()
        print(f"Chrono version: {chrono_version(loaded['pychrono'])}")

    print()
    print("Other packages:")
    for name in PYTHON_PACKAGES:
        module, error = try_import(name)
        if module is not None:
            print(f"  [ OK ]  {name:<12} {getattr(module, '__version__', '')}")
        else:
            print(f"  [FAIL]  {name:<12} {error}")
            all_required_ok = False

    # The sensor module can import fine and still be unusable: it needs an
    # NVIDIA GPU with OptiX at run time. Give an honest hint about that.
    print()
    has_nvidia = shutil.which("nvidia-smi") is not None
    if loaded["pychrono.sensor"] is None:
        print("Eyes: pychrono.sensor is missing -> the 'raycast' (CPU) backend will be used.")
    elif not has_nvidia:
        print("Eyes: pychrono.sensor imports, but no NVIDIA GPU was found (no nvidia-smi)")
        print("      -> the 'raycast' (CPU) backend will be used.")
    else:
        print("Eyes: pychrono.sensor imports and an NVIDIA GPU is present")
        print("      -> the 'sensor' backend will be tried first.")

    print()
    if all_required_ok:
        print("RESULT: OK. The required modules work.")
        return 0
    print("RESULT: FAILED. A required module is missing. See README.md, 'Setup'.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
