"""Fly Car: a fly-ready vehicle environment built on PyChrono."""

import os
import sys

# Windows only: guard against a silent hang.
#
# PyChrono's DLLs live in <environment>\Library\bin. `conda activate flycar`
# puts that folder on PATH. If Python is started WITHOUT activating (for
# example by pointing an editor straight at python.exe), `import pychrono`
# does not fail with an error: it hangs forever. Adding the folder ourselves
# makes everything that imports flycar work either way.
if os.name == "nt":
    _library_bin = os.path.join(sys.prefix, "Library", "bin")
    _path_entries = os.environ.get("PATH", "").lower().split(os.pathsep)
    if os.path.isdir(_library_bin) and _library_bin.lower() not in _path_entries:
        os.environ["PATH"] = _library_bin + os.pathsep + os.environ.get("PATH", "")
