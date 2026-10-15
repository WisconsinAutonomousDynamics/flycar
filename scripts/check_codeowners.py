"""Check that .github/CODEOWNERS covers every file in the repository.

    python scripts/check_codeowners.py

For every file git knows about (committed, plus new files that are not
ignored), find the CODEOWNERS lines that match it:

* no line         -> a GAP: nobody reviews changes to that file. Error.
* exactly one     -> fine.
* more than one   -> an OVERRIDE. GitHub uses the last matching line. That is
                     only allowed when the last line is in the frozen block at
                     the end of the file (a frozen file inside a team folder,
                     like scenarios/light_seek.yaml). Any other overlap is an
                     error, because it hides which team really owns the file.

Exit code 0 when there are no errors, 1 otherwise.

Only the pattern forms CODEOWNERS in this repository uses are supported:
``/path/to/file`` and ``/path/to/folder/`` (both anchored at the repository
root). Anything else is reported as an error, so it cannot be silently
misread.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CODEOWNERS = REPO_ROOT / ".github" / "CODEOWNERS"

# The comment line that starts the frozen block in CODEOWNERS.
FROZEN_MARKER = "# ---- Frozen"


def read_rules(path):
    """Return a list of (line_number, pattern, owners, frozen) from CODEOWNERS."""
    rules = []
    frozen = False
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith(FROZEN_MARKER):
            frozen = True
        if not stripped or stripped.startswith("#"):
            continue
        pattern, *owners = stripped.split()
        rules.append((number, pattern, owners, frozen))
    return rules


def matches(pattern, file):
    """Does a CODEOWNERS pattern match a repository-relative file path?"""
    if not pattern.startswith("/") or "*" in pattern or "?" in pattern:
        raise ValueError(f"unsupported CODEOWNERS pattern {pattern!r} (use /file or /folder/)")
    anchored = pattern[1:]
    if anchored.endswith("/"):
        return file.startswith(anchored)
    return file == anchored


def repository_files():
    """Every file git tracks, plus new files that are not ignored."""
    output = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout
    # A tracked file that was deleted but not yet committed is not there any more.
    return sorted(name for name in set(output.splitlines()) if (REPO_ROOT / name).is_file())


def check():
    """Return (gaps, bad_overlaps, overrides, unused) as lists of human-readable lines."""
    rules = read_rules(CODEOWNERS)
    files = repository_files()
    gaps, bad_overlaps, overrides = [], [], []

    for file in files:
        hits = [rule for rule in rules if matches(rule[1], file)]
        if not hits:
            gaps.append(file)
        elif len(hits) > 1:
            lines = ", ".join(f"line {number} ({pattern})" for number, pattern, _, _ in hits)
            winner_is_frozen = hits[-1][3]
            if winner_is_frozen and len(hits) == 2 and not hits[0][3]:
                overrides.append(f"{file}: {lines}; frozen line wins")
            else:
                bad_overlaps.append(f"{file}: {lines}")

    # A pattern that matches nothing is probably a typo or a stale path.
    unused = [f"line {number}: {pattern}" for number, pattern, _, _ in rules
              if not any(matches(pattern, file) for file in files)]
    return gaps, bad_overlaps, overrides, unused


def main():
    gaps, bad_overlaps, overrides, unused = check()

    print(f"Checked {len(repository_files())} files against {CODEOWNERS.relative_to(REPO_ROOT)}.")
    for title, items in (
        ("Files no CODEOWNERS line covers (gaps)", gaps),
        ("Files covered by more than one team line", bad_overlaps),
        ("CODEOWNERS lines that match no file", unused),
        ("Frozen overrides (expected)", overrides),
    ):
        print(f"\n{title}: {len(items)}")
        for item in items:
            print(f"  {item}")

    errors = len(gaps) + len(bad_overlaps) + len(unused)
    print("\nRESULT:", "OK" if errors == 0 else f"{errors} problem(s)")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
