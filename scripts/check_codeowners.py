"""Check that .github/CODEOWNERS covers every file in the repository.

    python scripts/check_codeowners.py

For every file git knows about (committed, plus new files that are not
ignored), find the CODEOWNERS lines that match it:

* no line         -> a GAP: nobody reviews changes to that file. Error.
* exactly one     -> fine.
* more than one   -> an OVERRIDE. GitHub uses the last matching line. That is
                     only allowed when the last line is MORE SPECIFIC than
                     every other line that matches: a file or folder inside
                     the folder those lines name. For example
                     /scenarios/light_seek.yaml (frozen, leads) listed after
                     /scenarios/ (world). Any other overlap is an error,
                     because it hides which team really owns the file (or
                     makes a more specific line useless).

Lines that match no file are listed as RESERVED, not as errors: some lines
name files a team will create later (the capstone drivers in
flycar/drivers/contrib/). Check that list for typos and stale paths.

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


def read_rules(path):
    """Return a list of (line_number, pattern, owners) from CODEOWNERS."""
    rules = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        pattern, *owners = stripped.split()
        rules.append((number, pattern, owners))
    return rules


def matches(pattern, file):
    """Does a CODEOWNERS pattern match a repository-relative file path?"""
    if not pattern.startswith("/") or "*" in pattern or "?" in pattern:
        raise ValueError(f"unsupported CODEOWNERS pattern {pattern!r} (use /file or /folder/)")
    anchored = pattern[1:]
    if anchored.endswith("/"):
        return file.startswith(anchored)
    return file == anchored


def is_inside(inner, outer):
    """True if pattern ``inner`` names something strictly inside folder pattern ``outer``."""
    return outer.endswith("/") and inner != outer and inner.startswith(outer)


def repository_files():
    """Every file git tracks, plus new files that are not ignored."""
    output = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout
    # A tracked file that was deleted but not yet committed is not there any more.
    return sorted(name for name in set(output.splitlines()) if (REPO_ROOT / name).is_file())


def check():
    """Return (gaps, bad_overlaps, overrides, reserved) as lists of human-readable lines."""
    rules = read_rules(CODEOWNERS)
    files = repository_files()
    gaps, bad_overlaps, overrides = [], [], []

    for file in files:
        hits = [rule for rule in rules if matches(rule[1], file)]
        if not hits:
            gaps.append(file)
        elif len(hits) > 1:
            lines = ", ".join(f"line {number} ({pattern})" for number, pattern, _ in hits)
            winner = hits[-1][1]
            if all(is_inside(winner, pattern) for _, pattern, _ in hits[:-1]):
                overrides.append(f"{file}: {lines}; line {hits[-1][0]} wins")
            else:
                bad_overlaps.append(f"{file}: {lines}")

    # A line that matches nothing is either reserved for a file to come, or a typo.
    reserved = [f"line {number}: {pattern}" for number, pattern, _ in rules
                if not any(matches(pattern, file) for file in files)]
    return gaps, bad_overlaps, overrides, reserved


def main():
    gaps, bad_overlaps, overrides, reserved = check()

    print(f"Checked {len(repository_files())} files against {CODEOWNERS.relative_to(REPO_ROOT)}.")
    for title, items in (
        ("Files no CODEOWNERS line covers (gaps)", gaps),
        ("Files whose matching lines overlap the wrong way", bad_overlaps),
        ("Lines that match no file yet (reserved; check for typos)", reserved),
        ("Specific lines overriding a folder line (expected)", overrides),
    ):
        print(f"\n{title}: {len(items)}")
        for item in items:
            print(f"  {item}")

    errors = len(gaps) + len(bad_overlaps)
    print("\nRESULT:", "OK" if errors == 0 else f"{errors} problem(s)")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
