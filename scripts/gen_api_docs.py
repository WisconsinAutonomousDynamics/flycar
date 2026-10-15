"""Regenerate the tables in docs/API.md from flycar/env/spaces.py.

    python scripts/gen_api_docs.py           # rewrite the generated block
    python scripts/gen_api_docs.py --check   # only report whether it is up to date

``spaces.py`` is the single source of truth for the action and observation
definitions. Everything in docs/API.md between the two marker comments is
generated from it; the rest of that file is written by hand. A test
(tests/test_contracts.py) fails if the two ever disagree, so after changing
``spaces.py`` just run this script.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))   # so "import flycar" works from anywhere

from flycar.env.spaces import api_markdown  # noqa: E402  (import after the path tweak)

API_DOC = REPO_ROOT / "docs" / "API.md"
BEGIN = "<!-- BEGIN GENERATED FROM flycar/env/spaces.py (do not edit by hand) -->"
END = "<!-- END GENERATED -->"


def rebuild(text):
    """Return ``text`` with the block between the markers replaced by fresh tables."""
    if BEGIN not in text or END not in text:
        raise SystemExit(f"{API_DOC} is missing the BEGIN/END GENERATED marker comments.")
    before = text.split(BEGIN)[0]
    after = text.split(END)[1]
    return before + BEGIN + "\n\n" + api_markdown() + "\n" + END + after


def main():
    parser = argparse.ArgumentParser(description="Regenerate the tables in docs/API.md.")
    parser.add_argument("--check", action="store_true", help="Do not write; exit with 1 if the file is out of date")
    args = parser.parse_args()

    current = API_DOC.read_text(encoding="utf-8")
    updated = rebuild(current)

    if args.check:
        if current == updated:
            print("docs/API.md is up to date.")
            return 0
        print("docs/API.md is OUT OF DATE. Run: python scripts/gen_api_docs.py")
        return 1

    API_DOC.write_text(updated, encoding="utf-8", newline="\n")
    print(f"Wrote {API_DOC}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
