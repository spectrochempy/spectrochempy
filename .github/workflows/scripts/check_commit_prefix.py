#!/usr/bin/env python3
"""Validate that the commit message starts with a prefix listed in CONTRIBUTING.md."""

import re
import sys
from pathlib import Path


def get_allowed_prefixes():
    """Read prefix table from CONTRIBUTING.md."""
    contributing = Path(__file__).resolve().parents[3] / "CONTRIBUTING.md"
    text = contributing.read_text(encoding="utf-8")
    # Match lines like: | `ENH:`   | User-facing enhancement...
    matches = re.findall(r"^\| `([A-Z]+):`", text, re.MULTILINE)
    return [f"{m}:" for m in matches]


def main():
    if len(sys.argv) < 2:
        print("Usage: check_commit_prefix.py <commit-msg-file>")
        sys.exit(1)

    msg_file = Path(sys.argv[1])
    lines = msg_file.read_text(encoding="utf-8").splitlines()

    if not lines:
        print("Error: empty commit message.")
        sys.exit(1)

    subject = lines[0].strip()

    # Skip merge commits and reverts — they have well-known prefixes
    if subject.startswith("Merge ") or subject.startswith("Revert "):
        sys.exit(0)

    prefixes = get_allowed_prefixes()
    prefix_re = re.compile(r"^(" + "|".join(re.escape(p) for p in prefixes) + r")")

    if not prefix_re.search(subject):
        print("Error: commit message does not start with a valid prefix.")
        print(f"Subject: '{subject}'")
        print(f"Allowed prefixes: {', '.join(prefixes)}")
        print("See CONTRIBUTING.md for the full prefix convention.")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
