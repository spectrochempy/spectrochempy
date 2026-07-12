#!/usr/bin/env python3
"""Evaluate whether a PR can safely bypass heavyweight CI jobs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

SAFE_DOCS_LABEL = "safe-docs-no-ci"

SAFE_EXACT = {
    "AGENTS.md",
    "CONTRIBUTING.md",
    "README.md",
    ".github/PULL_REQUEST_TEMPLATE.md",
}

SAFE_PREFIXES = (
    "maintainers/",
)

UNSAFE_PREFIXES = (
    "docs/",
    "examples/",
    "src/",
    "tests/",
    "plugins/",
    ".github/workflows/",
)


def _run_git(args: list[str]) -> list[str]:
    result = subprocess.run(
        ["git", *args],
        check=True,
        text=True,
        capture_output=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _is_zero_sha(value: str) -> bool:
    return bool(value) and set(value) == {"0"}


def changed_files(base: str | None, head: str | None) -> list[str]:
    """Return changed files between two refs, using conservative fallbacks."""
    if not head:
        head = "HEAD"

    if base and not _is_zero_sha(base):
        for separator in ("...", ".."):
            try:
                return _run_git(["diff", "--name-only", f"{base}{separator}{head}"])
            except subprocess.CalledProcessError:
                continue

    try:
        return _run_git(["diff", "--name-only", "HEAD~1..HEAD"])
    except subprocess.CalledProcessError:
        return []


def is_safe_docs_only(files: list[str]) -> bool:
    if not files:
        return False
    for path in files:
        if any(path.startswith(prefix) for prefix in UNSAFE_PREFIXES):
            return False
        if path in SAFE_EXACT:
            continue
        if any(path.startswith(prefix) for prefix in SAFE_PREFIXES):
            if path.endswith(".md"):
                continue
            return False
        if path.endswith(".md"):
            continue
        return False
    return True


def write_github_output(*, should_skip: bool, reason: str) -> None:
    output_file = os.environ.get("GITHUB_OUTPUT")
    if not output_file:
        return
    with Path(output_file).open("a", encoding="utf-8") as stream:
        stream.write(f"skip={'true' if should_skip else 'false'}\n")
        stream.write(f"reason={reason}\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base")
    parser.add_argument("--head")
    parser.add_argument("--label", action="append", default=[])
    args = parser.parse_args()

    files = changed_files(args.base, args.head)
    labels = set(args.label)
    has_label = SAFE_DOCS_LABEL in labels
    safe_only = is_safe_docs_only(files)
    should_skip = has_label and safe_only

    if not has_label:
        reason = f"label '{SAFE_DOCS_LABEL}' not present"
    elif not safe_only:
        reason = "changed files are not limited to safe documentation/policy paths"
    else:
        reason = "safe documentation/policy-only PR explicitly labeled for CI bypass"

    print("Changed files:")
    for path in files:
        print(f"  - {path}")
    if not files:
        print("  none")
    print(f"Labels: {sorted(labels)}")
    print(f"Safe docs only: {safe_only}")
    print(f"Skip heavyweight CI: {should_skip}")
    print(f"Reason: {reason}")

    write_github_output(should_skip=should_skip, reason=reason)
    return 0


if __name__ == "__main__":
    sys.exit(main())
