#!/usr/bin/env python3
"""Select conservative pytest targets for CI runs."""

# ruff: noqa: S603, S607, T201

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

FULL_TARGETS = ["tests"]
DOC_TARGETS = ["tests/test_docs"]
PLUGIN_TESTS = {
    "spectrochempy-cantera": "plugins/spectrochempy-cantera/tests",
    "spectrochempy-iris": "plugins/spectrochempy-iris/tests",
    "spectrochempy-nmr": "plugins/spectrochempy-nmr/tests",
}
ALL_PLUGIN_TARGETS = ["tests/test_plugins", *PLUGIN_TESTS.values()]
PROTECTED_REFS = {"master", "develop"}
FULL_TEST_FILES = {
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "tox.ini",
    "noxfile.py",
    "tests/conftest.py",
}
FULL_TEST_PREFIXES = (
    ".github/workflows/",
    ".github/actions/",
    "src/spectrochempy/",
)
PLUGIN_CORE_PREFIXES = (
    "src/spectrochempy/plugins/",
    "tests/test_plugins/",
    "plugins/plugin-template/",
)
DOC_PREFIXES = (
    "docs/",
    "examples/",
)


def _run_git(args: list[str]) -> list[str]:
    result = subprocess.run(  # noqa: S603
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


def _add_existing(targets: list[str], target: str) -> None:
    if target not in targets and Path(target).exists():
        targets.append(target)


def _plugin_targets(path: str, targets: list[str]) -> bool:
    for plugin_name, test_path in PLUGIN_TESTS.items():
        if path.startswith(f"plugins/{plugin_name}/"):
            # Individual plugin tests cannot be imported from the repo root
            # due to pytest namespace resolution (tests.test_* conflicts).
            # They must be run from their own directory. Only add core tests.
            _add_existing(targets, "tests/test_plugins")
            return True
    return False


def select_targets(
    files: list[str],
    *,
    event_name: str = "",
    ref_name: str = "",
    base_ref: str = "",
) -> tuple[str, list[str], str]:
    """
    Select pytest targets.

    The selector only narrows tests for low-risk categories. Shared core,
    packaging, and workflow changes keep the full test suite.
    """
    if event_name in {"schedule", "workflow_dispatch"}:
        return "full", FULL_TARGETS, "scheduled or manually dispatched run"

    if ref_name in PROTECTED_REFS or base_ref in PROTECTED_REFS:
        return "full", FULL_TARGETS, "protected branch run"

    if not files:
        return "full", FULL_TARGETS, "no changed files detected"

    targets: list[str] = []
    docs_only = True

    for path in sorted(set(files)):
        if path in FULL_TEST_FILES:
            return "full", FULL_TARGETS, f"{path} requires full tests"

        if any(path.startswith(prefix) for prefix in PLUGIN_CORE_PREFIXES):
            docs_only = False
            # Core plugin system changes only need the core plugin test suite.
            # Individual plugin tests are run only when that specific plugin is
            # modified (handled by _plugin_targets below).
            _add_existing(targets, "tests/test_plugins")
            continue

        if _plugin_targets(path, targets):
            docs_only = False
            continue

        if path.startswith("tests/"):
            docs_only = False
            if path.endswith(".py") and Path(path).exists():
                _add_existing(targets, path)
            else:
                return "full", FULL_TARGETS, f"{path} requires full tests"
            continue

        if any(path.startswith(prefix) for prefix in DOC_PREFIXES):
            for target in DOC_TARGETS:
                _add_existing(targets, target)
            continue

        if any(path.startswith(prefix) for prefix in FULL_TEST_PREFIXES):
            return "full", FULL_TARGETS, f"{path} requires full tests"

        docs_only = False

    if targets:
        return "targeted", targets, "low-risk plugin, test, or documentation changes"

    if docs_only:
        return "targeted", DOC_TARGETS, "documentation-only changes"

    return "full", FULL_TARGETS, "changed files are outside targeted rules"


def write_github_env(mode: str, targets: list[str], reason: str) -> None:
    env_file = os.environ.get("GITHUB_ENV")
    if not env_file:
        return

    with Path(env_file).open("a", encoding="utf-8") as stream:
        stream.write(f"SCPY_TEST_SELECTION={mode}\n")
        stream.write(f"SCPY_TEST_TARGETS={' '.join(targets)}\n")
        stream.write(f"SCPY_TEST_SELECTION_REASON={reason}\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base")
    parser.add_argument("--head")
    parser.add_argument("--event-name", default=os.environ.get("GITHUB_EVENT_NAME", ""))
    parser.add_argument("--ref-name", default=os.environ.get("GITHUB_REF_NAME", ""))
    parser.add_argument("--base-ref", default=os.environ.get("GITHUB_BASE_REF", ""))
    parser.add_argument(
        "--files", nargs="*", help="Explicit changed files for testing."
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    files = (
        args.files if args.files is not None else changed_files(args.base, args.head)
    )
    mode, targets, reason = select_targets(
        files,
        event_name=args.event_name,
        ref_name=args.ref_name,
        base_ref=args.base_ref,
    )

    print("Changed files:")
    for path in files:
        print(f"  - {path}")
    if not files:
        print("  none")
    print(f"Test selection: {mode}")
    print(f"Reason: {reason}")
    print(f"Pytest targets: {' '.join(targets)}")

    if not args.dry_run:
        write_github_env(mode, targets, reason)
    return 0


if __name__ == "__main__":
    sys.exit(main())
