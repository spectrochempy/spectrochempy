#!/usr/bin/env python3
"""
Validate that a plugin directory is declared as an official SpectroChemPy plugin.

Usage::

    python .github/workflows/scripts/validate_official_plugin.py plugins/spectrochempy-nmr

Checks that ``pyproject.toml`` contains::

    [tool.spectrochempy]
    official-plugin = true

Exits with code 0 on success, 1 on failure.
"""
# ruff: noqa: T201

from __future__ import annotations

import sys
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]

LEGACY_CLASSIFIER = "Framework :: SpectroChemPy :: Official Plugin"


def validate_plugin(plugin_dir: Path) -> list[str]:
    """Return a list of error messages (empty = valid)."""
    errors: list[str] = []
    pyproject = plugin_dir / "pyproject.toml"

    if not pyproject.is_file():
        return [f"pyproject.toml not found: {pyproject}"]

    try:
        data = tomllib.loads(pyproject.read_text())
    except Exception as exc:
        return [f"Failed to parse {pyproject}: {exc}"]

    # Check for the new [tool.spectrochempy] marker
    tool_sc = data.get("tool", {}).get("spectrochempy", {})
    official = tool_sc.get("official-plugin")

    if official is not True:
        errors.append(
            f"{plugin_dir.name}: [tool.spectrochempy] official-plugin is not set to true "
            f"(got {official!r}).  Add the following to pyproject.toml:\n"
            f"\n  [tool.spectrochempy]\n  official-plugin = true"
        )

    # Warn if the legacy classifier is still present (will cause PyPI rejection)
    classifiers = data.get("project", {}).get("classifiers", [])
    if LEGACY_CLASSIFIER in classifiers:
        errors.append(
            f"{plugin_dir.name}: pyproject.toml still contains the invalid Trove classifier "
            f"'{LEGACY_CLASSIFIER}'.  Remove it from project.classifiers — "
            f"official status is now declared via [tool.spectrochempy]."
        )

    return errors


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: validate_official_plugin.py <plugin-dir>", file=sys.stderr)
        return 2

    plugin_dir = Path(sys.argv[1])
    if not plugin_dir.is_dir():
        print(f"Error: directory not found: {plugin_dir}", file=sys.stderr)
        return 1

    errors = validate_plugin(plugin_dir)
    if errors:
        for error in errors:
            print(f"::error::{error}", file=sys.stderr)
            print(error, file=sys.stderr)
        return 1

    print(f"✓ {plugin_dir.name} is a valid official plugin")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
