#!/usr/bin/env python3
# ruff: noqa: T201
"""
Check plugin core version constraints before a release.

Usage:
    python check_plugin_core_compatibility.py VERSION [--bypass]
    python check_plugin_core_compatibility.py 1.0.0rc1
    python check_plugin_core_compatibility.py 1.0.0rc1 --bypass

The gating scope is the six OFFICIAL plugins (declared with
``[tool.spectrochempy] official-plugin = true``).  Cantera and the plugin
template are checked as informational only and never block the release.

``packaging.specifiers.SpecifierSet`` is used so PEP 440 semantics (including
release candidates such as ``1.0.0rc1``) are honored.

Exits with code 0 if all official plugins are compatible or --bypass is set.
Exits with code 1 if any official plugin has an incompatible constraint.
"""

import argparse
import sys
import tomllib
from pathlib import Path

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parents[3]
PLUGINS_DIR = REPO_ROOT / "plugins"


def is_official_plugin(pyproject: Path) -> bool:
    """Return whether a plugin declares itself official."""
    try:
        data = tomllib.loads(pyproject.read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return False
    return data.get("tool", {}).get("spectrochempy", {}).get("official-plugin") is True


def find_plugin_pyprojects() -> list[Path]:
    """Find all plugin pyproject.toml files in ``plugins/``."""
    result = []
    if not PLUGINS_DIR.exists():
        return result
    for plugin_dir in sorted(PLUGINS_DIR.iterdir()):
        pyproject = plugin_dir / "pyproject.toml"
        if pyproject.exists() and (
            plugin_dir.name.startswith("spectrochempy-")
            or plugin_dir.name == "plugin-template"
        ):
            result.append(pyproject)
    return result


def read_spectrochempy_constraint(pyproject: Path) -> str | None:
    """Return the spectrochempy dependency specifier from a pyproject.toml."""
    try:
        data = tomllib.loads(pyproject.read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return None
    for dep in data.get("project", {}).get("dependencies", []):
        name = dep.split(">=", 1)[0].split("<", 1)[0].split("==", 1)[0].strip()
        if name == "spectrochempy":
            return dep.split("spectrochempy", 1)[1].strip()
    return None


def version_allowed(version: str, specifier: str) -> bool:
    """
    Return whether ``version`` satisfies a PEP 440 specifier.

    ``prereleases=True`` is used explicitly: a release candidate of the target
    core version (e.g. ``1.0.0rc1``) is the precise case the gate must accept,
    and no release candidate of the core may be silently rejected merely
    because it is a prerelease.
    """
    try:
        parsed = Version(version)
    except InvalidVersion:
        return False
    return parsed in SpecifierSet(specifier, prereleases=True)


def main():
    parser = argparse.ArgumentParser(
        description="Check plugin core version constraint compatibility"
    )
    parser.add_argument(
        "version",
        help="Core version to check (e.g. 0.12.8, 1.0.0rc1, 1.0.0)",
    )
    parser.add_argument(
        "--bypass",
        action="store_true",
        help="Exit with code 0 even if incompatibilities are found",
    )
    args = parser.parse_args()

    try:
        Version(args.version)
    except InvalidVersion as exc:
        print(f"::error::Invalid core version {args.version!r}: {exc}")
        sys.exit(1)

    pyprojects = find_plugin_pyprojects()
    if not pyprojects:
        print("No plugin pyproject.toml files found — skipping constraint check.")
        sys.exit(0)

    official = [p for p in pyprojects if is_official_plugin(p)]
    informational = [p for p in pyprojects if not is_official_plugin(p)]

    print(
        f"Checking spectrochempy^{args.version} against "
        f"{len(official)} official plugin(s) and "
        f"{len(informational)} non-official plugin(s)\n"
    )

    failed = []

    for pyproject in official:
        plugin_name = pyproject.parent.name
        spec = read_spectrochempy_constraint(pyproject)
        if spec is None:
            print(
                f"::warning::{plugin_name}: no spectrochempy version constraint "
                "found (skipping check)"
            )
            continue
        compatible = version_allowed(args.version, spec)
        status = "OK" if compatible else "INCOMPATIBLE"
        print(
            f"{status}: [OFFICIAL] {plugin_name}: spectrochempy{spec} → "
            f"{args.version}"
        )
        if not compatible:
            failed.append((plugin_name, spec))

    for pyproject in informational:
        plugin_name = pyproject.parent.name
        spec = read_spectrochempy_constraint(pyproject)
        if spec is None:
            continue
        compatible = version_allowed(args.version, spec)
        status = "OK" if compatible else "INCOMPATIBLE (informational only)"
        print(
            f"{status}: [NON-OFFICIAL] {plugin_name}: spectrochempy{spec} → "
            f"{args.version} (does not gate the release)"
        )

    if failed:
        print("")
        print("::error::Some OFFICIAL plugins have incompatible constraints:")
        for name, spec in failed:
            print(
                f"::error::  - {name}: spectrochempy{spec} does not include "
                f"{args.version}"
            )
        print(
            "::error::Update the spectrochempy constraint in these plugins' "
            "pyproject.toml"
        )
        print("::error::before releasing, or re-run with --bypass to ignore.")

        if not args.bypass:
            sys.exit(1)
        print("--bypass set — exiting with code 0 despite incompatibilities.")

    sys.exit(0)


if __name__ == "__main__":
    main()
