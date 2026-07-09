#!/usr/bin/env python3
# ruff: noqa: T201
"""
Check plugin core version constraints before a release.

Usage:
    python check_plugin_core_compatibility.py VERSION [--bypass]
    python check_plugin_core_compatibility.py 0.11.0
    python check_plugin_core_compatibility.py 0.11.0 --bypass

Exits with code 0 if all plugins are compatible or --bypass is set.
Exits with code 1 if any plugin has an incompatible constraint.
"""

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PLUGINS_DIR = REPO_ROOT / "plugins"


def parse_version(version_str):
    """Parse a PEP 440 version string into a tuple of ints."""
    # Remove any dev/suffix (e.g. 0.11.0.dev42 -> 0.11.0)
    match = re.match(r"(\d+)\.(\d+)\.(\d+)", version_str)
    if not match:
        raise ValueError(f"Could not parse version: {version_str}")
    return tuple(int(g) for g in match.groups())


def parse_constraint(constraint_str):
    """Parse a version constraint like >=0.10,<0.12 into a list of (op, ver) pairs."""
    constraints = []
    for part in constraint_str.split(","):
        part = part.strip()
        match = re.match(r"(>=|<=|!=|==|~=|<|>)\s*(\d+\.\d+\.\d+)", part)
        if match:
            op, ver = match.group(1), parse_version(match.group(2))
            constraints.append((op, ver))
        else:
            match = re.match(r"(>=|<=|!=|==|~=|<|>)\s*(\d+\.\d+)", part)
            if match:
                op, ver = match.group(1), parse_version(match.group(2) + ".0")
                constraints.append((op, ver))
    return constraints


def version_satisfies(version, constraints):
    """Check if a version tuple satisfies all constraints."""
    for op, constraint_ver in constraints:
        if op == ">=":
            if not (version >= constraint_ver):
                return False
        elif op == "<=":
            if not (version <= constraint_ver):
                return False
        elif op == ">":
            if not (version > constraint_ver):
                return False
        elif op == "<":
            if not (version < constraint_ver):
                return False
        elif op == "==":
            if version != constraint_ver:
                return False
        elif op == "!=":
            if version == constraint_ver:
                return False
        elif op == "~=" and not (
            version >= constraint_ver and version < (constraint_ver[0] + 1, 0, 0)
        ):
            return False
    return True


def find_plugin_pyproject_tomls():
    """Find all official plugin pyproject.toml files."""
    result = []
    if not PLUGINS_DIR.exists():
        return result
    for plugin_dir in sorted(PLUGINS_DIR.iterdir()):
        if (
            plugin_dir.name.startswith("spectrochempy-")
            or plugin_dir.name == "plugin-template"
        ):
            pyproject = plugin_dir / "pyproject.toml"
            if pyproject.exists():
                result.append(pyproject)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Check plugin core version constraint compatibility"
    )
    parser.add_argument("version", help="Core version to check (e.g. 0.11.0)")
    parser.add_argument(
        "--bypass",
        action="store_true",
        help="Exit with code 0 even if incompatibilities are found",
    )
    args = parser.parse_args()

    try:
        core_version = parse_version(args.version)
    except ValueError as e:
        print(f"::error::{e}")
        sys.exit(1)

    plugins = find_plugin_pyproject_tomls()
    if not plugins:
        print("No plugin pyproject.toml files found — skipping constraint check.")
        sys.exit(0)

    failed = []

    for pyproject in plugins:
        plugin_name = pyproject.parent.name
        text = pyproject.read_text()
        # Extract spectrochempy dependency constraint from dependencies list
        match = re.search(
            r'"spectrochempy\s*((?:>=|<=|!=|==|~=|<|>)\s*[\d.]+\s*(?:,\s*(?:>=|<=|!=|==|~=|<|>)\s*[\d.]+)*)"',
            text,
        )
        if not match:
            print(
                f"::warning::{plugin_name}: no spectrochempy version constraint found "
                "(skipping check)"
            )
            continue

        constraint_str = match.group(1)
        constraints = parse_constraint(constraint_str)
        compatible = version_satisfies(core_version, constraints)

        status = "OK" if compatible else "INCOMPATIBLE"
        print(
            f"{status}: {plugin_name}: spectrochempy{constraint_str} → {args.version}"
        )

        if not compatible:
            failed.append((plugin_name, constraint_str))

    if failed:
        print("")
        print("::error::Some official plugins have incompatible version constraints:")
        for name, constraint in failed:
            print(
                f"::error::  - {name}: spectrochempy{constraint} "
                f"does not include {args.version}"
            )
        print(
            "::error::Update the spectrochempy constraint in these plugins' pyproject.toml"
        )
        print("::error::before releasing, or re-run with --bypass to ignore.")

        if not args.bypass:
            sys.exit(1)
        print("--bypass set — exiting with code 0 despite incompatibilities.")

    sys.exit(0)


if __name__ == "__main__":
    main()
