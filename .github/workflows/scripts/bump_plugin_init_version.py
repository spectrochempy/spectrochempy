#!/usr/bin/env python3
# ruff: noqa: T201
"""
Bump version string in a SpectroChemPy plugin's ``__init__.py``.

The plugin class attribute ``version = "..."`` is updated to match the
release version.  The corresponding file path is derived from the plugin
package name:

    spectrochempy-nmr
    -> plugins/spectrochempy-nmr/src/spectrochempy_nmr/__init__.py
"""

import re
import sys


def plugin_init_path(plugin_name: str) -> str:
    """Map a plugin package name to its ``__init__.py`` path."""
    src_dir = plugin_name.replace("-", "_")
    return f"plugins/{plugin_name}/src/{src_dir}/__init__.py"


def bump_plugin_init_version(init_path: str, new_version: str) -> None:
    pattern = re.compile(r"^( {4}version\s*=\s*)\"[^\"]*\"(\s*)$")

    with open(init_path, newline="") as f:
        lines = f.readlines()

    found = False
    old_val = ""

    for i, line in enumerate(lines):
        m = pattern.match(line)
        if m:
            old_m = re.search(r'"([^"]*)"', line)
            old_val = old_m.group(1) if old_m else ""
            lines[i] = f'{m.group(1)}"{new_version}"{m.group(2)}'
            found = True
            break

    if not found:
        print(f'::error::Could not find `    version = "..."` in {init_path}')
        sys.exit(1)

    with open(init_path, "w", newline="") as f:
        f.writelines(lines)

    print(f'Updated {init_path}: "{old_val}" -> "{new_version}"')


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: bump_plugin_init_version.py <plugin_name> <new_version>")
        sys.exit(1)
    plugin_name = sys.argv[1]
    new_version = sys.argv[2]
    init_path = plugin_init_path(plugin_name)
    bump_plugin_init_version(init_path, new_version)
