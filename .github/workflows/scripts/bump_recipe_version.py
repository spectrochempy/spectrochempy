#!/usr/bin/env python3
"""
Bump context.version in a conda recipe.yaml, leaving package.version intact.

conda recipes have two version: fields:
  - context.version: "0.1.0"      # actual value, must be bumped
  - package.version: "${{ version }}"   # Jinja2 reference, must NOT be changed
"""
# ruff: noqa: T201

import re
import sys


def bump_recipe_version(recipe_path: str, new_version: str) -> None:
    with open(recipe_path) as f:
        lines = f.readlines()

    context_indent = -1
    found = False
    old_val = ""

    for i, line in enumerate(lines):
        text = line.rstrip("\n")
        indent = len(text) - len(text.lstrip())
        content = text.strip()

        if content == "context:":
            context_indent = indent
            continue

        if context_indent >= 0 and indent <= context_indent and content:
            break

        if context_indent >= 0:
            m = re.match(rf"^ {{{context_indent + 2}}}version\s*:\s*\"(.+)\"\s*$", line)
            if m:
                old_val = m.group(1)
                prefix = " " * (context_indent + 2)
                lines[i] = f'{prefix}version: "{new_version}"\n'
                found = True
                break

    if not found:
        print(f"::error::Could not find context.version in {recipe_path}")
        sys.exit(1)

    with open(recipe_path, "w") as f:
        f.writelines(lines)

    print(f'Updated context.version in {recipe_path}: "{old_val}" -> "{new_version}"')


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: bump_recipe_version.py <recipe.yaml> <new_version>")
        sys.exit(1)
    bump_recipe_version(sys.argv[1], sys.argv[2])
