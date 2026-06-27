#!/usr/bin/env bash
# Install all local spectrochempy plugins in editable mode.
# Excludes plugin-template (scaffolding template, not a plugin).
#
# Prefer the more flexible Python alternative:
#   python -m spectrochempy.ci.install_plugins --editable all

set -euo pipefail

plugins_dir="$(cd "$(dirname "$0")/../../.." && pwd)/plugins"

for pkg in "$plugins_dir"/spectrochempy-*/; do
    echo "Installing $(basename "$pkg") in editable mode..."
    pip install -e "$pkg"
done

echo "All plugins installed."
