# ======================================================================================
# Copyright (C) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

"""Install bundled SpectroChemPy plugins for local development."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PLUGIN_DIRS = {
    "nmr": "spectrochempy-topspin",
    "iris": "spectrochempy-iris",
    "cantera": "spectrochempy-cantera",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _plugin_path(name: str) -> Path:
    return _repo_root() / "plugins" / PLUGIN_DIRS[name]


def install_plugins(names: list[str], *, editable: bool = False) -> None:
    """Install selected bundled plugins using the current Python interpreter."""
    selected = list(PLUGIN_DIRS) if names == ["all"] else names
    for name in selected:
        if name not in PLUGIN_DIRS:
            choices = ", ".join([*PLUGIN_DIRS, "all"])
            raise SystemExit(f"Unknown plugin '{name}'. Choose one of: {choices}")
        path = _plugin_path(name)
        cmd = [sys.executable, "-m", "pip", "install"]
        if editable:
            cmd.append("-e")
        cmd.append(str(path))
        subprocess.run(cmd, check=True)  # noqa: S603 - plugin names are allow-listed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Install bundled SpectroChemPy plugins for local development."
    )
    parser.add_argument(
        "plugins",
        nargs="*",
        default=["all"],
        help="Plugin names to install: nmr, iris, cantera, or all.",
    )
    parser.add_argument(
        "-e",
        "--editable",
        action="store_true",
        help="Install plugins in editable mode.",
    )
    args = parser.parse_args(argv)
    install_plugins(args.plugins, editable=args.editable)


if __name__ == "__main__":
    main()
