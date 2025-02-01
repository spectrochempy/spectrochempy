# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Setup script for matplotlib custom styles and fonts.

This module provides functionality to install custom matplotlib styles and fonts
"""

import shutil
import warnings
from os import environ
from pathlib import Path


def is_on_github_actions():
    """
    Check if the code is running on GitHub Actions.

    Returns
    -------
    bool
        True if running on GitHub Actions, False otherwise.

    """
    required_vars = ["CI", "GITHUB_RUN_ID", "GITHUB_REPOSITORY"]
    return all(var in environ and environ.get(var) for var in required_vars)


def setup_mpl():
    """
    Install matplotlib styles and fonts for SpectroChemPy.

    This function:
    1. Checks the execution environment (local or GitHub Actions)
    2. Verifies matplotlib installation
    3. Installs custom stylesheets if not already present
    4. Installs custom fonts if not already present
    5. Cleans up font cache after installation

    Raises
    ------
    ImportError
        If matplotlib is not installed
    IOError
        If source directories for styles or fonts are not found

    """
    # Check execution environment
    GITHUB = is_on_github_actions()
    if GITHUB:
        print("Running on GitHub Actions")  # noqa: T201

    # Verify matplotlib installation
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        from matplotlib import get_cachedir
    except ImportError:
        warnings.warn(
            "Sorry, but we cannot install mpl plotting styles and fonts "
            "if MatPlotLib is not installed.\n"
            "Please install MatPlotLib using:\n"
            "  pip install matplotlib\n"
            "or\n"
            "  conda install matplotlib\n"
            "and then install again.",
            stacklevel=2,
        )
        return

    # Setup paths for stylesheets
    stylesheets = Path(__file__).parent / "stylesheets"
    if not stylesheets.exists():
        raise OSError(
            f"Can't find the stylesheets from SpectroChemPy {stylesheets!s}.\n"
            f"Installation incomplete!",
        )

    # Ensure stylelib directory exists
    cfgdir = Path(mpl.get_configdir())
    stylelib = cfgdir / "stylelib"
    if not stylelib.exists():
        stylelib.mkdir()

    if GITHUB:
        print(f"MPL Configuration directory: {cfgdir}")  # noqa: T201
        print(f"Stylelib directory: {stylelib}")  # noqa: T201

    # Install stylesheets if needed
    styles = list(stylesheets.glob("*.mplstyle"))
    if not all((stylelib / src.name).exists() for src in styles):
        print("Installing custom stylesheets...")  # noqa: T201
        for src in styles:
            dest = stylelib / src.name
            shutil.copy(src, dest)
            if dest.exists():
                print(f"Stylesheet {src.name} installed successfully")  # noqa: T201
            else:
                print(f"Failed to install stylesheet {src.name}")  # noqa: T201

        # Reload matplotlib style library
        plt.style.reload_library()

        if GITHUB:
            print("\nAvailable stylesheets:")  # noqa: T201
            print("\n".join(f"- {style}" for style in plt.style.available))  # noqa: T201

    # Setup paths for fonts
    dir_source = Path(__file__).parent / "fonts"
    if not dir_source.exists():
        raise OSError(f"Fonts directory not found: {dir_source}")

    dir_dest = Path(mpl.get_data_path()) / "fonts" / "ttf"
    if not dir_dest.exists():
        dir_dest.mkdir(parents=True, exist_ok=True)

    # Install fonts if needed
    fonts = list(dir_source.glob("*.[ot]tf"))
    if not all((dir_dest / src.name).exists() for src in fonts):
        print("\nInstalling custom fonts...")  # noqa: T201
        for src in fonts:
            dest = dir_dest / src.name
            shutil.copy(src, dest)
            print(f"Font {src.name} installed successfully")  # noqa: T201

        # Clear font cache
        dir_cache = Path(get_cachedir())
        for cache_file in dir_cache.glob("*.cache"):
            if not cache_file.is_dir():
                cache_file.unlink()
                print(f"Cleared font cache: {cache_file.name}")  # noqa: T201


if __name__ == "__main__":
    setup_mpl()
