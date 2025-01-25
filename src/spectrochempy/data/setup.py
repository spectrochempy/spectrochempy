# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
import shutil
import warnings
from pathlib import Path


def setup_mpl():
    """
    Install matplotlib styles and fonts.

    This function installs custom matplotlib styles and fonts for SpectroChemPy.
    It copies the stylesheets and fonts from the 'src/spectrochempy/data' directory to the appropriate
    matplotlib configuration directories.
    """
    try:
        import matplotlib as mpl
        from matplotlib import get_cachedir
    except ImportError:
        warnings.warn(
            "Sorry, but we cannot install mpl plotting styles and fonts "
            "if MatPlotLib is not installed.\n"
            "Please install MatPlotLib using:\n"
            "  pip install matplotlib\n"
            "or\n"
            "  conda install matplotlib\n"
            "and then install again."
        )
        return

    # Install all plotting styles in the matplotlib stylelib library
    stylesheets = Path(__file__).parent / "stylesheets"
    if not stylesheets.exists():
        raise IOError(
            f"Can't find the stylesheets from SpectroChemPy {str(stylesheets)}.\n"
            f"Installation incomplete!"
        )

    cfgdir = Path(mpl.get_configdir())
    stylelib = cfgdir / "stylelib"
    if not stylelib.exists():
        stylelib.mkdir()

    styles = stylesheets.glob("*.mplstyle")
    # check if our custom stylesheet are already installed and it not the case, install them
    if not all((stylelib / src.name).exists() for src in styles):
        print("Installing custom stylesheets...")
        for src in styles:
            dest = stylelib / src.name
            shutil.copy(src, dest)
            # print(f"Stylesheet {src} installed in {dest}")
        print("Custom stylesheets installed.")

    # Install fonts in mpl-data
    # see https://stackoverflow.com/a/47743010 discussion
    _dir_data = Path(mpl.get_data_path())

    dir_source = Path(__file__).parent / "fonts"
    if not dir_source.exists():
        raise IOError(f"directory {dir_source} not found!")

    dir_dest = _dir_data / "fonts" / "ttf"
    if not dir_dest.exists():
        dir_dest.mkdir(parents=True, exist_ok=True)

    # check if our custom fonts are already installed and it not the case, install them
    if not all((dir_dest / src.name).exists() for src in dir_source.glob("*.[ot]tf")):
        print("Installing custom fonts...")
        for src in dir_source.glob("*.[ot]tf"):
            dest = dir_dest / src.name
            shutil.copy(src, dest)
            # print(f"Font {src} installed in {dest}")
        print("Custom fonts installed.")

        # Delete font cache
        dir_cache = Path(get_cachedir())
        for file in list(dir_cache.glob("*.cache")) + list(dir_cache.glob("font*")):
            if not file.is_dir():
                file.unlink()
                # print(f"Deleted font cache {file}.")


if __name__ == "__main__":
    setup_mpl()
