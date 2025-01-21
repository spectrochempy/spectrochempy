# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
import shutil
import warnings
from pathlib import Path

from setuptools import find_packages, setup
from setuptools.command.develop import develop as _develop
from setuptools.command.install import install as _install
from setuptools_scm import get_version


def version():
    return get_version(root=".", relative_to=__file__).split("+")[0]


def _install_mpl():
    """
    Install matplotlib styles and fonts
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

    # install all plotting styles in the matplotlib stylelib library
    stylesheets = Path("scp_data") / "stylesheets"
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
    for src in styles:
        dest = stylelib / src.name
        shutil.copy(src, dest)
        print(f"Stylesheet {src} installed in {dest}")

    # install fonts in mpl-data
    # https://stackoverflow.com/a/47743010

    # Copy files over
    _dir_data = Path(mpl.get_data_path())

    dir_source = Path("scp_data") / "fonts"
    if not dir_source.exists():
        raise IOError(f"directory {dir_source} not found!")

    dir_dest = _dir_data / "fonts" / "ttf"
    if not dir_dest.exists():
        dir_dest.mkdir(parents=True, exist_ok=True)

    for file in dir_source.glob("*.[ot]tf"):
        if not (dir_dest / file.name).exists():
            print(f'Adding font "{file.name}".')
            shutil.copy(file, dir_dest)
            if (dir_dest / file.name).exists():
                print("success")

    # Delete cache
    dir_cache = Path(get_cachedir())
    for file in list(dir_cache.glob("*.cache")) + list(dir_cache.glob("font*")):
        if not file.is_dir():  # don't dump the tex.cache folder... because dunno why
            file.unlink()
            print(f"Deleted font cache {file}.")


class PostInstallCommand(_install):
    """Post-installation for installation mode."""

    def run(self):
        _install_mpl()
        _install.run(self)


class PostDevelopCommand(_develop):
    """Post-installation for development mode."""

    def run(self):
        _install_mpl()
        _develop.run(self)


# Data for setuptools
packages = []
setup_args = dict(
    name="spectrochempy",
    version=version(),
    zip_safe=False,
    packages=find_packages() + packages,
    include_package_data=True,
    python_requires=">=3.10",
    cmdclass={
        "develop": PostDevelopCommand,
        "install": PostInstallCommand,
    },
    entry_points={
        "console_scripts": [
            "show-versions=spectrochempy.scripts.show_versions:main",
        ],
    },
)

if __name__ == "__main__":
    setup(**setup_args)
