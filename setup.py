# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2021 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory
# ======================================================================================================================

import atexit
import warnings
import shutil
from pathlib import Path

from setuptools import setup, find_packages
from setuptools.command.develop import develop as _develop
from setuptools.command.install import install as _install


def install_styles():
    """
    Install matplotlib styles

    """
    try:
        import matplotlib as mpl
    except ImportError:
        warnings.warn('Sorry, but we cannot install mpl plotting styles '
                      'if MatPlotLib is not installed.\n'
                      'Please install MatPlotLib using:\n'
                      '  pip install matplotlib\n'
                      'or\n'
                      '  conda install matplotlib\n'
                      'and then install again.')
        return

    # install all plotting styles in the matplotlib stylelib library
    stylesheets = Path("scp_data") / "stylesheets"
    if not stylesheets.exists():
        raise IOError(f"can't find the stylesheets from SpectroChemPy {str(stylesheets)}. Installation incomplete!")

    cfgdir = Path(mpl.get_configdir())
    stylelib = cfgdir / 'stylelib'
    if not stylelib.exists():
        stylelib.mkdir()

    styles = stylesheets.glob('*.mplstyle')
    for src in styles:
        dest = stylelib / src.name
        shutil.copy(src, dest)
    print(f'STYLESHEETS installed in {stylesheets}')


class PostInstallCommand(_install):
    """Post-installation for installation mode."""

    def run(self):
        atexit.register(install_styles)
        _install.run(self)


class PostDevelopCommand(_develop):
    """Post-installation for development mode."""

    def run(self):
        def installstyles():
            return install_styles()

        _develop.run(self)
        atexit.register(installstyles)


# Data for setuptools
packages = []

setup_args = dict(

        # packages informations
        name="spectrochempy", use_scm_version=True, license="CeCILL-B Free Software",
        author="Arnaud Travert & Christian Fernandez", author_email="contact (at) spectrochempy.fr",
        maintainer="C. Fernandez", maintainer_email="christian.fernandez (at) ensicaen.fr",
        url='http:/www.spectrochempy.fr', description='Processing, analysis and modelling Spectroscopic data for '
                                                      'Chemistry with Python',
        long_description=Path('README.md').read_text(), long_description_content_type="text/markdown",
        classifiers=["Development Status :: 3 - Alpha",
                     "Topic :: Utilities", "Topic :: Scientific/Engineering",
                     "Topic :: Software Development :: Libraries",
                     "Intended Audience :: Science/Research",
                     "License :: CeCILL-B Free Software License Agreement (CECILL-B)",
                     "Operating System :: OS Independent",
                     "Programming Language :: Python :: 3.7",
                     "Programming Language :: Python :: 3.8",
                     "Programming Language :: Python :: 3.9", ],
        platforms=['Windows', 'Mac OS X', 'Linux'],

        # packages discovery
        zip_safe=False, packages=find_packages() + packages, include_package_data=True,  # requirements
        python_requires=">=3.7", setup_requires=['setuptools_scm', 'matplotlib'], install_requires=[],
        # install_requires(dev=__DEV__),
        # tests_require=extras_require['tests'],

        # post-commands
        cmdclass={'develop': PostDevelopCommand, 'install': PostInstallCommand, },

        # entry points for scripts
        # scripts = {'scripts/launch_api.py'},
        entry_points={'console_scripts': ['scpy_gui=spectrochempy.gui.scpy_gui:main',
                                          'scpy_update=spectrochempy.scripts.scpy_update:main'], }, )

# ======================================================================================================================
if __name__ == '__main__':
    # execute setup
    setup(**setup_args)
