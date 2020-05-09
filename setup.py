# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

from setuptools import setup, find_packages
from setuptools.command.develop import develop as _develop
from setuptools.command.install import install as _install

import os
import sys
import subprocess
import shutil as sh
import warnings
import version

# ----------------------------------------------------------------------------------------------------------------------

__DEV__ = False
if 'develop' in sys.argv:
    __DEV__ = True
    
    
def path():
    return os.path.dirname(__file__)


def install_styles():
    """
    Install matplotlib styles

    """
    try:
        import matplotlib as mpl
    except:
        warnings.warn('Sorry, but we cannot install mpl plotting styles '
                      'if MatPlotLib is not installed.\n'
                      'Please install MatPlotLib using:\n'
                      '  pip install matplotlib\n'
                      'or\n'
                      '  conda install matplotlib\n'
                      'and then install again.')
        return
    
    # install all plotting styles in the matplotlib stylelib library
    
    cfgdir = mpl.get_configdir()
    stylelib = os.path.join(cfgdir, 'stylelib')
    if not os.path.exists(stylelib):
        os.mkdir(stylelib)
    styles_path = os.path.join(os.path.dirname(__file__), "scp_data", "stylesheets")
    styles = os.listdir(styles_path)
    for style in styles:
        src = os.path.join(styles_path, style)
        dest = os.path.join(stylelib, style)
        sh.copy(src, dest)


# def install_requires(dev=False):
#     import yaml
#     envyml = 'env/scpy{}.yml'.format('-dev' if dev else '')
#
#     with open(envyml, 'r') as f:
#         req = yaml.load(f)
#
#     for i, item in enumerate(req['dependencies']):
#         if '::' in item:
#             req['dependencies'][i] = req['dependencies'][i].split('::')[1]
#         if '>' not in item and '<' not in item and '=' in item:
#             req['dependencies'][i] = req['dependencies'][i].replace('=', '==')
#
#     print(req['dependencies'])
#     return req['dependencies']

def install_requires():
    return []


class PostDevelopCommand(_develop):
    """Post-installation for development mode."""
    
    def run(self):
        _develop.run(self)
        # for item in ['pre-commit', 'pre-push']:
        #     hook = os.path.join(path(), '.git', 'hooks', item)
        #     if os.path.exists(hook):
        #         os.remove(hook)
        #     nhook = os.path.join(path(), 'git_hooks', item)
        #     sh.copy(nhook, hook)
        #     print(('installation of `.git/hooks/{}` made.'.format(item)))
        install_styles()


class PostInstallCommand(_install):
    """Post-installation for installation mode."""
    
    def run(self):
        _install.run(self)
        install_styles()


def read(fname):
    with open(os.path.join(path(), fname), 'r') as f:
        return f.read()

def get_version():
    pass

# Data for setuptools
packages = []

extras_require = {
    'tests': [
        "pytest",
        "pytest-runner",
        "pytest-console-scripts",
        "scikit-image",
    ],
}

setup_args = dict(
    
    # packages informations
    name="spectrochempy",
    #use_scm_version=True,
    version = version.version,
    license="CeCILL-B",
    author="Arnaud Travert & Christian Fernandez",
    author_email="developpers@spectrochempy.fr",
    maintainer="SpectroChempy Developpers",
    maintainer_email="developpers@spectrochempy.fr",
    url='https://www.spectrochempy.fr',
    description='Processing, analysis and modelling Spectroscopic data for '
                'Chemistry with Python',
    long_description=read('README.rst'),
    long_description_content_type='text/x-rst',
    classifiers=["Development Status :: 3 - Alpha",
                 "Topic :: Utilities",
                 "Topic :: Scientific/Engineering",
                 "Topic :: Software Development :: Libraries",
                 "Intended Audience :: Science/Research",
                 "License :: CeCILL-B Free Software License Agreement (CECILL-B)",
                 "Operating System :: OS Independent",
                 "Programming Language :: Python :: 3.7",
                 ],
    platforms=['Windows', 'Mac OS X', 'Linux'],
    
    # packages discovery
    packages=find_packages() + packages,
    include_package_data=True,
    
    # requirements
    python_requires=">3.6",
    
    # post-commands
    cmdclass={'develop': PostDevelopCommand,
              'install': PostInstallCommand,
              },
    
    # entry points for scripts
    # scripts = {'scripts/launch_api.py'},
    entry_points={
        'console_scripts': ['scpy=spectrochempy.scripts.launch_api:main', ],
    },

)

# ======================================================================================================================
if __name__ == '__main__':
    # execute setup
    setup(**setup_args)
    
    # Install quaternion package (#TODO: may be we can make a single setup file)
    # from subprocess import check_output
    # out = check_output("""cd quaternion; python setup.py develop; cd .. """, shell=True)
    # print(out)
