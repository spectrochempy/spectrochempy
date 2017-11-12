# -*- coding: utf-8; tab-width: 4; indent-tabs-mode: t; python-indent: 4 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
# This software is a computer program whose purpose is to [describe
# functionalities and technical features of your software].
#
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty and the software's author, the holder of the
# economic rights, and the successive licensors have only limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading, using, modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean that it is complicated to manipulate, and that also
# therefore means that it is reserved for developers and experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and, more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
# =============================================================================

# this is a hidden setup file (that can be tested in debugged mode with pytest)
#
from setuptools import setup, find_packages
from setuptools.command.develop import develop
from setuptools.command.install import install

import os
import subprocess
import shutil as sh
import warnings

from spectrochempy.api import version

def path():
    return os.path.dirname(__file__)


class PostDevelopCommand(develop):
    """Post-installation for development mode."""

    def run(self):

        develop.run(self)

        for item in ['pre-commit','pre-push']:
            hook = os.path.join(path(), '.git', 'hooks', item)
            if os.path.exists(hook):
                os.remove(hook)
            nhook = os.path.join(path(), 'git_hooks', item)
            sh.copy(nhook, hook)

            print(('installation of `.git/hooks/{}` made.'.format(item)))


def read(fname):
    with open(os.path.join(path(), fname), 'r') as f:
        return f.read()


def get_dependencies():
    with open(os.path.join(path(), "requirements.txt"), 'r') as f:
        pkg = f.read().split("\n")
        while '' in pkg:
            pkg.remove('')
        for item in pkg:
            if item.startswith('#'):
                pkg.remove(item)
        # found a problem during pip install with pyqt (works when
        # replaced by PyQt5)
        pkg = ['PyQt5' if item.strip() == 'pyqt' else item for item in pkg]

        return pkg


def run_setup():
    setup(
            name='spectrochempy',
            version=version,
            packages=find_packages(
                    exclude=['docs', "*.tests", "*.tests.*", "tests.*",
                             "tests"]),
            include_package_data=True,
            url='http:/www-lcs.ensicaen.fr/spectrochempy',
            license='CeCILL-2.1',
            author='Arnaud Travert & christian Fernandez',
            author_email='spectrochempy@ensicaen.fr',
            description='Spectra Analysis & Processing with Python',
            long_description=read('README.rst'),
            setup_requires=['pytest-runner'],
            install_requires=get_dependencies(),
            dependency_links=[
                "git+ssh://git@github.com:sphinx-gallery/sphinx-gallery.git",
            ],
            tests_require=['pytest'],
            classifiers=[
                "Development Status :: 2 - Pre-Alpha",
                "Topic :: Utilities",
                "Topic :: Scientific/Engineering",
                "Intended Audience :: Science/Research",
                "License :: OSI Approved :: CEA CNRS Inria Logiciel Libre License, version 2.1 (CeCILL-2.1)",
                "Operating System :: OS Independent",
                "Programming Language :: Python :: 3.5",
            ],
            cmdclass={
                'develop': PostDevelopCommand,
                # 'install': PostInstallCommand,
            },
    )


if __name__ == '__main__':
    run_setup()
