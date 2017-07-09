# -*- coding: utf-8; tab-width: 4; indent-tabs-mode: t; python-indent: 4 -*-
#
# =============================================================================
# Copyright (©) 2015-2017 LCS
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


from setuptools import setup, find_packages
from setuptools.command.develop import develop
# from setuptools.command.install import install
from spectrochempy.version import get_version

import os
import shutil as sh


class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        for item in ['pre-commit', 'pre-push', 'post-merge', 'post-commit',
                     'post-checkout']:
            if os.path.exists ('.git/hooks/{}'.format(item)):
                os.remove('.git/hooks/{}'.format(item))
            sh.copy('git_hooks/{}'.format(item), '.git/hooks/{}'.format(item))

            print('installation of `.git/hooks/{}` made.'.format(item))

        develop.run(self)

# class PostInstallCommand(install):
#     """Post-installation for installation mode."""
#     def run(self):
#         # PUT YOUR POST-INSTALL SCRIPT HERE or CALL A FUNCTION
#         install.run(self)

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()



def get_dependencies():
    with open("requirements.txt", 'r') as f:
        pkg = f.read().split("\n")
        while '' in pkg:
            pkg.remove('')
        for item in pkg:
            if item.startswith('#'):
                pkg.remove(item)
        return pkg


setup(
    name='spectrochempy',
    version=get_version()[1],
    packages=find_packages(),
    include_package_data=True,
    url='http:/www-lcs.ensicaen.fr/spectrochempy',
    license='CeCILL-2.1',
    author='Arnaud Travert & christian Fernandez',
    author_email='spectrochempy@ensicaen.fr',
    description='Spectra Analysis & Processing with Python',
    long_description=read('README.md'),
    setup_requires=['pytest-runner'],
    install_requires=get_dependencies(),
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
    cmdclass = {
               'develop': PostDevelopCommand,
                          # 'install': PostInstallCommand,
               },
)

