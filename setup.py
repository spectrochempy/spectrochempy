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

import os
import shutil as sh
import subprocess
from warnings import warn

# get the version string
# -----------------------
def get_version():
    """Get the version string

    Returns
    -------
    version: str
        the version string such as  |version|
    release: str
        the release string such as  |release|
    """

    version = '0.1'
    release = '0.1'

    try:

        with open(os.path.expanduser("~/.spectrochempy/__VERSION__"), "r") as f:
            version = f.readline()
            release = f.readline()

    except IOError:

        with open(os.path.expanduser("~/.spectrochempy/__VERSION__"), "w") as f:
            f.write(version + "\n")
            f.write(release + "\n")

    finally:
        pass

    try:
        # get the version number (if we are using git)
        version_info = subprocess.getoutput("git describe")
        version_info = version_info.split('-')

        # in case of a just tagged version version str contain only one string
        if len(version_info) >= 2:  # case of minor revision
            version = "%s.%s" % tuple(version_info[:2])
            release = version_info[0]
        else:
            version = version_info[0]
            release = version

        with open(os.path.expanduser("~/.spectrochempy/__VERSION__"), "w") as f:
            f.write(version+"\n")
            f.write(release+"\n")

    except:
        warn('Could not get version string from GIT repository')

    finally:
        copyright = u'2014-2017, LCS - ' \
                    u'Laboratory for Catalysis and Spectrochempy'

        return version, release, copyright

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

