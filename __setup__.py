# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
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

def path():
	return os.path.dirname(__file__)

class PostDevelopCommand(develop):
	"""Post-installation for development mode."""
	def run(self):
		develop.run(self)
		for item in ['pre-commit', 'pre-push']:
			hook = os.path.join(path(), '.git', 'hooks', item)
			if os.path.exists(hook):
				os.remove(hook)
			nhook = os.path.join(path(), 'git_hooks', item)
			sh.copy(nhook, hook)
			print(('installation of `.git/hooks/{}` made.'.format(item)))

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        # PUT YOUR POST-INSTALL SCRIPT HERE or CALL A FUNCTION
        install.run(self)

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
			#version=version,
            use_scm_version=True,
			packages=find_packages(
					exclude=['docs', "*.tests", "*.tests.*", "tests.*",
							 "tests"]),
			include_package_data=True,
			url='http:/www-lcs.ensicaen.fr/spectrochempy',
			license='CeCILL-B',
			author='Arnaud Travert & christian Fernandez',
			author_email='spectrochempy@ensicaen.fr',
			description='Spectra Analysis & Processing with Python',
			long_description=read('README.rst'),
			setup_requires=['setuptools_scm'],
			install_requires=get_dependencies(),
			dependency_links=[
			],
			classifiers=[
				"Development Status :: 4 - Beta",
				"Topic :: Utilities",
				"Topic :: Scientific/Engineering",
				"Intended Audience :: Science/Research",
				"License :: CeCILL-B Free Software License Agreement (CECILL-B)",
				"Operating System :: OS Independent",
				"Programming Language :: Python :: 3.5",
			],
			cmdclass={
				'develop': PostDevelopCommand,
				'install': PostInstallCommand,
			},
	)


if __name__ == '__main__':
	run_setup()
