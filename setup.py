# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

from setuptools import setup, find_packages

import os

def path():
    return os.path.dirname(__file__)

def read(fname):
    with open(os.path.join(path(), fname), 'r') as f:
        return f.read()

setup_args = dict(
    name="spectrochempy",
    use_scm_version=True,
    license="CeCILL-B",
    author="Arnaud Travert & Christian Fernandez",
    author_email="contact@spectrochempy.fr",
    maintainer="SpectroChempy Developpers",
    maintainer_email="contact@spectrochempy.fr",
    url='https://www.spectrochempy.fr',
    description='Processing, analysis and modelling Spectroscopic data for Chemistry with Python',
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
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    python_requires=">=3.7",
    entry_points={
        'console_scripts': ['scpy=spectrochempy.scripts.launch_api:main', ],
    },
)

# ======================================================================================================================
if __name__ == '__main__':
    # execute setup
    setup(**setup_args)
