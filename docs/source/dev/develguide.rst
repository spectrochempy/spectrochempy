.. _develguide:

Developer Guide
###############

Installing a developper version
===============================

The best to proceed with development is that we (the developers) all have a similar environment for each python version.

The master is build on the 3.6 python version. 

1. Install anaconda3 or miniconda3

	* OSX : `miniconda3 <https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh>`_

2) issue the following command to be sure to have add the conda-forge channel (on top priority):

	conda config --add channels conda-forge


3) list the current environments

    conda info --envs


4) create an environment called **scp36**

	conda create -n scp36 python=3.6 --file requirements.txt


5) switch to this environment

    source activate scp36


You can make it permanent by putting this command in you bash_profile (MAC)


6) install the additional librairies with conda

    conda install --yes --file requirements.txt



Setup PyCharm
--------------
open preferences.

set the an interpreter call scp_36 corresponding to the path:

/anaconda/envs/scp_36/bin/python

Set the









jupyter qtconsole




Here are two useful command to use, either to export or import a shared environment:

Export our current environment::

	conda env export > myenvironment.yml

Import (create) an environment from file specifications of anather developper::

	conda env create -f myenvironment.yml

Once you environment is installed, you must activate it:

Linux, OS X ::

	source activate <environment>

Windows ::

	activate <environment>


Packaging
=========

for information about packaging, see::

	http://python-packaging.readthedocs.io/en/latest/index.html

conda config --add channels conda-forge

Building the documentation
==========================
pip install sphinx-gallery

One need to install pandoc : https://github.com/jgm/pandoc/releases
