.. _develguide:

Developer Guide
###############

Installing a developper version
===============================

The best to proceed with development is that we (the developers) all have a similar environment for each python version.

The master is build on the 3.6 python version. 

1) Install anaconda (website)

2) issue the following command to add the conda-forge channel:

	conda config --add channels conda-forge


3) create an environment called **scp_36**

	conda create --name scp_36 python=3.6


4) switch to this environment

    source activate scp_36


You can make it permanent by putting this command in you bash_profile (MAC)

















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
