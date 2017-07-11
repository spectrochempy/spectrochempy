.. _develguide:

Developer Guide
###############

Installing a developper version
===============================

The best to proceed with development is that we all have a similar environment for
each python version.

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
