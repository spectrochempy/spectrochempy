.. _version_0.1.17:

Version 0.1.17
---------------------

Bugs fixed
~~~~~~~~~~~

* FIX #2 - Documentation: TQDM generate errors during doc building in examples.
* FIX #8 - Documentation: Tutorial notebooks that contain a dialog for filename do not run silently during sphinx build.
* FIX #13 - Documentation: Size of the figures in pdf documentation often too wide. 
* FIX #19 - Core code: loose coord  when slicing by integer array
* FIX #21 - Core code: Test Console don't pass on WINDOWS
* FIX #24 - Core code: pca reconstruction for an omnic dataset
* FIX #31 - Documentation: Fix doc RST syntax

Features added
~~~~~~~~~~~~~~~~

* #4 - Core code: Add a progress bar during loading of the library 
* #6 - Deployment: PyPi, Conda, ...: Automate building and deployment of new releases
* #7 - Deployment: PyPi, Conda, ...: make changelog automatic when making the doc
* #11 - Deployment: PyPi, Conda, ...: Check for new version at the program start up
* #15 - Core code: The autosub function does not return the subtraction coefficients
* #30 - Redmine website: Create an importer to get the issues from Bitbucket and start the issue tracker here.

Tasks terminated
~~~~~~~~~~~~~~~~~

* #9 - Environment: IPython, Jupiter, ...: QT error in doc
* #17 - Documentation: Fix doctrings and rst files  so that the pdf manual get correct with titles and sections
* #18 - Notebook Tutorials: import data: tutorial, examples, tests
* #22 - Deployment: PyPi, Conda, ...: Conda Recipe
* #33 - Redmine website: Redmine website configuration

