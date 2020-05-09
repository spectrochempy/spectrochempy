.. _release_guide:

Instructions to release a new version
=====================================

(this is a tentative guide describing the necessary step to release a new version of |scpy|.)

**Under  work**

Versioning scheme
-----------------

versioning scheme follows the rule described here : [Semantic versioning](https://semver.org).

Given a version number MAJOR.MINOR.PATCH, increment the:

* MAJOR version when a new version make incompatible API changes,

* MINOR version when new functionalities are added in a backwards compatible manner

* PATCH version when backwards compatible bug fixes are made.

* Additional labels for pre-release and build metadata are available as extensions to the MAJOR.MINOR.PATCH format.

  Here we will use : beta and rc.# labels

  Additionally for development version the label dev.# where the number # is incremented at each commit.

Releasing a new version
------------------------

* This require a clean working directory. A new version is always based on the master branch.

  1. commit the last changes on master.

  2. push on origin/master

  3. Tag the last commit on master with the new version number

  4. test this version

  5. install it in development mode:
     
     pip install -e .  (this is necessary to update the development version number in the eggs)

  6. checkout on Master - make documentation for stable version

* update PyPi

  # Twine:

    if not installed, install wine:

    conda install twine

  # run the following commands:

    On both mac and on window:

    conda update pip setuptools wheel twine
    python setup.py sdist bdist_wheel

    TODO add linux

  # upload to PyPi


    twine upload --repository pypi dist/*




