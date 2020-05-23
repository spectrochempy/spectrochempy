###############
Spectrochempy
###############

+----------------------------+-------------------+
| Version STABLE (master)    |   |travis_master| |
+----------------------------+-------------------+
| Version DEV (develop)      |   |travis_develop||
+----------------------------+-------------------+

What is |scpy|?
=================

|scpy| is a framework for processing, analyzing and modeling **Spectro**\ scopic data for **Chem**\ istry with **Py**\ thon.
It is a cross platform software, running on Linux, Windows or OS X.

Among its major features:

#.  A ``NDDataset`` object embedding array of data with labeled axes and metadata.
#.  A ``Project`` manager to work on multiple ``NDDataset`` simultaneously.
#.  Physical ``Units`` for ``NDDataset``.
#.  Mathematical operations over ``NDDataset`` such addition, multiplication and many more ...
#.  Import functions to read data from experiments or modeling programs ...
#.  Display functions such as ``plot`` for 1D or nD datasets ...
#.  Export functions to *csv*, *xls* formats ...
#.  Preprocessing functions such as baseline correction, automatic subtraction and many more ...
#.  Fitting capabilities for single or multiple datasets ...
#.  Exploratory analysis such as ``SVD``, ``PCA``, ``MCR_ALS``, ``EFA`` ...


.. warning::

    |scpy| is still experimental and under active development. Its current design is subject to major changes,
    reorganizations, bugs and crashes!!!. Please report any issues to the
    `Issue Tracker <https://redmine.spectrochempy.fr/projects/spectrochempy/issues>`_


Documentation
===============

the online Html documentation is available here:

* `HTML documentation <https://www.spectrochempy.fr>`_


Installation
==============

* Follow the instructions here: `Installation guide <https://www.spectrochempy.fr/stable/gettingstarted/install/index.html>`_


Help
====

You can ask for help `here <https://redmine.spectrochempy.fr/projects/spectrochempy/boards>`_

Examples, tutorials
====================

The notebooks corresponding to the documentation are `located here. <https://www.spectrochempy.fr>`_

Issue Tracker
===============

You find a problem, want to suggest enhancements or want to look at the current issues and milestones, you can go there:

* `Issue Tracker  <https://redmine.spectrochempy.fr/projects/spectrochempy/issues>`_


Road Map
==========

The roadmap for this project is `here. <https://redmine.spectrochempy.fr/projects/spectrochempy/roadmap>`_


Citing |scpy|
===============

When using |scpy| for your own work, you are kindly requested to cite it this way::

     Arnaud Travert & Christian Fernandez, (2020) SpectroChemPy (Version 0.1). Zenodo. http://doi.org/10.5281/zenodo.3823841

.. |scpy| replace:: **SpectroChemPy**

Source repository
===================

The source are versioned using the git system and hosted on the `GitHub platform <https://github.com/spectrochempy/spectrochempy>`_

License
=========

`CeCILL-B FREE SOFTWARE LICENSE AGREEMENT <(https://cecill.info/licences/Licence_CeCILL-B_V1-en.html)>`_


.. |travis_master|  image:: https://travis-ci.com/spectrochempy/spectrochempy.svg?branch=master
   :target: https://travis-ci.com/spectrochempy/spectrochempy

.. |travis_develop|  image:: https://travis-ci.com/spectrochempy/spectrochempy.svg?branch=develop
   :target: https://travis-ci.com/spectrochempy/spectrochempy
