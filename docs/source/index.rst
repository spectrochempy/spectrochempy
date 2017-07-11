.. _main:


Welcome to the |scp| documentation!
###################################

:release: |release|
:version: |version| (|today|)

.. contents::

.. _main_introduction:

Introduction
============

What is |scp|?
--------------

|scp| is a framework for processing, analysing and modelling **Spectro**\ scopic
data for **Chem**\ istry with **Py**\ thon. It is is a cross platform software,
running on Linux, Windows or OS X.

  1.  A |NDDataset| object embedding array of data with labeled axes and metadata.
  2. `Units` and `Uncertainties` for |NDDataset| and |NDDataset| axes.
  3.  Mathematical operations over |NDDataset| such addition, multiplication and many more ...
  4.  Import functions to read data from experiments or modelling programs ...
  5.  Display functions such as ``print``, :meth:`plot` ...
  6.  Export functions to `csv`, `xls` ...
  7.  Preprocessing funtions such as baseline correction, automatic subtraction and many more ...
  8.  Exploratory analysis such as ``svd``, ``pca``, ``mcr_als``, ``efa``...


.. warning::

	|scp| is still experimental and under active development.
	Its current design is subject to major changes, reorganizations, bugs and crashes!!!.

.. _main_intallation:

Installation
============

.. toctree::
    :maxdepth: 1

    main/install
    main/license


.. _main_documentation:

Documentation
===============

Documentation is available here:

* `https://www-lcs.ensicaen.fr/cfnews/spectrochempy/html/ <https://www-lcs.ensicaen.fr/cfnews/spectrochempy/html/>`_

and through docstrings provided with the code.

.. _main_user_guides:

User Guide and Tutorials
=========================

Tutorial pages and Jupyter notebooks with running examples are also available.

.. toctree::
    :maxdepth: 1

    api/userguide
    main/faq
    api/auto_examples/index

.. _main_issue_tracker:

Issue Tracker
==============

You find a problem, want to suggest enhancement, or want to look at the current issues and milestones, you can go there:

* `Issue Tracker  <https://bitbucket.org/spectrocat/spectrochempy/issues>`_

.. _main_citing :

Citing |scp|
============

When using |scp| for your own work, you are kindly requested to cite it this
way::

     Arnaud Travert & Christian Fernandez,
     SpectroChemPy, a framework for processing, analysing and modelling of Spectroscopic data for Chemistry with Python
     https://bitbucket.org/spectrocat/spectrochempy, (version 0.1)
     Laboratoire Catalyse and Spectrochemistry, ENSICAEN/Université de Caen/CNRS, 2017

Credits
----------
.. toctree::
    :maxdepth: 1

    main/credits


.. _main_developper:

Developper's Documentation
==========================

.. toctree::
    :maxdepth: 1

    dev/develguide
    dev/reference

See also
=========

.. toctree::
    :maxdepth: 1

    main/seealso

Changes in the last release
=============================
.. toctree::
    :maxdepth: 1

    main/changelog






