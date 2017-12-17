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

#.  A |NDDataset| object embedding array of data with coordinates and metadata.
#.  A |Project| manager to work on multiple |NDDataset| simultaneoulsly.
#.  `Units` and `Uncertainties` for |NDDataset| and NDDataset coordinates
    |Coord|.
#.  Mathematical operations over |NDDataset| such addition, multiplication
    and many more ...
#.  Import functions to read data from experiments or modelling programs ...
#.  Display functions such as ``plot`` ...
#.  Export functions to ``csv``, ``xls`` ... (NOT YET)
#.  Preprocessing functions such as baseline correction, masking bad data,
    automatic subtraction and many more ...
#.  Exploratory analysis such as ``svd``, ``pca``, ``mcr_als``, ``efa``...

.. warning::

    |scp| is still experimental and under active development.
    Its current design is subject to major changes, reorganizations,
    bugs and crashes!!!. Please report any issues to the
    `Issue Tracker  <https://bitbucket.org/spectrocat/spectrochempy/issues>`_


.. _main_intallation:

Installation
============

.. toctree::
    :maxdepth: 2

    install


.. _main_license:

License
=======

.. toctree::
    :maxdepth: 2

    CeCILL-B FREE SOFTWARE LICENSE AGREEMENT <license>


.. _main_documentation:

Documentation
===============

Last updated Html documentation is always available there:

* `http://www-lcs.ensicaen.fr/cfnews/spectrochempy/html/ <http://www-lcs.ensicaen.fr/cfnews/spectrochempy/html/>`_

Pdf documentation can also be downloaded there:

* `spectrochempy.pdf <http://www-lcs.ensicaen.fr/cfnews/spectrochempy/pdf/spectrochempy.pdf>`_


.. _main_user_documentation:

User Documentation
==================

.. toctree::
    :maxdepth: 2

    user/index


.. _gallery_of_examples:

Gallery of examples
=====================

.. toctree::
    :maxdepth: 2

    auto_examples/index


.. _main_issue_tracker:

Issue Tracker
==============

You find a problem, want to suggest enhancement or want to look at the current issues and milestones, you can go there:

* `Issue Tracker  <https://bitbucket.org/spectrocat/spectrochempy/issues>`_


Developper documentation
========================
.. _main_developper:


.. toctree::
    :maxdepth: 2

    dev/index


.. _main_see_also:

See also
=========

.. toctree::
    :maxdepth: 2

    seealso


.. _main_changelog:

Changes in the last release
=============================
.. toctree::
    :maxdepth: 2

    changelog


.. _main_citing:

Citing |scp|
============

When using |scp| for your own work, you are kindly requested to cite it this
way::

     Arnaud Travert & Christian Fernandez,
     SpectroChemPy, a framework for processing, analysing and modelling of
     Spectroscopic data for Chemistry with Python
     https://bitbucket.org/spectrocat/spectrochempy, (version 0.1a)
     Laboratoire Catalyse and Spectrochemistry,
     ENSICAEN/Université de Caen/CNRS, 2017


.. _main_credits:

Credits
=======
.. toctree::
    :maxdepth: 2

    credits




