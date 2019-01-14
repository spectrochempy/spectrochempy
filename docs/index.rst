.. _main:

Welcome to the |scpy| documentation!
#####################################

:version: |version| (|today|)

.. warning::

    This software is not yet publicly released.

    It is expected to be released in june or july 2019.

Introduction
============

What is |scpy|?
----------------

|scpy| is a framework for processing, analysing and modelling **Spectro**\ scopic
data for **Chem**\ istry with **Py**\ thon. It is is a cross platform software,
running on Linux, Windows or OS X.

#.  A |NDDataset| object embedding array of data with coordinates and metadata.
#.  A |Project| manager to work on multiple |NDDataset| simultaneoulsly.
#.  `Units`` and ``Uncertainties`` for |NDDataset| and NDDataset coordinates
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

    |scpy| is still experimental and under active development.
    Its current design is subject to major changes, reorganizations,
    bugs and crashes!!!. Please report any issues to the
    `Issue Tracker  <https://bitbucket.org/spectrocat/spectrochempy/issues>`_


Installation
============

.. toctree::
    :maxdepth: 2

    install


License
=======

.. toctree::
    :maxdepth: 2

    CeCILL-B FREE SOFTWARE LICENSE AGREEMENT <license>


Documentation
=============

Last updated Html documentation is always available there:

* `http://www.spectrochempy.fr <http://www.spectrochempy.fr>`_

.. Pdf documentation can also be downloaded there:


User Documentation
------------------

.. toctree::
    :maxdepth: 2

    user/userguide/index
    user/api/generated/index
    user/auto_examples/index
    user/faq


Developper documentation
------------------------

.. toctree::
    :maxdepth: 2

    dev/index


Issue Tracker
=============

You find a problem, want to suggest enhancement or want to look at the current
issues and milestones, you can go there:

* `Issue Tracker  <https://bitbucket.org/spectrocat/spectrochempy/issues>`_


See also
=========

.. toctree::
    :maxdepth: 2

    seealso


Changes in the last release
===========================
.. toctree::
    :maxdepth: 2

    changelog


Citing |scpy|
=============

When using |scpy| for your own work, you are kindly requested to cite it this
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




