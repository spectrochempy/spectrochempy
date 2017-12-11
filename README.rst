.. -_\- coding: utf-8 -_-

Spectrochempy
=============

What is |scp|?
--------------

|scp| is a framework for processing, analysing and modelling **Spectro**\ scopic
data for **Chem**\ istry with **Py**\ thon. It is is a cross platform software,
running on Linux, Windows or OS X.

#.  A ``NDDataset`` object embedding array of data with labeled axes and
    metadata.
#.  A ``Project`` manager to work on multiple ``NDDataset`` simultaneoulsly.
#.  ``Units`` and ``Uncertainties`` for ``NDDataset``.
#.  Mathematical operations over ``NDDataset`` such addition,
    multiplication and many more ...
#.  Import functions to read data from experiments or modelling programs ...
#.  Display functions such as ``plot`` ...
#.  Export functions to ``csv``, ``xls`` ... (NOT YET)
#.  Preprocessing funtions such as baseline correction, automatic
    subtraction and many more ...
#.  Fitting capabilities for single or multiple datasets ...
#.  Exploratory analysis such as ``svd``, ``pca``, ``mcr_als``, ``efa``...


.. warning::

	|scp| is still experimental and under active development.
	Its current design is subject to major changes, reorganizations, bugs
	and crashes!!!. Please report any issues to the `Issue Tracker <https://bitbucket.org/spectrocat/spectrochempy/issues>`_


.. _main_installation:

Installation
============

Installation can be performed using conda, pip or by cloning the sources from Bitbucket:

.. sourcecode:: bash

	conda install spectrochempy  (NOT YET!)

or

.. sourcecode:: bash

	pip install spectrochempy     (NOT YET!)


License
=======

Note::

	Fix the correct license to use. Not sure Cecill 2.1 is compatible
	with BSD-3 code sometimes included (such as NMRGlue)

Documentation
===============

the online Html documentation is available here:

* `HTML documentation <http://www-lcs.ensicaen.fr/cfnews/spectrochempy/html/>`_


Issue Tracker
==============

You find a problem, want to suggest enhancement or want to look at the current issues and milestones, you can go there:

* `Issue Tracker  <https://bitbucket.org/spectrocat/spectrochempy/issues>`_


.. _main_citing :

Citing |scp|
============

When using |scp| for your own work, you are kindly requested to cite it this
way::

     Arnaud Travert & Christian Fernandez,
     SpectroChemPy, a framework for processing, analysing and modelling of Spectroscopic data for Chemistry with Python
     https://bitbucket.org/spectrocat/spectrochempy, (version 0.1)
     Laboratoire Catalyse and Spectrochemistry, ENSICAEN/Universit\'e de Caen/CNRS, 2017



.. |scp| replace:: **SpectroChemPy**




