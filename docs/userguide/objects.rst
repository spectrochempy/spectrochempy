.. _userguide.objects:

Data Structures
###############


The NDDataset is the main object used by |scpy|.

Like numpy ndarrays, NDDataset have the capability to be sliced, sorted and subjected to mathematical operations.

But, in addition, NDDataset may have units, can be masked and each dimensions can have coordinates (Coord) also with
units.
This make NDDataset aware of unit compatibility, *e.g.*, for binary operation such as additions or subtraction or
during the application of mathematical operations.

In addition or in replacement of numerical data for coordinates, NDDataset can also have labeled coordinates where
labels can be different kind of objects (strings, datetime, numpy nd.ndarray
or othe NDDatasets, etc...).

This offers a lot of flexibility in using NDDatasets that, we hope, will be useful for the users.

.. todo::

    **SpectroChemPy** (will soon) provides another kind of data structure,
    aggregating several datasets: **NDPanel**. [*Under construction*]

Finally, **Project** is a structure that allow agregating NDDatasets, NDPanels and processing scripts (Script) for a better management of complex set of spectroscopic data.

NDDataset
*********

.. autosummary::
    :nosignatures:
    :toctree: reference/generated/

    spectrochempy.NDDataset

.. toctree::
   :glob:
   :maxdepth: 2

   dataset/dataset

Project
*******

.. autosummary::
    :nosignatures:
    :toctree: reference/generated/

    spectrochempy.Project

.. toctree::
   :glob:
   :maxdepth: 2

   project/project

Script
******

.. autosummary::
    :nosignatures:
    :toctree: reference/generated/

    spectrochempy.Script

.. todo::

   To be documented
