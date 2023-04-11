.. _userguide.objects:

Data structures
###############


The NDDataset is the main object used by `SpectroChemPy` .

Like numpy ndarrays, NDDataset have the capability to be sliced, sorted and subjected to mathematical operations.

But, in addition, NDDataset may have units, can be masked and each dimension can have coordinates (Coord) also with
units.
This make NDDataset aware of unit compatibility, *e.g.,*, for binary operation such as additions or subtraction or
during the application of mathematical operations.

In addition or in replacement of numerical data for coordinates, NDDataset can also have labeled coordinates where
labels can be different kinds of objects (strings, datetime, numpy nd.ndarray
or other NDDatasets, etc.).

This offers a lot of flexibility in using NDDatasets that, we hope, will be useful for the users.

.. todo::

    **SpectroChemPy** (will soon) provides another kind of data structure,
    aggregating several datasets: **NDPanel**. [*Under construction*]

Finally, **Project** is a structure that allows aggregating NDDatasets, NDPanels and processing scripts (Script) for a better management of complex set of spectroscopic data.

NDDataset
*********

.. toctree::
   :glob:
   :maxdepth: 2

   dataset/dataset

Project
*******

.. toctree::
   :glob:
   :maxdepth: 2

   project/project

Script
******

.. todo::

   To be documented
