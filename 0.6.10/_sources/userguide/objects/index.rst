.. _userguide.objects:

Core objects
############

The `~spectrochempy.NDDataset` object is the main object used by `SpectroChemPy` .

Like a numpy `~numpy.ndarray`, a `~spectrochempy.NDDataset` have the capability to be sliced, sorted and subjected to mathematical operations.

But, in addition, `~spectrochempy.NDDataset` may have units, can be masked and each dimension can have coordinates (`~spectrochempy.Coord` ) also with
units.

This make NDDataset aware of unit compatibility, *e.g.,*, for binary operation such as additions or subtraction or
during the application of mathematical operations.

In addition or in replacement of numerical data for coordinates, `~spectrochempy.NDDataset` can also have labeled coordinates where
labels can be different kinds of objects (strings, datetime, `~numpy.ndarray`
or other `~spectrochempy.NDDataset` s, etc.).

This offers a lot of flexibility in using `~spectrochempy.NDDataset` s that, we hope, will be useful for the users.

Finally, **Project** is a structure that allows aggregating other `Project` s, `NDDataset` s, and processing scripts (`Script`)
for a better management of complex set of spectroscopic data.

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
