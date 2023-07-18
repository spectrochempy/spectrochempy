
:orphan:

What's new in revision {{ revision }}
---------------------------------------------------------------------------------------

These are the changes in SpectroChemPy-{{ revision }}.
See :ref:`release` for a full changelog including other versions of SpectroChemPy.

..
   Do not remove the ``revision`` marker. It will be replaced during doc building.
   Also do not delete the section titles.
   Add your list of changes between (Add here) and (section) comments
   keeping a blank line before and after this list.


.. section

New features
~~~~~~~~~~~~
.. Add here new public features (do not delete this comment)

* Fancy indexing using location now supported.
* Add an example for NMR processing of a series of CP-MAS spectra.
* Add an example for processing NMR relaxation data
* Add an option to `read_topspin` to create ``y`` coordinates
  of pseudo-2D NMR spectra from a file (e.g. ``vdlist`` ).
* Add option to plot to add markers on curves
* Add a new method to the `Optimize` class to perform a least-square fitting. It is
  based on the `scipy.optimize.least_squares` function, allowing much faster operation
  for simple curve fitting
* Add the possibility to define user-defined functions in the `Optimize` class.

.. section

Bug fixes
~~~~~~~~~
.. Add here new bug fixes (do not delete this comment)

* Sorting coordinates now work with multi-coordinates axis.
* Fix a bug when concatenating datasets with multi-coordinates axis.
* Fix a bug in coordset definition for integration methods.
* Fix coordinates definitions in Analysis methods.
* Fix a bug in `write_csv` when the filename was provided as a string (issue #706)

.. section

Breaking changes
~~~~~~~~~~~~~~~~
.. Add here new breaking changes (do not delete this comment)


.. section

Deprecations
~~~~~~~~~~~~
.. Add here new deprecations (do not delete this comment)
