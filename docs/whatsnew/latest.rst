:orphan:

What's new in revision 0.6.8.dev
---------------------------------------------------------------------------------------

These are the changes in SpectroChemPy-0.6.8.dev.
See :ref:`release` for a full changelog including other versions of SpectroChemPy.

New features
~~~~~~~~~~~~

* Add Stejskal-Tanner kernel for 2D IRIS
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
* Traceback are now fully displayed when an error occurs in a script.
* Add the possibility to read Thermo high speed series files
* Add the possibility to use several sets of experimental conditions
  in `ActionMassKinetics` class.

Bug fixes
~~~~~~~~~

* Sorting coordinates now work with multi-coordinates axis.
* Fix a bug when concatenating datasets with multi-coordinates axis.
* Fix a bug in coordset definition for integration methods.
* Fix coordinates definitions in Analysis methods.
* Fix a bug in `write_csv` when the filename was provided as a string (issue #706)
* Fix issue #716
* Fix issue #714 : show versions of dependencies now working

Breaking changes
~~~~~~~~~~~~~~~~

* Changed the default QP solver (quadprog -> osqp): The new solver is compatible with
  python 3.11 and later. Fastness and robustness are improved. The quadprog solver can still be
  used if available
* Change the default value of the `whiten` parameter in the `FastICA` class to
  `unit-variance` instead of `arbitrary-variance` for compatibility with ScikitLearn
  1.3 and later
