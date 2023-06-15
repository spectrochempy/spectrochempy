:orphan:

What's new in revision 0.6.6.dev
---------------------------------------------------------------------------------------

These are the changes in SpectroChemPy-0.6.6.dev.
See :ref:`release` for a full changelog including other versions of SpectroChemPy.

New features
~~~~~~~~~~~~

* `plot_multiple` method now accept keyword arguments to change the default
  plot style of the different spectra. See :ref:`plot_multiple` for details.
* `BaselineCorrection` class has been renamed into
  `Baseline`, and there are changes in the way it
  is now used. It allows to perform baseline correction
  on a dataset with multiple algorithms. See :ref:`baseline` for details. BaselineCorrection is still valid but deprecated.
* Three new baseline algorithms have been added to the new Baseline processor:
  `rubberband`, `asls` and `snip`. See :ref:`Baseline` for details.
* Filters has been refactored. A new `Filter` processor class allows to define various
  filters and apply them to a dataset. See `Filtering and Smoothing` tutorials and `Filter`
  for details. Note: Backward compatibility is ensured with the previous `smooth` and `savgol_filter` methods.
* A `whittaker` filter has been added to the `Filter` processor class. See `Filtering and Smoothing`
  tutorials and `Filter` for details.

Bug fixes
~~~~~~~~~

* Docs problems fixed (#687).

Deprecations
~~~~~~~~~~~~

* `parameters` method of Analysis configurables is now deprecated in favor of `params`.
* The BaselineCorrection processor has been deprecated in favor of Baseline.
* `abc` (and its alias `ab`) method has been deprecated in favor of `basc`.
