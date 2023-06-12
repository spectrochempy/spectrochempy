:orphan:

What's new in revision 0.6.6.dev
---------------------------------------------------------------------------------------

These are the changes in SpectroChemPy-0.6.6.dev.
See :ref:`release` for a full changelog including other versions of SpectroChemPy.

New features
~~~~~~~~~~~~

* `plot_multiple` method now accept keyword arguments to change the default
  plot style of the different spectra. See :ref:`plot_multiple` for details.
* Two new baseline algorithms have been added: `asls` and `snip`. See :ref:`Baseline` for details.
* Filters has been refactored. A new `Filter` processor class allows to define various
  filters and apply them to a dataset. See `Filtering and Smoothing` tutorials and `Filter`
  for details. Note: Backward compatibility is ensured with the previous `smooth` and `savgol_filter` methods.
* A `whittaker` filter has been added to the `Filter` processor class. See `Filtering and Smoothing`
  tutorials and `Filter` for details.
* A new `Baseline` processor class has been added. It allows to perform baseline correction
  on a dataset with multiple algorithms. See :ref:`Baseline` for details.

Bug fixes
~~~~~~~~~

* #687 fixed.

Breaking changes
~~~~~~~~~~~~~~~~

* `BaselineCorrection` class has been renamed into
  `Baseline`, and there are changes in the way it
  is now used. It allows to perform baseline correction
  on a dataset with multiple algorithms. See :ref:`baseline` for details.

* `abc` (and its alias `ab`) method has been removed in favor of `basc`.

Deprecations
~~~~~~~~~~~~

* `parameters` method of Analysis configurables is now deprecated in favor of `params`.
