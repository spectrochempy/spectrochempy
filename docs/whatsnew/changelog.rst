
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

* `plot_multiple` method now accept keyword arguments to change the default
  plot style of the different spectra. See :ref:`plot_multiple` for details.
* `BaselineCorrection` class has been renamed into
  `Baseline`, and there are changes in the way it
  is now used. It allows to perform baseline correction
  on a dataset with multiple algorithms. See :ref:`baseline` for details. BaselineCorrection is still valid but deprecated.
* Three new baseline algorithms have been added to the new Baseline processor:
  `~spectrochempy.rubberband`, `~spectrochempy.asls` and `~spectrochempy.snip` .
* Filters has been refactored. A new `Filter` processor class allows to define various
  filters and apply them to a dataset. See `Filtering and Smoothing` tutorials and `Filter`
  for details. Note: Backward compatibility is ensured with the previous `smooth` and `savgol_filter` methods.
* A `whittaker` filter has been added to the `Filter` processor class. See `Filtering and Smoothing`
  tutorials and `Filter` for details.
* A `denoise` method based on PCA analysis has been added which allows to apply a denoising filter to a 2D dataset.
* A `despike` method has been added to the `Filter` processor class.
  It allows to remove spikes from a 1D or 2D dataset. This close issues #688.

.. section

Bug fixes
~~~~~~~~~
.. Add here new bug fixes (do not delete this comment)

* Docs problems fixed (#687).

.. section

Breaking changes
~~~~~~~~~~~~~~~~
.. Add here new breaking changes (do not delete this comment)


.. section

Deprecations
~~~~~~~~~~~~
.. Add here new deprecations (do not delete this comment)

* `parameters` method of Analysis configurables is now deprecated in favor of `params`.
* The BaselineCorrection processor has been deprecated in favor of Baseline.
* `abc` (and its alias `ab`) method has been deprecated in favor of `basc`.
