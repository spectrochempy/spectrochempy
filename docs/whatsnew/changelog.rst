
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
* Two new baseline algorithms have been added: `asls` and `snip`. See :ref:`Baseline` for details.
* Filters has been refactored. A new `Filter` processor class allows to define various
  filters and apply them to a dataset. See `Filtering and Smoothing` tutorials and `Filter`
  for details. Note: Backward compatibility is ensured with the previous `smooth` and `savgol_filter` methods.

* A `whittaker` filter has been added to the `Filter` processor class. See `Filtering and Smoothing`
  tutorials and `Filter` for details.

* A new `Baseline` processor class has been added. It allows to perform baseline correction
  on a dataset with multiple algorithms. See :ref:`Baseline` for details.

.. section

Bug fixes
~~~~~~~~~
.. Add here new bug fixes (do not delete this comment)


.. section

Breaking changes
~~~~~~~~~~~~~~~~
.. Add here new breaking changes (do not delete this comment)

* `BaselineCorrection` class has been renamed into
  `Baseline`, and there are changes in the way it
  is now used. It allows to perform baseline correction
  on a dataset with multiple algorithms. See :ref:`baseline` for details.

* `abc` (and its alias `ab`) method has been removed in favor of `basc`.

.. section

Deprecations
~~~~~~~~~~~~
.. Add here new deprecations (do not delete this comment)
