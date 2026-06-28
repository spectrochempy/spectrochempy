
:orphan:

What's New in Revision {{ revision }}
---------------------------------------------------------------------------------------

These are the changes in SpectroChemPy-{{ revision }}.
See :ref:`release` for a full changelog, including other versions of SpectroChemPy.

..
   Do not remove the ``revision`` marker. It will be replaced during doc building.
   Also do not delete the section titles.
   Add your list of changes between (Add here) and (section) comments
   keeping a blank line before and after this list.

.. section

New Features
~~~~~~~~~~~~
.. Add here new public features (do not delete this comment)

- SpectroChemPy now exposes direct top-level helpers for the built-in 1D line
  shapes: ``scp.gaussian(...)``, ``scp.lorentzian(...)``, ``scp.voigt(...)``,
  ``scp.asymmetricvoigt(...)``, and ``scp.sigmoid(...)``. This makes it easier
  to build synthetic profiles directly from the public API, including in
  gallery and notebook workflows. (#1301)

- ``scp.concatenate(..., axis=1)`` now supports promoting 1D datasets into
  column-wise 2D `NDDataset` results. This makes it easier to assemble profile
  or concentration matrices directly within the SpectroChemPy API.

- ``scp.stack(..., axis=1)`` now supports stacking 1D datasets as columns
  into a 2D `NDDataset`. This makes it easier to build workflow-style
  concentration or profile matrices without falling back to
  ``np.column_stack(...)`` followed by a manual `NDDataset` wrapper.

- Reading multi-object files (MATLAB ``.mat``, multi-subfile SPC, ZIP
  archives) now returns a list with convenience methods for selecting
  the dataset you need. After ``datasets = scp.read(...)``:

  - ``datasets.select_largest(ndim=2)`` picks the biggest 2D dataset.
  - ``datasets.select_by_name("spectra")`` picks the first dataset whose
    name contains the given word.
  - ``datasets.filter_by_ndim(2)`` keeps only 2D datasets.
  - ``datasets.filter_by_shape((80, 700))`` keeps only datasets of that
    exact shape.
  - ``datasets.names`` lists all dataset names at a glance.

  This makes it much easier to work with files that contain multiple
  variables or spectra. (#1306)

- The generic ``scp.read(...)`` API and the main explicit reader entry points
  now document the namespace-based import convention more clearly, including
  when readers may return a list-like ``ScpObjectList`` and which helper
  methods are available to select datasets from multi-object imports.


.. section

Bug Fixes
~~~~~~~~~
.. Add here new bug fixes (do not delete this comment)

- ``PLSRegression`` now works with a 1D ``NDDataset`` as the response variable
  ``y`` (shape ``(n_obs,)``). Previously, the ``_set_output`` coordinate wrapping
  decorator assumed the metadata source was always 2D, causing a ``ValueError``
  on ``predict()``, ``y_scores``, ``y_loadings``, ``y_weights``, ``y_rotations``
  and ``result`` when fitting with a 1D target.  Fixes the ``coef`` property
  coordinate assignment which incorrectly used ``self._Y.x`` instead of
  ``self._Y.y`` for the target dimension. (#1305)


.. section

Dependency Updates
~~~~~~~~~~~~~~~~~~
.. Add here new dependency updates (do not delete this comment)


.. section

Breaking Changes
~~~~~~~~~~~~~~~~
.. Add here new breaking changes (do not delete this comment)


.. section

Deprecations
~~~~~~~~~~~~
.. Add here new deprecations (do not delete this comment)


.. section

Developer
~~~~~~~~~
.. Add here developer changes (do not delete this comment)
