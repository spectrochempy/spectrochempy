
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

- SpectroChemPy now exposes top-level helpers for common 1D line shapes:
  ``scp.gaussian(...)``, ``scp.lorentzian(...)``, ``scp.voigt(...)``,
  ``scp.asymmetricvoigt(...)``, and ``scp.sigmoid(...)``. The line-shape
  helpers except ``scp.sigmoid`` also accept ``normalized=False`` to return a
  profile whose peak amplitude is exactly ``ampl``. Common mathematical
  helpers are also available at top level: ``scp.exp(...)``, ``scp.log(...)``,
  ``scp.log10(...)``, ``scp.sin(...)``, and ``scp.cos(...)``. (#1301)

- ``stack(..., axis=1)`` is now supported for stacking 1D profiles as
  columns into a 2D dataset.  This makes the workflow for building
  synthetic concentration or profile matrices fully native within the
  SpectroChemPy API, without falling back to ``np.column_stack(...)``
  or manual `NDDataset` wrapping.

- New preprocessing operations in `spectrochempy.processing.transformation`:
  ``normalize()``, ``center()``, ``autoscale()``, ``snv()``, ``msc()``,
  ``pareto_scale()``, ``range_scale()``, ``robust_scale()``, and
  ``log_transform()``.
  These implement standard chemometric scaling and scatter-correction steps
  as first-class NDDataset methods, removing the need for manual NumPy
  arithmetic in notebooks.

- Added stateful transformer classes for ML workflows:
  ``CenterTransformer``, ``AutoscaleTransformer``, and ``SNVTransformer``.
  Each implements ``fit()``, ``transform()``, ``fit_transform()``, and
  ``inverse_transform()``, allowing learned statistics (e.g., mean and std
  from a training set) to be reused safely on test data or new batches.
  They complement the existing procedural API and are registered as
  top-level API objects and NDDataset methods.

- 2D ``plot(method="lines"/"stack")`` now automatically uses coordinate labels
  as matplotlib line labels, so that ``ax.legend()`` shows meaningful names
  without needing to pass labels explicitly.  Legend entries are displayed
  in natural (first-to-last) order. (#1320)

- Reading multi-object files such as MATLAB ``.mat`` files, multi-subfile SPC
  files, and ZIP archives now returns a list-like result with helper methods
  for selecting datasets by size, name, dimensionality, or shape. (#1306)


.. section

Bug Fixes
~~~~~~~~~
.. Add here new bug fixes (do not delete this comment)

- Fixed scatter plotting regressions introduced after 0.8.2:
  ``plot_multiple(..., method="scatter")`` now shows markers again,
  single-dataset ``plot_multiple`` calls preserve the requested plotting method,
  and ``plot(scatter=True)`` once again selects scatter-style plotting.

- Fixed several plotting API inconsistencies: ``multiplot()`` now preserves the
  requested plotting method for single datasets and accepts ``nrows`` /
  ``ncols`` aliases; 1D artists now honor ``alpha``, ``markeredgewidth``, and
  ``mew``; 2D contour-style plots accept ``alpha`` and ``levels`` consistently;
  ``use_plotly=True`` now fails with a clear error when Plotly support is
  unavailable; and legacy ``lines`` / ``pen`` aliases continue to work across
  dimensional fallbacks.

- Fixed several processing subpackage bugs: multi-dimensional ZPD detection in interferogram apodization now uses the median of per-row argmax positions; ``rs()``, ``ls()``, and ``roll()`` now pass ``axis=-1`` to ``np.roll`` for proper multi-dimensional shifting; ``denoise()`` guard now checks ``ndim != 2`` instead of the incorrect combined condition, and the ratio display factor is corrected; and ``npy.dot()`` ``isinstance`` check now tests ``b`` instead of ``a`` twice. (#xxx)

- ``PLSRegression`` now works with a 1D ``NDDataset`` as the response variable
  ``y``. This fixes failures in ``predict()``, ``y_scores``, ``y_loadings``,
  ``y_weights``, ``y_rotations``, ``result``, and ``coef`` when fitting with a
  1D target. (#1305)


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

- Centralized internal plotting kwargs normalization in a private helper
  module to reduce duplication across plotting entry points while preserving
  public plotting aliases and rendering behavior.

- Centralized internal plotting method normalization in a private helper
  module to reduce duplication across backend dispatch, multiplot handling,
  and 1D/2D fallback validation, without changing the public plotting API.
