
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
