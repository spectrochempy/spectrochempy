
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

- Group update announcements for SpectroChemPy and installed plugins into native
  desktop notifications, with a text fallback and configurable delivery. Cache
  checks per Python environment and offer a manual version-status command. (:pr:`1900`)

- Add an IRIS tutorial covering the Langmuir kernel, regularization and
  interpretation of the CO adsorption example. (:pr:`1892`)

- Split one dataset axis into blocks with ``reshape(coord_policy='split')``.
- Compute and invert one-sided Fourier transforms along a dataset dimension with ``rfft`` and ``irfft``.
- ``NDDataset.to_markdown_report()`` now renders a deterministic Markdown
  summary of one explicitly supplied dataset, selected metadata, structured
  history and explicit figure references. It does not capture Matplotlib state
  or claim exhaustive provenance. Every recorded history operation is included
  in order; only individual values are explicitly truncated. Paths are rendered
  as supplied, and figure references accept relative paths and HTTP(S) URLs. (:pr:`1775`)

- IRIS can evaluate a fixed regularization grid in parallel with the OSQP
  builtin backend. Set ``workers`` to opt in. (:pr:`1887`)

.. section

Bug Fixes
~~~~~~~~~
.. Add here new bug fixes (do not delete this comment)

- Reject unknown despike methods and baseline wrapper keywords instead of
  silently ignoring them; correct filter defaults and baseline range help. (:pr:`1904`)

- Round nonlinear coordinates per value so small logarithmically spaced
  coordinates remain nonzero, while preserving linear-axis rounding.

- Renew update-notification reminder dates, refresh obsolete announcements,
  tolerate unavailable PyPI responses and unwritable caches, and skip checks
  during documentation builds. Use the PyPI Simple JSON API and filter releases
  by Python requirements, release channel and yanked status. (:pr:`1900`)

- Reject ambiguous inner alignment when coordinate rounding would merge
  samples within one dataset. (:pr:`1894`)

- Match each measurement to its coordinate in ``align()`` reference and
  outer joins. Preserve complex intensities and per-input labels, and
  mask interpolation results outside the source range or depending on
  masked samples. Interpolation can also use an explicit target coordinate
  along one dimension. (:pr:`1895`)

- Preserve the values at shared coordinates when aligning spectra with
  ``align(method="inner")``, including datasets containing a single spectrum. (:pr:`1893`)

- Preserve IRIS quadrature weights and regularization when the kernel grid
  is supplied in descending order. (:pr:`1892`)

- Correct the association of residual sums of squares (RSS) and sharpness
  values (SM) with their evaluated regularization strengths in adaptive IRIS
  fits. Keep valid curvature points during the search, restart each fit
  from the configured bounds, and allow switching off regularization between
  fits. (:pr:`1892`)

- Preserve metadata lists whose length differs from the number of dataset
  dimensions when transposing or swapping axes after loading an SCP file. (:pr:`1885`)


.. section

Dependency Updates
~~~~~~~~~~~~~~~~~~
.. Add here new dependency updates (do not delete this comment)


.. section

Breaking Changes
~~~~~~~~~~~~~~~~
.. Add here new breaking changes (do not delete this comment)

- IRIS uses SciPy NNLS by default for regularized fits. Install OSQP
  separately and pass ``qpsolver="osqp"`` to keep using the previous solver.
  Quadprog also requires a separate installation. (:pr:`1892`)


.. section

Deprecations
~~~~~~~~~~~~
.. Add here new deprecations (do not delete this comment)


.. section

Developer
~~~~~~~~~
.. Add here developer changes (do not delete this comment)

- PERF: Speed up IRIS fits by preparing the OSQP matrix once per regularization
  parameter, resolving the backend once per fit, and reducing the size of the
  sharpness calculation for wide datasets. (:pr:`1887`)
