
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

- ``AnalysisResult`` (PCA, NMF, PLS, MCR‑ALS, …) and ``FitResult`` now have a
  rich ``_repr_html_`` for Jupyter notebooks, with collapsible Parameters,
  Outputs, and Diagnostics sections and recursive rendering of embedded
  NDDataset objects using their native HTML representation.  The text
  ``__repr__`` and the public API are unchanged. (#1299)

- Peak analysis workflows are now easier to inspect and validate:
  ``find_peaks(..., as_result=True)`` returns a lightweight
  ``PeakFindingResult`` object with ``peaks``, ``properties``, a ``table``
  view, ``to_dict()``, and ``to_csv()`` helpers, while
  ``Optimize.validate_script(script=None)`` can validate a curve-fitting
  script before launching the optimisation and returns structured
  ``ScriptError`` diagnostics.  The default ``find_peaks()`` return value
  remains the historical ``(peaks, properties)`` tuple, and the new helpers do
  not add a pandas dependency. (#1351)

- SpectroChemPy now exposes top-level helpers for common 1D line shapes:
  ``scp.polynomial(...)``, ``scp.gaussian(...)``,
  ``scp.lorentzian(...)``,
  ``scp.voigt(...)``, ``scp.asymmetricvoigt(...)``, and ``scp.sigmoid(...)``.
  The line-shape helpers except ``scp.sigmoid`` also accept
  ``normalized=False`` to return a profile whose peak amplitude is exactly
  ``ampl``. Common mathematical helpers are also available at top level:
  ``scp.exp(...)``, ``scp.log(...)``, ``scp.log10(...)``, ``scp.sin(...)``,
  and ``scp.cos(...)``. Synthetic profile creation also gains
  ``scp.normal(...)`` for native Gaussian noise generation without dropping to
  NumPy. (#1301)

- ``stack(..., axis=1)`` is now supported for stacking 1D profiles as
  columns into a 2D dataset.  This makes the workflow for building
  synthetic concentration or profile matrices fully native within the
  SpectroChemPy API, without falling back to ``np.column_stack(...)``
  or manual `NDDataset` wrapping.

- SpectroChemPy now has a fuller preprocessing API for chemometric workflows:
  standard operations such as ``normalize()``, ``center()``, ``autoscale()``,
  ``snv()``, ``msc()``, ``pareto_scale()``, ``range_scale()``,
  ``robust_scale()``, and ``log_transform()`` are available as first-class
  `NDDataset` methods, and matching stateful transformer classes
  (``CenterTransformer``, ``AutoscaleTransformer``, ``SNVTransformer``,
  ``NormalizeTransformer``, ``MSCTransformer``, ``ParetoScaleTransformer``,
  ``RangeScaleTransformer``, ``RobustScaleTransformer``, ``LogTransformer``)
  support ``fit()``, ``transform()``, ``fit_transform()``, and
  ``inverse_transform()`` where applicable.  The transformers also expose
  ``get_params()`` / ``set_params(**params)`` and a clear ``__repr__``,
  enabling safer reuse on new data and easier parameter inspection or cloning.

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

- WiRE (``.wdf``) reader no longer attaches YLST data as a confusing auxiliary ``m``
  coordinate in ``coordset``. The YLST data is now stored on ``dataset.meta`` (as
  ``ylst_data``, ``ylst_title``, ``ylst_units``) where it belongs as per-spectrum
  metadata. (#1332)

- ``scp.dot()`` now honours its ``strict`` argument, forwarding it to
  ``numpy.ma.dot`` so masked values are propagated (``strict=True``) or treated
  as zero (``strict=False``) as documented. The signature default now matches
  the documentation and NumPy (``strict=False``), which preserves the previous
  behaviour for existing callers (``strict`` was silently ignored before).

- ``MCRALS`` is now more robust in constrained and hard-model workflows:
  closure constraints are applied correctly for empty and single-component
  selections, zero-norm spectral normalization no longer produces
  ``nan``/``inf``, ``getConc`` / ``getSpec`` argument handling is fixed, and
  pathological unimodality settings no longer trigger crashes or infinite
  loops. Validation around ``getSt_to_St_idx`` is also safer. (issue #911)

- Plotting behavior is more consistent again: scatter plots now show markers as
  expected, single-dataset ``plot_multiple()`` / ``multiplot()`` calls preserve
  the requested plotting method, 1D and 2D artists honor more style keywords
  consistently, ``use_plotly=True`` fails with a clear error when Plotly is not
  available, and legacy ``lines`` / ``pen`` aliases continue to work across
  dimensional fallbacks.

- Fixed several processing regressions and edge cases: multi-dimensional ZPD
  detection in interferogram apodization is more reliable, ``rs()``, ``ls()``,
  and ``roll()`` now shift multi-dimensional data along the correct axis,
  ``denoise()`` validates its 2D input correctly, and ``npy.dot()`` no longer
  checks the wrong operand type.

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

- ``MCRALS``: internal architecture overhaul, no public API or numerical
  behavior change. ``MCRALS._fit`` is now structured around an internal
  constraint engine: the public traitlets (``nonnegConc``, ``unimodConc``,
  ``monoIncConc``/``monoDecConc``, ``closureConc``, ``normSpec``,
  ``hardConc``/``hardSpec`` and their spectral counterparts) are translated
  once per fit into a private, ordered list of ``_Constraint`` objects, and
  the ALS loop iterates over them rather than calling each constraint by
  name. All constraint classes are private (``_``-prefixed) and not
  exported in ``__all__``; the constraint order, the in-place vs. copy
  semantics, the traitlets, the ``fit`` / ``transform`` / ``fit_transform``
  / ``inverse_transform`` signatures, the ``_outfit`` return tuple, and
  the ``getConc`` / ``getSpec`` external-generator contract are preserved
  byte-for-byte. This refactoring reduces ``_fit`` from a ~230-line
  monolith to a high-level loop over constraint pipelines and prepares
  the ground for future work (constraint pipelines, generated-profile
  abstraction, multi-criteria convergence, scikit-learn compatibility).

- Centralized internal plotting kwargs normalization in a private helper
  module to reduce duplication across plotting entry points while preserving
  public plotting aliases and rendering behavior.

- Centralized internal plotting method normalization in a private helper
  module to reduce duplication across backend dispatch, multiplot handling,
  and 1D/2D fallback validation, without changing the public plotting API.

- CI: Added a PR compliance workflow (`pr-compliance.yml`) that automatically
  checks:
  (1) the PR title starts with a valid prefix listed in `CONTRIBUTING.md`
  (read dynamically so the list stays single-source),
  (2) `docs/sources/whatsnew/changelog.rst` has been updated.
  Both checks can be bypassed via the labels `non-standard-prefix` and
  `no-changelog` respectively. The PR-title prefix extraction now tolerates
  Markdown table spacing in `CONTRIBUTING.md`, and Dependabot PRs are exempt
  from both the prefix and changelog checks so automated dependency updates do
  not require manual title edits or labels.

- DEV: Added a `commit-msg` pre-commit hook (`check_commit_prefix`) that
  reads the allowed prefixes directly from `CONTRIBUTING.md` and rejects
  any commit whose subject line does not start with one of them.
  Merge commits and reverts are automatically exempt.
  Use `git commit --no-verify` to bypass locally if necessary.

- MAINT: Refactored all nine procedural preprocessing functions
  (`normalize`, `center`, `autoscale`, `snv`, `msc`, `pareto_scale`,
  `range_scale`, `robust_scale`, `log_transform`) to internally
  delegate to their transformer counterparts.  This eliminates code
  duplication between the procedural and transformer APIs while
  preserving full backward compatibility, including `inplace` behaviour
  and history messages.

- All analysis estimators (PCA, PLSRegression, Baseline, MCRALS, NMF,
  etc.) now expose ``get_params()`` and ``set_params(**params)``
  following scikit-learn conventions, plus a clear ``__repr__``.
  This enables parameter inspection and grid exploration for any
  ``AnalysisConfigurable`` subclass without adding new dependencies.
  Full ``sklearn.base.clone()`` compatibility is best-effort because
  complex traitlets traits (e.g., lists) may fail sklearn's strict
  identity check.

- CI: Moved archived stable docs builds (oldest supported + recent
  versions) out of the per-push ``build_docs.yml`` workflow into a
  dedicated weekly ``build_docs_archived_versions.yml`` workflow.  This
  shrinks the main deployment artifact (~382 MB previously) and avoids
  GitHub Pages ``syncing_files`` timeouts caused by oversized bundles.
  The main workflow now builds only the current version (plus preview
  docs for documentation branches).  Added ``timeout-minutes: 360`` to
  both workflows and ``continue-on-error: true`` on archived-version
  steps so a single failing old-tag build does not block the rest.
