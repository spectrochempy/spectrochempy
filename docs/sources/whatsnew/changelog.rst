
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

- ``MCRALS``: added the ``constraints=`` parameter that accepts a list of
  public ``Constraint`` objects (``NonNegative``, ``Unimodal``, ``Closure``,
  ``Monotonic``, ``ModelProfile``, …).  Both the new ``constraints=`` API and
  the legacy traitlet-based parameters now route through the same internal
  pipeline: the public ``Constraint`` objects have become the canonical
  intermediate representation.  Mixed usage (legacy traitlets + ``constraints=``
  at the same time) raises a clear ``ValueError``.  The legacy traitlet-based
    API remains fully supported for backward compatibility.  (:pr:`1383`)

- ``AnalysisResult`` (PCA, NMF, PLS, MCR‑ALS, …) and ``FitResult`` now have a
  rich ``_repr_html_`` for Jupyter notebooks, with collapsible Parameters,
  Outputs, and Diagnostics sections and recursive rendering of embedded
  NDDataset objects using their native HTML representation.  The text
  ``__repr__`` and the public API are unchanged. (:pr:`1299`)

- Peak analysis workflows are now easier to inspect and validate:
  ``find_peaks(..., as_result=True)`` returns a lightweight
  ``PeakFindingResult`` object with ``peaks``, ``properties``, a ``table``
  view, ``to_dict()``, ``to_csv()``, ``sort_by()``, ``head()``, ``top()``,
  and ``column()`` helpers, while
  ``Optimize.validate_script(script=None)`` can validate a curve-fitting
  script before launching the optimisation and returns structured
  ``ScriptError`` diagnostics.  The default ``find_peaks()`` return value
  remains the historical ``(peaks, properties)`` tuple, and the new helpers do
  not add a pandas dependency. (:pr:`1351`)

- The analysis user guide now includes a reference peak-analysis workflow
  tutorial connecting peak detection, ``PeakFindingResult``, ``PeakTable``,
  CSV export, manual script writing, script validation, and ``Optimize.fit()``
  in one end-to-end example. The `Optimize` user-facing documentation now also
  clarifies method selection (`least_squares`, `leastsq`, `simplex`,
  `basinhopping`), the meaning of ``FitResult.parameters`` as run
  configuration, and the lightweight ``validate_constraints()`` /
  ``constraints`` normalization surface for constraint validation before fit.

- ``FitResult`` from ``Optimize.result`` now exposes a residual dataset as
  ``result.outputs["residuals"]`` / ``result.residuals`` and basic fit-quality
  diagnostics under ``result.diagnostics``: ``rss`` / ``sse``, ``rmse``,
  ``r_squared``, ``n_observations``, ``n_varying_parameters``,
  ``degrees_of_freedom``, ``reduced_chi_square``, and
  ``adjusted_r_squared``, plus normalized solver ``success``, ``status``,
  and ``message`` fields. (:pr:`1365`)

- ``Optimize`` now retains the raw least-squares Jacobian on
  ``opt.jacobian`` when a backend naturally provides one, while methods
  without a native Jacobian expose ``None``. (:pr:`1369`)

- ``FitResult.covariance`` now exposes an approximate local least-squares
  covariance matrix for the fitted varying parameters, derived from the
  Jacobian when available and residual degrees of freedom are positive.
  ``FitResult.variance`` and ``FitResult.stderr`` expose the covariance
  diagonal and corresponding approximate parameter standard errors, and
  ``FitResult.correlation`` exposes the corresponding parameter-correlation
  matrix through the same availability rules. (:pr:`1373`, :pr:`1375`)

- ``FitResult.confidence_intervals`` now exposes approximate two-sided 95%
  confidence intervals for the fitted varying parameters, derived from the
  fitted values, standard errors, and Student-t critical values using the
  residual degrees of freedom. (:pr:`1376`)

- ``FitResult.diagnostics`` now exposes Gaussian-residual model-comparison
  criteria ``aic`` and ``bic`` derived from the residual sum of squares,
  observation count, and effective varying-parameter count. (:pr:`1378`)

  Existing ``result.fitted`` and ``result.components`` behavior is preserved.

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
    NumPy. (:pr:`1311`, :pr:`1312`)

- ``stack(..., axis=1)`` is now supported for stacking 1D profiles as
  columns into a 2D dataset.  This makes the workflow for building
  synthetic concentration or profile matrices fully native within the
  SpectroChemPy API, without falling back to ``np.column_stack(...)``
  or manual `NDDataset` wrapping. (:pr:`1309`, :pr:`1310`)

- Reading multi-object files such as MATLAB ``.mat`` files, multi-subfile SPC
  files, and ZIP archives now returns a ``ScpObjectList`` result with helper
  methods for selecting datasets by size, name, dimensionality, or shape.
  The ``ScpObjectList`` also gained ``dataset_by_ndim()``, ``dataset_by_name()``,
  ``datasets_by_shape()`` and other selection helpers. The list-like
  ``__getitem__``/``__len__`` interface is unchanged, and the new helpers
  do not add new dependencies. (:pr:`1306`, :pr:`1362`)

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
  (:pr:`1339`, :pr:`1344`, :pr:`1345`)

- 2D ``plot(method="lines"/"stack")`` now automatically uses coordinate labels
  as matplotlib line labels, so that ``ax.legend()`` shows meaningful names
  without needing to pass labels explicitly.  Legend entries are displayed
  in natural (first-to-last) order. (:pr:`1320`)

- Added top-level ``scp.polynomial(...)`` and ``scp.normal(...)`` helpers
  for synthetic spectral profile generation, completing the set of
  accessible line-shape helpers. (:pr:`1360`)


.. section

Bug Fixes
~~~~~~~~~
.. Add here new bug fixes (do not delete this comment)

- FIX: The ``Optimize`` script parser now rejects three categories of
  previously-silent misconfiguration: ``MODEL:`` blocks without a ``shape:``
  definition, ``shape:`` lines that appear before any ``MODEL:`` declaration,
  and duplicate ``MODEL:`` labels (which historically overwrote the first
  model's parameters).  These scripts could never produce valid fit results,
  and the parser now reports them as explicit errors instead of silently
  accepting them. (:pr:`1386`)

- WiRE (``.wdf``) reader no longer attaches YLST data as a confusing auxiliary ``m``
  coordinate in ``coordset``. The YLST data is now stored on ``dataset.meta`` (as
  ``ylst_data``, ``ylst_title``, ``ylst_units``) where it belongs as per-spectrum
  metadata. (:pr:`1332`)

- ``MCRALS`` correctness fixes (:pr:`1340`, :pr:`1363`): empty ``closureConc`` no longer runs a
  wasteful closure block (PR1 B4); single-component closure targets
  (``closureConc=[0]``, ``closureConc="all"``) are now honoured instead of
  being silently disabled by ``np.any`` (issue #911); ``normSpec='max'`` /
  ``'euclid'`` with a zero-norm spectrum no longer produces ``nan``/``inf``
  (PR1 B9); ``getConc`` / ``getSpec`` dispatch now correctly uses
  ``argsGetSpec`` / ``kwargsGetSpec`` instead of the ``argsGetSpecc`` typo
  (PR1 B1) and accepts bare-profile, 2-tuple ``(profiles, new_args)``, and
  3-tuple ``(profiles, new_args, extra)`` returns (PR1 B2);
  ``getSt_to_St_idx`` with ``None`` entries no longer crashes in the
  validator's ``max()`` call (PR1 B5); ``_unimodal_1D`` no longer
  infinite-loops or indexes out of bounds for pathological tolerances
  (``tol < 1``, including ``tol == 1.0``) while remaining byte-identical
  in the documented regime (``tol >= 1.1``) (PR1 B6); ``monoIncTol`` and
  ``monoDecTol`` are now documented as ``Float`` traits (PR1 B7/B8).
  The same ``np.any`` component-selection anti-pattern that caused
  issue #911 in ``_ClosureConstraint`` has now been fixed across the
  remaining constraint activation guards: selecting only component 0
  with ``nonnegConc=[0]``, ``nonnegSpec=[0]``, ``unimodConc=[0]``,
  ``unimodSpec=[0]``, ``monoIncConc=[0]``, ``monoDecConc=[0]``,
  ``hardConc=[0]`` or ``hardSpec=[0]`` no longer silently disables the
  constraint (the guards now use an explicit truthiness test instead of
  ``np.any(...)``, since ``np.any([0])`` evaluates to ``False``).

- Plotting behavior is more consistent again: scatter plots now show markers as
  expected, single-dataset ``plot_multiple()`` / ``multiplot()`` calls preserve
  the requested plotting method, 1D and 2D artists honor more style keywords
  consistently, ``use_plotly=True`` fails with a clear error when Plotly is not
  available, and legacy ``lines`` / ``pen`` aliases continue to work across
  dimensional fallbacks. (:pr:`1381`, :pr:`1384`)

- ``plotmerit()`` / ``plot_compare()`` are now clearer for fit inspection:
  the reconstructed trace remains visible even for near-perfect overlaps, the
  historical ``kind="scatter"`` / ``method="scatter"`` options now produce
  real marker-based rendering, ``nb_traces`` is honored in the current plotting
  path, and ``offset`` once again separates the residual trace from the main
  signal to improve notebook readability. (:pr:`1377`)

- Fixed several processing regressions and edge cases: multi-dimensional ZPD
  detection in interferogram apodization is more reliable, ``rs()``, ``ls()``,
  and ``roll()`` now shift multi-dimensional data along the correct axis,
  ``denoise()`` validates its 2D input correctly. (:pr:`1352`)

- Preprocessing transformers no longer emit a traitlets dtype warning when
  operating on ``MaskedArray`` inputs — the internal cast now uses
  ``np.asarray()`` instead of ``np.array()`` to preserve the plain-ndarray
  contract while silencing the warning. (:pr:`1348`)

- Fixed several processing regressions and edge cases: ``npy.dot()`` now
  checks the correct operand type and honours the ``strict`` argument. (:pr:`1352`)

- ``PLSRegression`` now works with a 1D ``NDDataset`` as the response variable
  ``y``. This fixes failures in ``predict()``, ``y_scores``, ``y_loadings``,
  ``y_weights``, ``y_rotations``, ``result``, and ``coef`` when fitting with a
  1D target. (:pr:`1305`)


.. section

Dependency Updates
~~~~~~~~~~~~~~~~~~
.. Add here new dependency updates (do not delete this comment)

- Bumped ``actions/cache`` from 5 to 6 (CI only). (:pr:`1316`)
- Bumped ``actions/checkout`` from 4 to 7 (CI only). (:pr:`1353`)


.. section

Breaking Changes
~~~~~~~~~~~~~~~~
.. Add here new breaking changes (do not delete this comment)


.. section

Deprecations
~~~~~~~~~~~~
.. Add here new deprecations (do not delete this comment)

- The historical internal import path
  ``spectrochempy.processing.alignement`` is superseded by the canonical
  ``spectrochempy.processing.alignment`` namespace. The public
  ``scp.align(...)`` and ``dataset.align(...)`` APIs are unchanged. The old
  path remains available as a deprecated compatibility alias, scheduled for
  removal in ``0.12.0``. (:pr:`1372`)

- Plotting ``method="stack"`` and ``method="map"`` aliases are deprecated.
  Use the canonical ``method="lines"`` (for 1D) or let the dispatcher choose
  automatically.  The aliases will be removed in ``0.12.0``.

- The ``force_stack`` keyword argument in concatenation functions is
  deprecated in favour of the unqualified ``method="stack"`` call. It will
  be removed in ``0.12.0``.


.. section

Developer
~~~~~~~~~
.. Add here developer changes (do not delete this comment)

- MAINT: Moved the Optimize backend onto the canonical ``_FitModelSpec``
  representation for model preparation, solver vectorization/restoration,
  diagnostics, and result parameter extraction while keeping
  ``FitParameters`` as a compatibility layer. The preferred execution path
  now resolves references through ``prepare_model()`` before numerical
  optimization, preserving the existing fit behaviour and public API.
  The parser now also produces the canonical model representation first and
  only derives ``FitParameters`` for legacy compatibility surfaces such as
  ``Optimize.fp`` and the historical parser return value. (:pr:`1392`, :pr:`1393`)

- MAINT: Extracted parameter-space transforms from ``FitParameters`` into
  standalone ``_to_internal`` / ``_to_external`` utilities
  (``_parameter_transform.py``).  Two historical bugs fixed: ``lob is not
  None`` → ``upb is not None``, ``pei = pi`` → ``pei = item``.  (:pr:`1388`, :pr:`1389`)

- MAINT: Added private ``_FitModelSpec`` structured model-definition object
  with ``from_fitparameters(fp)`` constructor, ``to_script()`` serializer, and
  ``count_varying()`` / ``extract_varying_values()`` inspection methods.
  ``_ComponentParamsView`` adapter decouples ``getmodel()`` from the
  ``{param}_{label}`` parser convention by duck-typing.  Strategy B of the
  Optimize convergence plan — three phases delivered: parameter transform
  extraction, structured model spec, component-params adapter.  (:pr:`1387`, :pr:`1390`)

- ``MCRALS``: introduced the public constraint API skeleton
  (``spectrochempy.analysis.decomposition.mcrals_constraints``). The new
  public classes ``Constraint`` (abstract base), ``NonNegative``,
  ``Closure``, ``Unimodal``, ``Monotonic``, ``ZeroRegion``,
  ``Selectivity``, ``FixedValues``, ``ReferenceProfile`` and
  ``ModelProfile`` describe scientific prior knowledge about the
  concentration (``"C"``) or spectral (``"St"``) profiles of ``MCRALS``
   as declarative, first-class objects. All ten classes are importable
   from the ``spectrochempy.analysis.constraints`` submodule
   (``from spectrochempy.analysis import constraints``), registered in the
   public API reference, and covered by dedicated construction / validation
  / equality / repr / model / tolerance tests
  (``tests/test_analysis/test_decomposition/test_mcrals_constraints.py``).
  This PR introduces the vocabulary and public surface **only**: the
  classes are data containers and validators, they are **not yet
  connected** to the internal constraint engine, and using them does
  not change the behaviour of ``MCRALS.fit``. No public APIs and no
  numerical behaviour change. Connection to the internal engine, the
  legacy traitlet converter, and the actual enforcement implementations
  are the subject of subsequent PRs (see the project RFC for the
  roadmap).

- ``MCRALS``: added a behavioral characterization test matrix
  (``tests/test_analysis/test_decomposition/test_mcrals.py``) freezing the
  current numerical output of ``MCRALS`` across the documented constraint
  (non-negativity, unimodality, monotonicity, closure, spectral
  normalization, hard-generated profiles), solver (``lstsq``, ``nnls``,
  ``pnnls``) and initialization (``C`` vs. ``St``) space, ahead of the next
  structural refactoring. The tests use a tiny deterministic synthetic
  dataset and compare against fixed expected arrays. The matrix initially
  exposed the same ``np.any([0])`` component-selection anti-pattern that
  affected issue #911 in ``_ClosureConstraint`` (selecting only component 0
  with ``nonnegConc=[0]`` / ``monoIncConc=[0]`` / ``monoDecConc=[0]`` /
  ``hardConc=[0]`` / ``hardSpec=[0]`` silently disabled the constraint);
  these are now fixed (see the Bug Fixes section above) and frozen as
  passing regression tests. See ``MCRALS_PR4_BEHAVIOR_TESTS.md`` for the
  characterization report.

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
  public plotting aliases and rendering behavior. (:pr:`1370`)

- Centralized internal plotting method normalization in a private helper
  module to reduce duplication across backend dispatch, multiplot handling,
  and 1D/2D fallback validation, without changing the public plotting API. (:pr:`1370`)

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
  and history messages. (:pr:`1344`)

- CI: Moved archived stable docs builds (oldest supported + recent
  versions) out of the per-push ``build_docs.yml`` workflow into a
  dedicated weekly ``build_docs_archived_versions.yml`` workflow.  This
  shrinks the main deployment artifact (~382 MB previously) and avoids
  GitHub Pages ``syncing_files`` timeouts caused by oversized bundles.
  The main workflow now builds only the current version (plus preview
  docs for documentation branches).  Added ``timeout-minutes: 360`` to
  both workflows and ``continue-on-error: true`` on archived-version
  steps so a single failing old-tag build does not block the rest. (:pr:`1350`)

- DOC: Refreshed the installation guide across all platforms:
  added ``uv`` as the recommended installation method, created a
  "Choosing an installation method" decision table, replaced
  deprecated Mambaforge references with Miniforge, and updated
  contributor setup instructions to match the project's own
  ``uv``-based development toolchain.  Conda/mamba and pip remain
  documented as alternatives.  (:pr:`1368`)
