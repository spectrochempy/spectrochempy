
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

- MCRALS now supports three dimensionless stopping tolerances:
  ``tol_residual_change`` (default ``1e-3``),
  ``tol_reconstruction_error``, and ``tol_profile_change``. The latter two
  are disabled by default. The optimisation stops when any enabled criterion
  is satisfied, and ``result.diagnostics`` reports all three values together
  with ``convergence_reason``. The INFO log now displays
  ``reconstruction_error``, ``residual_change``, ``profile_change``, and the
  residual trend, followed by the exact value and tolerance responsible for
  convergence. The former ``RSE / PCA``, ``RSE / Exp``, and ``%change``
  columns have been removed because they did not map clearly to the stopping
  criteria.
.. Add here new public features (do not delete this comment)

- Score plot labels now use ``adjustText`` for intelligent placement (collision
  avoidance with markers and other labels). Falls back to a fixed offset if
  ``adjustText`` is not installed.

- PCA component labels now display as ``PC1``, ``PC2``, ... instead of ``#0``,
  ``#1``, ... in legends and coordinate display. Other analysis methods
  retain the default ``#0``, ``#1``, ... labels. (#1404)

- Introduced ``scp.nmr.Experiment``, an NMR-specific scientific model that
  wraps an ``NDDataset`` and provides state-aware processing.  The class
  classifies the current data domain (time, frequency, mixed), identifies
  source kind (FID, SER, processed 1D/2D), exposes NMR metadata (encoding,
  nuclei), validates NMR-specific requirements, and orchestrates processing
  that is appropriate for the current domain — never performing FFT on
  already-transformed data.  This is the first step toward a simplified NMR
  processing workflow.

- Added ``download_extra_testdata()`` to ``spectrochempy.application.testdata``
  for fetching extra NMR datasets (agilent, jeol, bruker_3d, simpson, tecmag)
  from the ``data-extra`` branch of the ``spectrochempy_data`` repository.
  Extra data is cloned into ``~/.spectrochempy/testdata-extra/``. (:pr:`1418`)

- Added four new NMR format readers in the NMR plugin: Agilent/Varian
  (``scp.nmr.read_agilent``, binary ``fid`` + ``procpar``), JEOL JDF
  (``scp.nmr.read_jeol``), TecMag TNT (``scp.nmr.read_tecmag``), and
  SIMPSON (``scp.nmr.read_simpson``, ``TEXT``/``BINARY``/``RAWBIN`` formats).
  All use vendored NMRGlue code and support automatic format detection.

- Added plugin-contributed I/O namespaces for NMR and PerkinElmer readers.
  ``scp.topspin.read`` and ``scp.agilent.read`` now expose the format-specific
  readers with the same short-method namespace API used by core I/O domains.
  ``scp.nmr.read`` is a generic dispatcher that auto-detects TopSpin and
  Agilent/Varian formats (or uses ``protocol=``).  Root-level aliases
  ``scp.read_topspin``, ``scp.read_agilent`` and ``scp.read_perkinelmer``
  remain available as compatibility shims.


.. section

Bug Fixes
~~~~~~~~~
.. Add here new bug fixes (do not delete this comment)

- Display legend in 2D lines/stack plots when ``legend=True`` is passed as a
  plotting keyword argument. Previously the ``legend`` kwarg was silently
  ignored by the 2D plot backend, so lines rendered with auto-populated labels
  from coordinate metadata never showed a legend. (#1404)

- Improved TopSpin reader reliability across several areas.  ``FnMODE``/``MC2``
  is now read from ``acqu2s`` (not mis-indexed ``acqus``) for correct 2D SER
  indirect-dimension encoding.  The SER reshape fallback uses proper dictionary
  keys (``acqu2s``/``acqus``) to avoid ``KeyError``.  Processed-data ``phc0``
  is read from ``procs`` instead of being unconditionally zeroed.
  Normalisation guards against division by zero when ``ns`` or ``rg`` is zero.
  ``datetime.fromtimestamp`` is protected against invalid or negative
  timestamps.  Metadata exception suppression is narrowed from ``Exception``
  to ``(TypeError, IndexError)``.  ``sw_h`` computation is ``None``-safe.
  Nucleus-string parsing and missing ``use_list`` files are handled gracefully.
  Dead comments and uncertain TODOs removed. (#1420, #1424)

- ``scp.nmr.Experiment`` now correctly classifies non-Bruker datasets
  (JEOL, TecMag, SIMPSON).  Previously, metadata extraction was hardcoded
  to Bruker field names, causing silent misclassification of other formats.

- Fixed 2D NMR FFT chain so ``fft()`` works on quaternion-encoded 2D data
  (STATES, TPPI, ECHO-ANTIECHO).  The encoding handler dispatch read
  ``meta.encoding`` after ``swapdims`` had reordered the list, selecting the
  wrong handler (DQD instead of STATES).  The second FFT pass through the
  encoding handler also produced conjugated subspectra because the quaternion
  rebuild/extract cycle is not invertible by the encoding-specific
  decomposition formula.  Both issues are resolved: encoding is captured
  before the swap, and the second pass extracts complex subspectra directly
  from the rebuilt quaternion.  Three end-to-end tests validate the full
  ``fft(dim=-1)`` then ``fft(dim=0)`` chain on synthetic 2D SER data.


.. section

Dependency Updates
~~~~~~~~~~~~~~~~~~
.. Add here new dependency updates (do not delete this comment)


.. section

Breaking Changes
~~~~~~~~~~~~~~~~
.. Add here new breaking changes (do not delete this comment)

- The TopSpin reader (``scp.nmr.read_topspin``) now supports 1D and 2D data
  only. Reading 3D/4D data raises ``NotImplementedError``. The previous
  "nD" claim was not backed by a suitable hypercomplex representation for
  dimensions higher than two.

- ``MCRALS.constraints`` is now a validated traitlet, enabling both constructor
  and post-construction assignment while preserving the distinction between
  ``None`` (built-in defaults) and ``[]`` (explicitly unconstrained fit).
  Assignment of ``constraints`` after fitting invalidates the fitted state.
  The ``constraints`` parameter is not config-file serializable. (:pr:`XXXX`)

- MCRALS public outputs ``C``, ``St`` and residuals now correspond to the
  **constrained** factor pair ``(C_constrained, St_constrained)`` instead of
  the previous mixed pair ``(C_LS, St_constrained)``.  This matches the
  semantics of Tauler MATLAB MCR-ALS, pyMCR, and PLS_Toolbox.  Convergence
  diagnostics also use the constrained pair, which can change convergence
  speed and iteration counts compared with the old behaviour.  The
   unconstrained least-squares estimate is still available via the new
   ``C_ls`` property. (:pr:`XXXX`)

- Refactored the internal ALS iteration loop in ``MCRALS._fit`` to match the
  standard Tauler formulation: each iteration now performs exactly one C solve
  followed by one constraint pass, then one St solve followed by one constraint
  pass (previously the concentration constraint pipeline ran twice per
  iteration, causing side-effects to double for ``ModelProfile`` generators).
  This may change iterate counts and numerical results under active
  constraints, but the publicly documented ``C @ St ≈ X`` reconstruction
  invariant is preserved. (:pr:`XXXX`)


.. section

Deprecations
~~~~~~~~~~~~
.. Add here new deprecations (do not delete this comment)

- MCRALS public documentation now exposes only dimensionless convergence
  tolerances, profile-specific solvers, and the unified ``constraints`` API.

- ``AnalysisBase.plotmerit`` alias is deprecated; use ``plot_merit``
  instead.  ``plotmerit`` will be removed in version 0.12.

- ``parityplot`` is renamed to ``plot_parity`` for naming consistency with
  other composite plot functions (``plot_score``, ``plot_scree``, ...).
  ``parityplot`` (both the standalone function and the
  ``CrossDecompositionAnalysis`` method) is retained as a deprecated alias
  that will be removed in version 0.12.


.. section

Developer
~~~~~~~~~
.. Add here developer changes (do not delete this comment)

- Reactivated NMR legacy tests in
  ``tests/test_processing/test_fft/test_nmr.py``. Removed the module-level
  ``pytestmark = pytest.mark.skip`` and deleted 10 purely visual tests that
  had zero data assertions. Rewrote the remaining 8 tests with proper
  numerical assertions: reader string repr, em/gm apodization with inplace
  and window-value checks, FFT energy preservation, manual phasing invariants,
  and axis-specific 2D em shape preservation. Rewrote
  ``plugins/spectrochempy-nmr/tests/test_nmr_smooth.py`` with shape and
  noise-reduction assertions replacing the visual-only original. Zero
  regressions across the full NMR + hypercomplex + decoupling suite.

- DOC: Improved example gallery to showcase SpectroChemPy-native idioms
  (``Coord.linspace``, ``Coord.arange``, ``scp.abs``) for coordinate creation
  and dataset operations, replacing redundant ``np.linspace`` + ``Coord``
  wrapping patterns, ``np.abs`` usage, list-comprehension synthetic data
  generators (``scp.fromfunction``), ``np.random.normal`` on datasets
  (``scp.normal``), ``np.arange`` wrapped in NDDataset (``scp.arange``),
  and ``np.random.rand`` + NDDataset constructor (``NDDataset.random``).
  Also updated API docstring examples to use ``scp.gaussian``,
  ``Coord.linspace``, ``scp.arange``, and ``NDDataset.random``
  instead of raw NumPy equivalents. (#1370)

- MAINT: Unified the plotting lifecycle across core and plugin composite
  functions (``plot_score``, ``plot_scree``, ``plot_compare``,
  ``plot_merit``, ``plot_baseline``, ``plot_parity``, ``plot_multiple``,
  and IRIS ``plot_iris_*``).  Extracted shared ``_setup_axes``/``_maybe_show``
  helpers into ``mplutils.py``, replacing duplicated figure/axes/show
  boilerplate.  Removed all ``plt.*`` global calls.  ``plot_parity``
  (formerly ``parityplot``) extracted from ``CrossDecompositionAnalysis``
  into a standalone function; ``plot_multiple`` gains ``ax``, ``clear``,
  ``show`` parameters.  Style parameters (``marker``, ``s``, ``alpha``)
  and kwargs normalization (``color``/``c``, ``linestyle``/``ls``, …)
  integrated into composite plots.  Removed the non-functional Plotly/Dash
  backend (never a declared dependency).  31 structural and functional
  tests added. (#1412, #1413, #1414, #1416)

- MAINT: Extracted generic nmrglue utilities (``create_blank_udic``,
  ``unit_conversion``, ``uc_from_udic``, ``reorder_submatrix``,
  ``complexify_data``, ``uncomplexify_data``, ND array iterators, …)
  from ``_bruker.py`` into a shared ``_base.py`` module, eliminating
  cross-reader coupling between Bruker, Varian, and JEOL. (#1408)

- DOC: Documented the three-layer plotting architecture (scientific objects →
  composite plotters → dataset plotting) in the developer guide.  Updated API
  reference to include composite functions as top-level entries.  Added
  composite plot customization section to the user guide.  Deprecated
  ``AnalysisBase.plotmerit`` in favor of ``plot_merit``. (#1415)

- MAINT: Replaced the invalid Trove classifier ``Framework :: SpectroChemPy
  :: Official Plugin`` with a private ``[tool.spectrochempy]``
  ``official-plugin = true`` marker in all official plugin ``pyproject.toml``
  files and all CI/workflow scripts.  The classifier was never registered
  with PyPI and caused ``400 Bad Request`` errors on upload.  The new marker
  is the single registration point for CI (publishing, testing, release
  validation).  Added ``validate_official_plugin.py`` and 9 structural tests.
  (:pr:`XXXX`)
