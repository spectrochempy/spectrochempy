# SpectroChemPy Test Suite Audit

Date: 2026-05-20

This is a first-pass audit after the plugin-system work. It focuses on test
organization, plugin/core coupling, lazy imports, missing-plugin behavior, and
obvious cleanup candidates. It is not a full test-suite redesign.

## Current Picture

The plugin-system tests are now concentrated mainly in:

- `tests/test_plugins/`: core plugin manager, registry, lazy import, namespace,
  missing-plugin, root alias, and integration behavior.
- `plugins/spectrochempy-iris/tests/`: IRIS plugin API, lazy exports, root
  compatibility aliases, dataset accessor behavior, and migrated IRIS core tests.
- `plugins/spectrochempy-nmr/tests/`: NMR plugin registration, TopSpin reader,
  missing core stub behavior, and NMR processing/data tests.
- `plugins/spectrochempy-cantera/tests/`: Cantera plugin metadata, namespace
  exposure, root alias behavior, and PFR scope checks.

This split is mostly coherent. The remaining issues are limited enough to clean
incrementally.

## Findings

### Core vs Plugin Coupling

The main `tests/` tree still contains some plugin-dependent tests:

- `tests/test_analysis/test_plotmerit.py`
  imports `spectrochempy_iris` and exercises IRIS plotting behavior. This is a
  plugin test living in the core test tree. It is currently guarded with
  `pytest.importorskip("spectrochempy_iris")`.
- `tests/test_plotting/test_composite_imports.py`
  imports IRIS plotting helpers from `spectrochempy_iris`. These are lightweight
  import checks, but they are still plugin-specific.
- `tests/test_analysis/test_kinetic/test_kineticutilities.py`
  imports `spectrochempy_cantera._pfr` inside a Cantera availability test. This
  is now a cross-boundary dependency from a core kinetic test into a plugin
  implementation module.
- `tests/conftest.py`
  defines `NMR_dataset_1D` and `NMR_dataset_2D` fixtures that call
  `spectrochempy.read_topspin(...)`. These fixtures are safe as long as they are
  only used by NMR tests, but they make the global conftest aware of an optional
  plugin reader.
- `tests/test_processing/test_fft/test_nmr.py`
  is globally skipped as WIP and uses `scp.read_topspin(...)`.
- `tests/test_processing/test_filter/test_smooth.py`
  is globally skipped as WIP with NMR data but also contains one IR smoothing
  test.

Recommended direction: keep the core tests plugin-free by default, and move or
mark these as plugin integration tests in a later cleanup.

### Import and Lazy-Loading Tests

Coverage is already good in `tests/test_plugins/test_integration.py`:

- plain `import spectrochempy` does not import plugin modules;
- `scp.iris`, `scp.nmr`, `scp.cantera` namespace access is lazy;
- `import spectrochempy.iris` / `spectrochempy.nmr` / `spectrochempy.cantera`
  works through namespace modules;
- `from spectrochempy.nmr import read_topspin` is tested;
- `repr(...)`, `__doc__`, `__name__`, `__wrapped__`, and `inspect.signature(...)`
  are tested for lazy proxies.

Potential gap: the current introspection checks cover NMR and Cantera, but IRIS
class introspection through `scp.iris.IRIS` is lighter. Add it only if this
becomes a regression risk; it is not urgent.

### Missing Plugin Behavior

Missing official plugins are tested in several places:

- `tests/test_plugins/test_integration.py`
  checks missing namespace errors and actionable install hints.
- `tests/test_plugins/test_manager.py`
  checks `MissingPluginError` formatting.
- `tests/test_core/test_dataset/test_reader_api_policy.py`
  checks the `scp.read_topspin` missing-plugin stub without importing NMR.

This area is in good shape. The important policy is already covered:

- official known optional readers, such as `read_topspin`, get actionable stubs;
- unknown third-party reader names do not get fake stubs;
- missing namespaces produce plugin-specific install hints.

### Deprecated Root Aliases

Deprecated aliases are covered:

- `scp.IRIS` and `scp.IrisKernel` are tested in IRIS plugin tests and plugin
  integration tests.
- `scp.PFR` is tested in Cantera plugin tests and plugin integration tests.
- Examples and docs mostly recommend namespaced APIs such as `scp.iris.IRIS` and
  `scp.cantera.PFR`.

This is coherent. Keep deprecated aliases out of new examples.

### Skips and Xfails

Skips that look expected:

- Cantera plugin tests skip runtime PFR checks when Cantera is missing.
- NMR plugin tests skip data-dependent checks when NMR test data is unavailable.
- docs/example tests skip plugin-dependent examples when the matching plugin is
  missing.
- Windows docs/example tests are currently skipped because that path is known to
  be unreliable.

Skips that should be revisited:

- `plugins/spectrochempy-iris/tests/test_iris_core.py`
  has a skip reason "raises an error in github test"; this should become a
  specific xfail or be fixed.
- `tests/test_analysis/test_crossdecomposition/test_pls.py`,
  `tests/test_analysis/test_decomposition/test_pca.py`, and
  `tests/test_core/test_dataset/test_dataset.py` have similarly vague GitHub
  skip reasons.
- `tests/test_processing/test_fft/test_nmr.py` and
  `tests/test_processing/test_filter/test_smooth.py` are globally skipped WIP
  files and should be either moved, split, or deleted after review.

### Slow or Repeated Work

Candidates for later optimization:

- `plugins/spectrochempy-iris/tests/test_iris_core.py` emits a large number of
  OSQP warnings and does substantial IRIS fitting. It is valuable but should
  probably be marked or split into fast API checks and slower numerical checks.
- `tests/test_analysis/test_plotmerit.py` repeats the same IRIS data setup in
  several tests. A local fixture would reduce duplicated work if the file is kept.
- `tests/test_docs/test_py_in_docs.py` executes generated documentation scripts
  and example modules; it already uses plugin detection and `MPLBACKEND=Agg`,
  but it remains inherently slow and should stay clearly marked as `slow`.

### Gallery and Documentation Examples

`tests/test_docs/test_py_in_docs.py` detects plugin requirements through:

- explicit `OPTIONAL_PLUGIN = "..."`
- fallback markers such as `read_topspin`, `spectrochempy_iris`, and
  `spectrochempy_cantera`.

This is adequate for now. The examples are centralized and plugin-dependent
examples are skipped cleanly when the plugin is absent. Prefer adding explicit
`OPTIONAL_PLUGIN` markers to future plugin-dependent examples because text-marker
detection is easy to miss.

## Small Patches Made

`plugins/spectrochempy-nmr/tests/test_plugin.py` was adjusted so
`test_package_namespace_exposes_topspin_reader` checks public behavior rather
than proxy object identity. `scp.nmr.read_topspin` and `scp.read_topspin` can be
distinct lazy proxy objects while still resolving to the same wrapped reader.

A follow-up stabilization pass also:

- made the shared NMR fixtures in `tests/conftest.py` skip explicitly when the
  NMR plugin is unavailable;
- clarified IRIS/Cantera plugin integration checks that still live in the main
  `tests/` tree;
- replaced vague "raises an error in github test" skip reasons with explicit
  docstring-checker quarantine reasons;
- made globally skipped legacy NMR/smoothing files explain why they remain
  quarantined.

## Prioritized Follow-Up Plan

### P0: Before Merging Plugin Work

- Move plugin-specific tests remaining in the main `tests/` tree:
  `test_plotmerit.py`, IRIS imports in `test_composite_imports.py`, and the
  Cantera `_pfr` import in `test_kineticutilities.py`. They have been clarified
  as plugin integrations, but not moved yet.
- Decide what to do with globally skipped NMR/WIP files:
  `tests/test_processing/test_fft/test_nmr.py` and
  `tests/test_processing/test_filter/test_smooth.py`. Their skip reasons are now
  explicit, but their long-term home is still unresolved.

### P1: Soon

- Move NMR fixtures out of global `tests/conftest.py` or guard them with an
  explicit plugin/data availability helper.
- Add explicit `OPTIONAL_PLUGIN` markers to every plugin-dependent example or
  generated docs script.
- Split slow IRIS numerical tests from fast IRIS API/lazy-loading tests.

### P2: Later Cleanup

- Introduce a small shared plugin-availability helper for tests and examples if
  duplication grows.
- Add optional IRIS lazy proxy introspection tests if regressions appear.
- Revisit warning volume in IRIS tests, especially OSQP warnings, without hiding
  real numerical issues.
- Consider a clearer test taxonomy:
  core unit tests, plugin integration tests, plugin package tests, docs/example
  tests, and slow numerical regression tests.

## Commands Executed

All commands were run in the `scpy` conda environment unless noted.

- `pytest tests/test_plugins -q -ra`
  - Result: `205 passed in 57.05s`
- `pytest plugins/spectrochempy-iris/tests -q -ra`
  - Result: `27 passed, 1 skipped, 1728 warnings in 23.40s`
- `pytest plugins/spectrochempy-cantera/tests -q -ra`
  - Result: `13 passed, 1 skipped in 2.65s`
- `pytest plugins/spectrochempy-nmr/tests -q -ra`
  - First run exposed one brittle proxy-identity assertion.
  - After the small patch: `16 passed, 4 warnings in 10.54s`

`pre-commit run --all-files` was intentionally not run during this audit pass to
avoid broad formatting churn. Run it once only at the final synchronization
checkpoint.
