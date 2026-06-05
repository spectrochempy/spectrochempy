
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


.. section

Bug Fixes
~~~~~~~~~
.. Add here new bug fixes (do not delete this comment)

- Fixed ``NDDataset.squeeze()`` for singleton dimensions without explicit
  coordinates and aligned Labspec in-memory byte decoding with the Latin-1
  fallback already used for local files.
- Fixed coordinate propagation in ``@_wrap_ndarray_output_to_nddataset``
  when ``typey="features"`` and ``typex="components"`` are both set —
  replaced sequential ``if`` blocks with an ``elif`` chain and added an
  explicit combined case to prevent conflicting coordinate assignments.


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

- MAINT: Modernized PLS regression tests (``test_pls.py``): replaced the
  network-dependent Corn dataset with a deterministic synthetic latent-variable
  fixture (seeded RNG), split the monolithic test into 15 focused functions
  across three test classes covering univariate/multivariate fits, score,
  predict, transform, fit_transform, inverse_transform, and masked-data
  handling, removed all plotting from algorithmic validation, and strengthened
  assertions with sklearn numerical parity checks.

- MAINT: Modernized selected analysis tests with deterministic synthetic datasets,
  reducing unnecessary real-data dependencies and strengthening numerical,
  shape, metadata, masking, and reporting assertions.

- MAINT: Modernized SIMPLISMA analysis tests using deterministic synthetic datasets,
  removing the dependency on ``als2004dataset.MAT`` and strengthening validation
  of pure-variable selection, decomposition outputs, reconstruction behavior,
  and error handling.

- MAINT: Modernized FastICA analysis tests using deterministic synthetic ICA
  mixtures, removing the dependency on ``als2004dataset.MAT`` and strengthening
  wrapper parity, transform, inverse-transform, masking, and validation
  coverage.

- MAINT: Modernized baseline analysis tests using deterministic synthetic
  datasets, removing dependencies on external IR and MS test files and
  strengthening validation of baseline estimation, masking, preprocessing APIs,
  and multivariate baseline workflows.

- MAINT: Converted NMF analysis tests from the legacy MATLAB fixture to a
  deterministic non-negative synthetic dataset with focused parity,
  reconstruction, masking, metadata, and non-negativity checks.

- MAINT: Moved IRIS ``plot_merit`` coverage out of core analysis tests into the
  IRIS plugin test suite, clarifying plugin ownership for plugin/data plotting
  integration checks.

- MAINT: Moved IRIS plugin import/export checks (``TestIrisImports`` for
  ``plot_iris_lcurve``, ``plot_iris_distribution``, ``plot_iris_merit``) from
  the core composite plotting test module into the IRIS plugin test suite,
  removing all direct IRIS plugin runtime import references from the core
  test tree.

- MAINT: Classified plugin-dependent tests with explicit markers and skip guards
  so optional plugin imports do not fail during core-only collection, while
  keeping core-only and plugin/integration validation clearly separated.

- MAINT: Modernized core dataset tests with stronger ``NDDataset``,
  ``NDMath``, ``Coord``, ``CoordSet``, ``NDArray``, and ``NDComplexArray``
  validation and split the large dataset test module into focused test files.

- MAINT: Converted legacy plotting tests (1D, 2D, 3D, multiplot) from external
  spectroscopy files to deterministic synthetic fixtures, making them runnable
  in core-only CI without testdata downloads.

- MAINT: Converted PCA plotting tests (``plot_score``, ``plot_scree``) from
  external IR data to deterministic synthetic datasets; decoupled 5 tests from
  PCA computation using precomputed score fixtures.

- MAINT: Classified documentation plotting tests as ``docs``/``integration``
  tests with explicit ``pytest.mark.docs`` and ``pytest.mark.data`` markers;
  added graceful skip when IR testdata is unavailable.

- MAINT: Registered ``docs`` and ``data`` pytest markers in the top-level
  conftest for use across the test suite.

- MAINT: Classified all reader/writer I/O tests with explicit
  ``pytest.mark.data`` (real instrument format files) or
  ``pytest.mark.network`` (remote downloads); core-safe synthetic I/O tests are
  selectable via ``-m "not data and not network"``.

- MAINT: Converted ``test_write_csv.py`` from a real IR dataset fixture to
  deterministic synthetic ``NDDataset`` and ``Coord`` objects, removing its
  hidden external testdata dependency.

- MAINT: Split the monolithic ``test_read.py::test_read`` into focused tests
  with explicit marker separation and shared skip helpers.

- DEV: Added TODO/FIXME note in
  ``docs/sources/devguide/plugins/packaging.rst`` explaining that dev conda
  uploads for plugins are disabled until distinct dev versions are generated.

- DEV: Added reusable plugin version-status tooling and temporary
  ``next_patch.devN`` metadata injection for plugin development builds, with
  automatic conda uploads to ``spectrocat/label/dev`` on ``master`` when
  release-relevant plugin files changed.

- CI: Added a dedicated core-only package-test validation row that skips
  official plugin installation, plugin diagnostics, documentation script
  execution, external testdata restore, coverage generation, and Codecov upload.

- CI: Aligned the plugin release workflow status summary with the shared
  release-relevant plugin change detector used by the read-only status workflow.

- CI: Restored unique TestPyPI development uploads by deriving core push builds
  from the next patch version and publishing plugin TestPyPI builds from
  ``master`` only when release-relevant plugin files changed.

- CI: Limited the slow Colab compatibility workflow to manual runs and pull
  requests labeled ``needs-colab``.

- CI: Added an explicit documentation versions manifest and a manual
  ``Repair docs version index`` workflow to refresh the published version
  dropdown without rebuilding the full documentation.

- CI: Allowed documentation builds to use canonical core release tags such as
  ``spectrochempy-v0.9.2`` directly, while publishing versioned docs under the
  plain semver directory.

- CI: Added plugin status table to ``release_plugin.yml`` step summary,
  listing official plugins and whether they have changed since their last
  release tag.

- CI: Added ``confirm_zenodo_enabled`` checkbox to
  ``prepare_new_release.yml`` so maintainers verify Zenodo is active before
  creating a release PR.

- CI: Disabled automatic dev conda uploads for official plugins. Plugin conda
  packages are still built in CI for testing, but only uploaded during stable
  plugin releases.

- CI: Added ``plugin_release_status.yml`` workflow
  (``workflow_dispatch``) to inspect official plugin changes since their last
  release tag before deciding to run a release.

- DOC: Improved maintainer release documentation with explicit Zenodo
  verification steps, versioned docs checks, docs build notes, and roadmap
  guidance.

- DOC: Clarified that official plugin versions are independent from the core
  package version and that conda dev plugin builds are not currently published
  automatically.

- DOC: Updated contributor documentation build instructions to use the
  repository-root ``docs/make.py`` entry point and current ``build/html``
  output.

- DOC: Updated optional plugin installation notes so conda development
  channels no longer imply automatic plugin dev uploads.

- DOC: Aligned workflow comments and recovery notes with the current plugin
  conda policy.

- DOC: Updated plugin release documentation to use canonical core tags,
  current branch pushes, plugin ``__init__.py`` version bumps, and current
  Anaconda upload behavior.

- DOC: Clarified that official plugin releases must be run from ``master`` and
  can be prepared with the read-only plugin release status workflow.

- DOC: Finished maintainer release notes cleanup for master checkout, draft
  release titles, TestPyPI wording, plugin PyPI ``skip-existing`` behavior, and
  recovery links.

- DOC: Tightened the developer guide entry points, Git commands, pull request
  template guidance, and testing guidance.
