:orphan:

What's New in Revision 0.8.2.dev
---------------------------------------------------------------------------------------

These are the changes in SpectroChemPy-0.8.2.dev.
See :ref:`release` for a full changelog, including other versions of SpectroChemPy.

New Features
~~~~~~~~~~~~

- The welcome message display functionality has been removed.
- Implemented lazy matplotlib initialization: SpectroChemPy now loads matplotlib only when plotting is actually used, reducing import-time overhead for non-plotting workflows.
- Full interpolation feature for NDDataset (issue #673)
- Added CP (Candecomp/Parafac)
- Added PSD (Phase Sensitive Detection)
- New plugin infrastructure: external packages can now extend SpectroChemPy via ``spectrochempy.plugins`` entry points. Readers, writers, file types, unit contexts, and dtype handlers are auto-discovered at runtime.
- The Bruker TopSpin NMR reader has been moved to an external plugin package ``spectrochempy-topspin``. Install with ``pip install spectrochempy-topspin`` — ``scp.read_topspin(...)`` works transparently when the plugin is installed.
- Improved plugin developer experience: declarative contribution hooks, lifecycle tracking with introspection API, error isolation, plugin validation helpers, and a ``PluginTestHarness`` for isolated testing.
- New ``spectrochempy-testing`` subpackage with ``PluginTestHarness`` and ``spectrochempy.api.plugins`` now exports all SDK symbols (contribution dataclasses, lifecycle types, error classes).
- New plugin template and developer documentation (testing guide, packaging guide, validation reference).

Bug Fixes
~~~~~~~~~

- Refactored SpectroChemPy initialization and plotting setup to reduce Matplotlib side effects (issue #877). SpectroChemPy no longer modifies global ``matplotlib.rcParams``.
- Improved handling of Matplotlib styles, separating logical styles (e.g. ``default``) from file-based ``.mplstyle`` styles.
- Stabilized plot preferences reset and style application across tests, docs, and CI environments.
- Fixed scikit-learn ``PLSRegression`` compatibility with scikit-learn >= 1.5.
- Fixed issue #858 (OMNIC series reader): Added ``reverse_x`` parameter for wavenumber ordering.
- Fixed issue #856 (osqp dependency, used for IRIS): Added warning for osqp < 1.0.
- Fixed MCRALS: Corrected indexing for intermediate spectral matrices (C and St lists).
- Fixed issue #875: Cannot subtract offset-naive and offset-aware datetimes.
- Fixed issue #911: MCR-ALS ``closureConc="all"`` no longer raises "The truth value of an array with more than one element is ambiguous".
- Fixed quaternion ``abs()`` and NumPy 2.0 dtype compatibility issues.
- Fixed ``dtype == typequaternion`` false detection on float64 arrays.
- Fixed ``NDMath.all`` builtin shadowing and generator iteration over objects without ``.dtype``.
- Fixed Python 3.14 ``copy`` compatibility with optional quaternion guards.

Dependency Updates
~~~~~~~~~~~~~~~~~~

- Major Python compatibility updates:
    - Maximum Python version increased to 3.14
    - Minimum Python version increased to 3.11
    - Dropped support for Python 3.10 and below
- osqp > 1.0 is now allowed (#856). A warning is shown for osqp < 1.0, which will not be supported in future versions.
- removed docrep dependency
- Added ``pluggy`` as a core dependency for the plugin hook system.
- Moved ``numpy-quaternion`` from a hard dependency to an optional dependency (``[nmr]`` extra). Install with ``pip install spectrochempy[nmr]``.

Breaking Changes
~~~~~~~~~~~~~~~

- Dropped support for Python 3.10 and below.
- ``restore_rcparams()`` is now deprecated: SpectroChemPy no longer modifies global ``matplotlib.rcParams``, so restoration is unnecessary.
- Plotting initialization behavior has changed internally to reduce global Matplotlib side effects. While intended to be backward compatible, code relying on implicit global ``rcParams`` side effects may behave differently.
- The Bruker TopSpin NMR reader (``read_topspin``) has been moved out of the core into an external plugin. The ``spectrochempy`` package provides a stub that raises ``MissingPluginError`` with install instructions. Install ``pip install spectrochempy-topspin`` to restore the reader.
- ``numpy-quaternion`` is no longer a hard dependency. It is an optional dependency installable via ``pip install spectrochempy[nmr]``. Core functionality (complex arrays, FFT, math operations) remains unaffected; quaternion (hypercomplex) support requires the ``numpy-quaternion`` package.

Deprecations
~~~~~~~~~~~~

- ``IRIS.plotdistribution(index)`` is now deprecated. Use ``IRIS.f[index].plot()`` instead.
