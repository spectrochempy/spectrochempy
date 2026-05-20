:orphan:

What's New in Revision 0.8.5.dev
---------------------------------------------------------------------------------------

These are the changes in SpectroChemPy-0.8.5.dev.
See :ref:`release` for a full changelog, including other versions of SpectroChemPy.

New Features
~~~~~~~~~~~~

- Core plugin system: external packages can now register readers, writers,
  processors, visualizers, analyses, simulations, and dataset accessors via
  ``spectrochempy.plugins`` entry points.
- Plugin API: stable public interface at ``spectrochempy.api.plugins`` for
  plugin authors, with versioned API guarantees.
- Plugin template: cookie-cutter project template in ``plugins/plugin-template/``.
- Plugin test harness: ``PluginTestHarness`` in ``spectrochempy.testing.plugins``
  for isolated plugin testing without global state leakage.
- Plugin documentation: developer guide covering architecture, API policy,
  packaging, and testing.
- NMR plugin (``spectrochempy-nmr``): TopSpin/Bruker reader and NMR utilities
  are now provided by an official plugin at ``plugins/spectrochempy-nmr/``.
- IRIS plugin (``spectrochempy-iris``): IRIS analysis and plotting are now
  provided by an official plugin at ``plugins/spectrochempy-iris/``.
- Cantera plugin (``spectrochempy-cantera``): Plug Flow Reactor simulation
  utilities added at ``plugins/spectrochempy-cantera/``.

Breaking Changes
~~~~~~~~~~~~~~~~

- Reader functions (e.g. ``read_omnic``, ``read_opus``) are no longer
  available as ``NDDataset`` methods. Use ``scp.read_omnic(...)`` or
  plugin-namespaced access (e.g. ``scp.nmr.read_topspin(...)``) instead.
- TopSpin reading is now provided by the official NMR plugin. Use
  ``scp.nmr.read_topspin(...)`` after installing ``spectrochempy[nmr]``.
  The historical ``scp.read_topspin(...)`` alias remains available as a
  compatibility layer.
- IRIS analysis is now provided by the official IRIS plugin. Use
  ``scp.iris.IRIS`` or ``from spectrochempy.iris import IRIS`` after installing
  ``spectrochempy[iris]``. Limited root-level compatibility aliases remain
  available with deprecation warnings.
