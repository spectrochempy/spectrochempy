:orphan:

What's New in Revision 0.8.3.dev
---------------------------------------------------------------------------------------

These are the changes in SpectroChemPy-0.8.3.dev.
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

Breaking Changes
~~~~~~~~~~~~~~~~

- Reader functions (e.g. ``read_omnic``, ``read_opus``) are no longer
  available as ``NDDataset`` methods. Use ``scp.read_omnic(...)`` or
  plugin-namespaced access (e.g. ``scp.nmr.read_topspin(...)``) instead.
