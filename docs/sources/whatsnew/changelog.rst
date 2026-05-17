
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
  extracted into standalone plugin at ``plugins/spectrochempy-nmr/``.
- IRIS plugin (``spectrochempy-iris``): IRIS analysis and plotting extracted
  into standalone plugin at ``plugins/spectrochempy-iris/``.
- Cantera plugin (``spectrochempy-cantera``): Plug Flow Reactor simulation
  utilities added at ``plugins/spectrochempy-cantera/``.

.. section

Bug Fixes
~~~~~~~~~
.. Add here new bug fixes (do not delete this comment)


.. section

Dependency Updates
~~~~~~~~~~~~~~~~~~
.. Add here new dependency updates (do not delete this comment)


.. section

Breaking Changes
~~~~~~~~~~~~~~~~
.. Add here new breaking changes (do not delete this comment)

- Reader functions (e.g. ``read_omnic``, ``read_opus``) are no longer
  available as ``NDDataset`` methods. Use ``scp.read_omnic(...)`` or
  plugin-namespaced access (e.g. ``scp.nmr.read_topspin(...)``) instead.
- ``read_topspin`` removed from core — requires ``pip install spectrochempy[nmr]``.
- ``IRIS`` and ``IrisKernel`` removed from core — requires ``pip install spectrochempy[iris]``.

.. section

Deprecations
~~~~~~~~~~~~
.. Add here new deprecations (do not delete this comment)
