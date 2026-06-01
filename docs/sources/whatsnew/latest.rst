:orphan:

What's New in Revision 0.9.2.dev
---------------------------------------------------------------------------------------

These are the changes in SpectroChemPy-0.9.2.dev.
See :ref:`release` for a full changelog, including other versions of SpectroChemPy.

New Features
~~~~~~~~~~~~

- Added plugin lifecycle tests for ``spectrochempy-hypercomplex``,
  ``spectrochempy-carroucell``, ``spectrochempy-nmr``, and
  ``spectrochempy-iris`` covering import, metadata, compatibility,
  registration, lifecycle state, reader/accessor registration,
  and isolated harness behaviour.
- Added ``_ensure_carroucell_filetype_registered()`` to register the
  ``carroucell`` filetype with the legacy :class:`~spectrochempy.core.readers.filetypes.FileTypeRegistry`,
  fixing ``TypeError: Filetype '.carroucell' is unknown`` when reading
  CarrouCELL data.
- Fixed test isolation in ``spectrochempy-iris`` tests: monkeypatched
  :func:`sys.modules` cleanup ensures each test uses its own harness
  when validating the ``scp.iris`` namespace.

Breaking Changes
~~~~~~~~~~~~~~~~

- Removed deprecated compatibility kwargs in :class:`~spectrochempy.analysis.decomposition.efa.EFA`,
  :class:`~spectrochempy.analysis.decomposition.pca.PCA`,
  and :class:`~spectrochempy.analysis.decomposition.nmf.NMF`:
  ``used_components`` (use ``n_components``).
- Removed deprecated compatibility kwargs in :class:`~spectrochempy.analysis.decomposition.simplisma.SIMPLISMA`:
  ``verbose`` (use ``log_level``),
  ``n_pc`` (use ``n_components``),
  ``max_components`` (use ``n_components``).
- Removed deprecated compatibility kwargs in :func:`~spectrochempy.processing.baselineprocessing.baselineprocessing.detrend`:
  ``type`` (use ``order``), ``dim`` (no longer needed), ``inplace`` (no longer supported).
- Removed deprecated :meth:`~spectrochempy.utils.metaconfigurable.MetaConfigurable.parameters`
  method (use :meth:`~spectrochempy.utils.metaconfigurable.MetaConfigurable.params` instead).
- Removed deprecated ``n_pc`` fallback in
  :meth:`~spectrochempy.analysis._base._analysisbase.DecompositionAnalysis.transform`
  and :meth:`~spectrochempy.analysis._base._analysisbase.DecompositionAnalysis.inverse_transform`
  (use ``n_components`` instead).

Deprecations
~~~~~~~~~~~~

- Removed ``IRIS.plotdistribution()`` from the ``spectrochempy-iris`` plugin
  (target ``removed="0.9.0"`` was reached). Use ``IRIS.f[index].plot()`` instead.
- Added explicit ``removed="0.10.0`` to all remaining API deprecations that
  did not yet specify a removal version :
  ``Baseline.show_regions``, ``PCA.screeplot``, ``PCA.scoreplot``,
  ``DecompositionAnalysis.reduce``, ``DecompositionAnalysis.reconstruct``,
  ``MCRALS.St_unconstrained``, ``MCRALS.S_soft``,
  ``concatenate(force_stack=...)``, ``read_jdx``, ``read_dx``,
  ``trapz``, ``multiplot_stack``, ``multiplot_map``, ``multiplot_image``,
  ``Project.remove_all_dataset``, ``Project.remove_all_project``,
  ``Project.remove_all_script``, ``restore_rcparams``,
  ``get_import_time_rcparams``, and legacy plot-method aliases
  (``stack`` → ``lines``, ``map`` → ``contour``, ``image`` → ``contourf``).
