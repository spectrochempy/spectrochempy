:orphan:

What's New in Revision 0.9.2.dev
---------------------------------------------------------------------------------------

These are the changes in SpectroChemPy-0.9.2.dev.
See :ref:`release` for a full changelog, including other versions of SpectroChemPy.

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
