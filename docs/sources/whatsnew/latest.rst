:orphan:

What's New in Revision 0.9.1.dev
---------------------------------------------------------------------------------------

These are the changes in SpectroChemPy-0.9.1.dev.
See :ref:`release` for a full changelog, including other versions of SpectroChemPy.

Deprecations
~~~~~~~~~~~~

- Removed deprecated :class:`~spectrochempy.core.dataset.coord.LinearCoord` class — use :class:`~spectrochempy.core.dataset.coord.Coord` instead.
- Removed deprecated compatibility kwargs in :class:`~spectrochempy.analysis.decomposition.mcrals.MCRALS`:
  ``verbose`` (use ``log_level``), ``unimodTol`` (use ``unimodConcTol``),
  ``unimodMod`` (use ``unimodConcMod``), ``hardC_to_C_idx`` (use ``getC_to_C_idx``),
  ``hardSt_to_St_idx`` (use ``getSt_to_St_idx``).
- Removed deprecated compatibility kwargs in :func:`~spectrochempy.processing.filter.filter.smooth`:
  ``window_length`` (use ``size``).
- Removed deprecated compatibility kwargs in :func:`~spectrochempy.processing.filter.filter.savgol`:
  ``window_length`` (use ``size``) and ``polyorder`` (use ``order``).
