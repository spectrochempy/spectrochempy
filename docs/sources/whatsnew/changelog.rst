
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


.. section

Dependency Updates
~~~~~~~~~~~~~~~~~~
.. Add here new dependency updates (do not delete this comment)


.. section

Breaking Changes
~~~~~~~~~~~~~~~~
.. Add here new breaking changes (do not delete this comment)

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


.. section

Deprecations
~~~~~~~~~~~~
.. Add here new deprecations (do not delete this comment)
