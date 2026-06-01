
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


.. section

Bug Fixes
~~~~~~~~~
.. Add here new bug fixes (do not delete this comment)

- Fixed missing ``import sys`` in :mod:`~spectrochempy.utils.docutils` that caused
  ``NameError`` when validating docstrings with ``Examples`` sections
  (the root cause of "docstring checker unstable in CI" skips).
- Fixed docstring examples in :meth:`~spectrochempy.analysis.decomposition.pca.PCA.plot_score`
  and :meth:`~spectrochempy.analysis.decomposition.pca.PCA.plot_scree`:
  removed auto-imported ``import spectrochempy`` and used ``_ = pca.fit(X)``
  to suppress output.
- Fixed docstring examples in :meth:`~spectrochempy.core.dataset.nddataset.NDDataset.reshape`:
  added ``doctest: +SKIP`` to examples that required undefined ``X`` variable.
- Fixed ``See Also`` entries in :meth:`~spectrochempy.core.dataset.nddataset.NDDataset.plot`
  that used incorrect ``spectrochempy.plotting.*`` prefix (should omit ``spectrochempy.``).
- Resolved ``%(analysis_fit.parameters.X)s`` docrep template leftovers in
  :meth:`~spectrochempy.analysis.crossdecomposition.pls.PLSRegression.fit`
  by replacing with the actual parameter text (docrep dependency was removed in 0.8.4).
- Reactivated 4 docstring validation tests for ``PLSRegression``, ``PCA``, ``NDDataset``,
  and ``IRIS`` that had been disabled as "unstable in CI".
- Removed stale commented-out ``@pytest.mark.skipif`` blocks from 5 reader test files
  (test data is available in CI).


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
