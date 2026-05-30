
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


.. section

Deprecations
~~~~~~~~~~~~
.. Add here new deprecations (do not delete this comment)

- Removed deprecated :class:`~spectrochempy.core.dataset.coord.LinearCoord` class — use :class:`~spectrochempy.core.dataset.coord.Coord` instead.
- Removed deprecated compatibility kwargs in :class:`~spectrochempy.analysis.decomposition.mcrals.MCRALS`:
  ``verbose`` (use ``log_level``), ``unimodTol`` (use ``unimodConcTol``),
  ``unimodMod`` (use ``unimodConcMod``), ``hardC_to_C_idx`` (use ``getC_to_C_idx``),
  ``hardSt_to_St_idx`` (use ``getSt_to_St_idx``).
- Removed deprecated compatibility kwargs in :func:`~spectrochempy.processing.filter.filter.smooth`:
  ``window_length`` (use ``size``).
- Removed deprecated compatibility kwargs in :func:`~spectrochempy.processing.filter.filter.savgol`:
  ``window_length`` (use ``size``) and ``polyorder`` (use ``order``).
