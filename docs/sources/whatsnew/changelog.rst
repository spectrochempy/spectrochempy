
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

- Add :mod:`spectrochempy-hypercomplex` to CI test selection and install from local checkout in docs CI.
- Vectorise :func:`~spectrochempy_hypercomplex._quaternion.as_quaternion` with :func:`numpy.stack` for better performance.
- Add :mod:`bump_plugin_init_version.py` script to bump plugin class ``version`` attribute on release, with cross-source verification.
- Sync plugin class ``version`` to ``0.1.1`` for all official plugins, matching ``pyproject.toml``.


.. section

Bug Fixes
~~~~~~~~~
.. Add here new bug fixes (do not delete this comment)

- Fix 2D hypercomplex NMR coordinate mismatch when :meth:`~spectrochempy_hypercomplex.set_quaternion` halves the last dimension.
- Fix :meth:`~spectrochempy.core.dataset.basearrays.ndcomplex.NDComplex.real` to handle quaternion dtype, enabling plotting of 2D hypercomplex NMR data.
- Restrict :mod:`setuptools-scm` to ``spectrochempy-v*`` tags so development versions derive from the correct tag.
- Replace fragile ``.T`` pattern with ``float_arr[..., 0:4]`` in :func:`~spectrochempy_hypercomplex._quaternion.quat_as_complex_array` and :func:`~spectrochempy_hypercomplex._quaternion.get_component` for correct ND handling.


.. section

Dependency Updates
~~~~~~~~~~~~~~~~~~
.. Add here new dependency updates (do not delete this comment)


.. section

Breaking Changes
~~~~~~~~~~~~~~~~
.. Add here new breaking changes (do not delete this comment)

- Removed deprecated :class:`~spectrochempy.core.dataset.coord.LinearCoord` class — use :class:`~spectrochempy.core.dataset.coord.Coord` instead.
- Removed deprecated compatibility kwargs in :class:`~spectrochempy.analysis.decomposition.mcrals.MCRALS`:
  ``verbose`` (use ``log_level``), ``unimodTol`` (use ``unimodConcTol``),
  ``unimodMod`` (use ``unimodConcMod``), ``hardC_to_C_idx`` (use ``getC_to_C_idx``),
  ``hardSt_to_St_idx`` (use ``getSt_to_St_idx``).
- Removed deprecated compatibility kwargs in :func:`~spectrochempy.processing.filter.filter.smooth`:
  ``window_length`` (use ``size``).
- Removed deprecated compatibility kwargs in :func:`~spectrochempy.processing.filter.filter.savgol`:
  ``window_length`` (use ``size``) and ``polyorder`` (use ``order``).


.. section

Deprecations
~~~~~~~~~~~~
.. Add here new deprecations (do not delete this comment)
