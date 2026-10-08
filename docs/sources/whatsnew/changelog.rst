
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

- The TopSpin reader accepts a ``proc_axis`` keyword to select the
  processing parameters (``OFFSET``, ``SW_p``, ``SF`` in
  ``procs``/``proc2s``) as the source for processed spectral axes,
  following the TopSpin convention (first point at ``OFFSET``, bin width
  ``SW_p / (SF * SI)``).  The default preserves the historical
  acquisition-based axis; the two references can disagree when a spectrum
  was re-referenced during processing.  This selects the axis parameter
  source and does not replay or apply vendor processing.  Only full
  spectra are supported (``STSR``/``STSI`` extraction triggers a warning
  and falls back to the acquisition-based axis)
  (:issue:`1742`).

- The NMR reader now accepts a ``remove_dc_offset`` keyword argument
  (default ``False``). When ``True``, the receiver DC offset is removed
  from the FID before digital filter correction, which eliminates the
  spike at the centre of the spectrum caused by the receiver electronics
  (:pr:`1745`).

.. section

Bug Fixes
~~~~~~~~~
.. Add here new bug fixes (do not delete this comment)

- Selection by coordinate value on a dimension with multiple coordinates now
  uses the selected default coordinate and handles quantities as for a single
  coordinate, instead of raising a ``TypeError`` (:issue:`1751`).

- The NMR digital filter removal algorithm in the TopSpin reader has been
  rewritten. The previous implementation introduced a constant phase
  rotation (``exp(i*pi*phase)``) with no physical justification, added a
  flat pedestal to the spectrum via an incorrect DC subtraction, derived the
  output length from ``TD//2`` instead of the actual data size, and
  mutated the input dictionary. All four defects are fixed (:pr:`1745`).

- The TopSpin reader now resolves non-numeric experiment directories
  (e.g. ``my_experiment/1/fid``) without returning ``None``.  Previously,
  a non-numeric parent directory name caused ``int(expno)`` to fail, and
  the generic exception handler silently returned ``None``.  The resolver
  now discovers experiment directories with Bruker data files regardless
  of the parent directory name.

- Component processed data files (``1i``, ``2ri``, ``2ir``, ``2ii``) are
  now accepted as entry points to the full assembled spectrum.  Previously,
  these files were excluded from the valid filename set, causing a futile
  remote download attempt followed by a bare ``FileNotFoundError``.
  Reading ``2ri`` returns the same quaternion spectrum as ``2rr`` (all four
  components assembled); reading ``1i`` returns the same complex spectrum
  as ``1r``.  Individual channel isolation is not supported.

- The TopSpin test suite no longer skips ``test_read_topspin`` due to a
  404 download error.  Local fixture assertions are now independent of
  remote download behaviour.

- TopSpin spectra read from processed data are now consistently treated as
  frequency-domain data by FFT processing.  Calling ``fft`` on an already
  frequency-domain dimension now raises an explicit diagnostic naming the
  incompatible dimension, while raw FID/SER data and remaining time-domain
  dimensions in partially transformed 2D data remain transformable.


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


.. section

Developer
~~~~~~~~~
.. Add here developer changes (do not delete this comment)
