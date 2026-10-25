
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

- Fixed the Agilent TPPI direct FFT path so ordinary complex acquisitions are
  no longer interpreted as quaternion data, preventing non-finite results.
- Filter outputs now retain the applied method, effective dimension, and
  compact filter parameters in their structured history entries.
- Whittaker-Eilers filtering now applies along the requested dimension.
- Baseline correction outputs now record the fitted model and its effective
  parameters in their structured history entry instead of only an arithmetic
  subtraction.
- ``basc()`` and ``get_baseline()`` now honor ``ranges=`` passed as a keyword
  instead of silently overwriting it with the positional ranges.

- Restored parsing of unit expressions that use a dot as a separator (e.g.
  ``kg.m`` or ``cm.s^-1``), which pint 0.26.2 rejects. (#1781)
- Fix ``directory=`` imports with ``merge=True`` so compatible datasets
  discovered under different file extensions can be merged across extension
  groups, rather than being returned separately. (#1780)

- Fixed ``scp.simpson`` so it always resolves to the public numerical
  integration function ``simpson(dataset, dim='x')``, even when a plugin
  contributes a SIMPSON I/O reader; ``scp.read_simpson`` and
  ``scp.nmr.read_simpson`` remain the explicit file-reading surfaces and
  ``dataset.simpson()`` is unchanged.
- Correct OMNIC ``.spa`` interferogram optical-path-difference coordinates by
  honoring the native header sample-spacing factor instead of always assuming
  a doubled step.
- Correct OMNIC ``.srs`` series time-axis anchoring (it now starts from the
  recorded series minimum time) and fix per-spectrum labels, which no longer
  include binary metadata leaked past the 84-byte record.
- Normalize OMNIC ``.srs`` spectral series to the public ``read_spa``
  convention: wavenumbers are exposed descending with the intensity data
  matched to them. Rapid-scan interferograms keep their ascending data-points
  axis and records with an unrecognized X-axis type are left unchanged with a
  warning.

- TopSpin imports now report when hypercomplex data require the optional
  ``spectrochempy-hypercomplex`` and ``numpy-quaternion`` dependencies. (#1743)


.. section

Dependency Updates
~~~~~~~~~~~~~~~~~~
.. Add here new dependency updates (do not delete this comment)

- Updated the documentation dependency ``nbconvert`` to 7.17.2 to address an
  arbitrary-file-write vulnerability when converting untrusted notebooks.


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
