
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

- Added ``scp.Pipeline``, a public minimal Pipelines class for linear,
  reproducible composition of SpectroChemPy estimators: allowlisted
  preprocessing transformers optionally followed by a final transformer
  (``PCA``) or a supervised estimator (``PLSRegression``, ``LSTSQ``, ``NNLS``).
  Steps are templates cloned on fit (``fitted_steps_`` /
  ``fitted_named_steps_``); ``transform`` / ``fit_transform`` are available
  for transformer-final pipelines and ``predict`` / ``score`` for
  estimator-final pipelines, with nested ``set_params`` support and fitted-state
  invalidation.

.. section

Bug Fixes
~~~~~~~~~
.. Add here new bug fixes (do not delete this comment)

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

- The ``reverse_x`` keyword argument of ``read_srs`` is deprecated: SRS
  spectral orientation is now handled automatically. Supplying it (with any
  value) emits a ``DeprecationWarning`` and is ignored.


.. section

Developer
~~~~~~~~~
.. Add here developer changes (do not delete this comment)

MAINT: Added a centralized reserved-root-symbol policy
(``spectrochempy/lazyimport/root_symbols.py``) so plugins cannot shadow
public ``scp`` symbols; conflicting plugin I/O namespaces are rejected at
registration with a controlled warning while the ``read_<format>`` reader
remains available through its explicit surfaces. (#1599)

MAINT: Added the internal estimator-contract helpers required by the
``Pipeline`` implementation: allowlist-based fitted-state inspection,
unfitted cloning, canonical not-fitted behavior for supported transformers,
and opt-in lifecycle invalidation for the accepted analysis terminal
candidates. (#1589)

DOC: Added developer-guide references for the OMNIC SRS and SPA file
formats. (#1596, #1597)
