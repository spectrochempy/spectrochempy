
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

- Added ``scp.Pipeline`` for linear, reproducible composition of SpectroChemPy
  estimators: allowlisted
  preprocessing transformers optionally followed by a final transformer
  (``PCA``) or a supervised estimator (``PLSRegression``, ``LSTSQ``, ``NNLS``).
  Steps are templates cloned on fit (``fitted_steps_`` /
  ``fitted_named_steps_``); ``transform`` / ``fit_transform`` are available
  for transformer-final pipelines and ``predict`` / ``score`` for
  estimator-final pipelines, with nested ``set_params`` support and fitted-state
  invalidation. (#1590)

- OMNIC ``.spa`` imports now expose additional acquisition metadata in
  ``dataset.meta`` when available for the file variant: scan and FFT point
  counts, sample and background scan counts and gains, interferogram peak
  position, aperture, digitizer bit depth, filter settings, and reference
  frequency. (#1606)

.. section

Bug Fixes
~~~~~~~~~
.. Add here new bug fixes (do not delete this comment)

- Fixed ``scp.simpson`` so it always resolves to the public numerical
  integration function ``simpson(dataset, dim='x')``, even when a plugin
  contributes a SIMPSON I/O reader; ``scp.read_simpson`` and
  ``scp.nmr.read_simpson`` remain the explicit file-reading surfaces and
  ``dataset.simpson()`` is unchanged. (#1599)
- Corrected OMNIC ``.spa`` interferogram optical-path-difference coordinates by
  honoring the native header sample-spacing factor instead of always assuming
  a doubled step. (#1598)
- Corrected OMNIC SPA metadata decoding: optical velocity now comes from the
  canonical acquisition-parameter block when available, with a fallback for
  older files; Raman ``laser_frequency`` now reports the excitation frequency
  separately from ``reference_frequency``. Experiment Information in SPA and
  SPG files now correctly distinguishes the path, title, description, and
  accessory fields. (#1603, #1605, #1606)
- Recognized library/retrieved OMNIC ``.spa`` variants no longer receive a
  spurious acquisition date. They use an undated spectrum coordinate instead
  of interpreting a non-acquisition header value as a timestamp. (#1605)
- Corrected OMNIC ``.srs`` series time-axis anchoring (it now starts from the
  recorded series minimum time) and fixed per-spectrum labels, which no longer
  include binary metadata leaked past the 84-byte record. (#1593)
- Normalized OMNIC ``.srs`` spectral series to the public ``read_spa``
  convention: wavenumbers are exposed descending with the intensity data
  matched to them. Rapid-scan interferograms keep their ascending data-points
  axis and records with an unrecognized X-axis type are left unchanged with a
  warning. (#1594, #1595)


.. section

Dependency Updates
~~~~~~~~~~~~~~~~~~
.. Add here new dependency updates (do not delete this comment)


.. section

Breaking Changes
~~~~~~~~~~~~~~~~
.. Add here new breaking changes (do not delete this comment)

- OMNIC Experiment Information no longer exposes
  ``dataset.meta.omnic_experiment_file``. Use
  ``dataset.meta.omnic_experiment_title`` for the native title/name field;
  it is not a reliable filename. The separate description is now exposed as
  ``dataset.meta.omnic_experiment_description``. (#1605)


.. section

Deprecations
~~~~~~~~~~~~
.. Add here new deprecations (do not delete this comment)

- The ``reverse_x`` keyword argument of ``read_srs`` is deprecated: SRS
  spectral orientation is now handled automatically. Supplying it (with any
  value) emits a ``DeprecationWarning`` and is ignored. (#1594, #1595)


.. section

Developer
~~~~~~~~~
.. Add here developer changes (do not delete this comment)

MAINT: Refactored the OMNIC SPA reader to separate native parsing, signal
selection, coordinate construction, metadata attachment, and dataset
finalization, preserving spectral and standalone/associated interferogram
reading modes. (#1604, #1606, #1607)

MAINT: Centralized the reserved-root-symbol policy so plugins cannot shadow
public ``scp`` symbols; conflicting plugin I/O namespaces are rejected at
registration with a controlled warning while the ``read_<format>`` reader
remains available through its explicit surfaces. (#1599)

MAINT: Standardized fitted-state inspection, unfitted cloning, and lifecycle
invalidation for the estimators supported by ``Pipeline``. (#1589)

DOC: Expanded the Pipeline user guide and documented the validated OMNIC SRS
and SPA file layouts and SPA reader behavior. (#1592, #1596, #1597, #1602, #1607)
