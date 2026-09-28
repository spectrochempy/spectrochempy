
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

- Added a structured operation history for ``NDDataset`` with a readable,
  list-compatible, read-only ``history`` view and detached
  ``history_entries``. Update history through ``annotate()``,
  ``replace_history()``, ``clear_history()``, or the compatible ``history``
  assignments. Structured entries cover transposition, selection, out-of-place
  addition and subtraction, and ``mean()`` when it returns an ``NDDataset``;
  other operations may continue to record normal text-only entries
  (:pr:`1676`–:pr:`1681`).
- Added a bounded supervised cross-validation API for PLS and PLS-ending
  Pipelines. ``cross_validate`` fits preprocessing inside each fold and returns
  an aligned ``CrossValidationResult`` with out-of-fold predictions,
  per-target metrics, fold records, and optional fitted estimators. Documented
  ``KFold``, ``GroupKFold``, and ``LeaveOneOut`` adapters are available from the
  SpectroChemPy namespace, together with a user guide and Gallery example
  (:pr:`1658`, :pr:`1659`, :pr:`1660`).

.. section

Bug Fixes
~~~~~~~~~
.. Add here new bug fixes (do not delete this comment)

- ``PLSRegression.predict`` now verifies that masked feature positions match
  those used at fit time, preventing coefficients from being applied silently
  to different variables. Cross-validation also preserves the original feature
  geometry for preprocessors such as MSC (:pr:`1657`).
- ``NDDataset.acquisition_date`` is preserved by copies, out-of-place
  arithmetic, and native ``.scp`` round trips, including timezone offsets
  (:pr:`1688`).
- SPC sub-spectra with distinct x axes retain their associated x and
  sub-spectrum coordinates, titles, and units (:pr:`1691`). CSV round trips
  preserve each recognized coordinate and data title and unit independently
  when only one column carries units (:pr:`1692`). JCAMP-DX LINK exports retain
  text and datetime labels stored in label column zero (:pr:`1693`).
- ``ifft(size=...)`` honors larger and smaller output sizes, including on a
  non-final dimension, and reconstructs physical time coordinates from the
  frequency-bin spacing (:pr:`1694`).
- ``detrend()`` rejects unsupported keyword arguments instead of silently
  ignoring them; its supported ``order`` and ``breakpoints`` parameters are
  unchanged (:pr:`1695`).
- The NMR plugin readers for JEOL, TecMag, SIMPSON, Agilent, and TopSpin honor
  explicit ``origin`` and ``description`` overrides while retaining their
  defaults when either option is omitted or ``None`` (:pr:`1696`).
- ``concatenate()`` rejects partial coordinate information and shared
  coordinate references whose geometry cannot survive the concatenated size,
  instead of returning an inconsistent dataset (:pr:`1697`).
- ``trapezoid()`` and ``simpson()`` no longer integrate masked points as valid
  data. Any incomplete output slice is represented by a masked raw ``NaN``;
  complete neighboring slices remain available (:pr:`1698`).
- ``trapezoid()`` and ``simpson()`` consistently consume ``dim``, ``dims``,
  and ``axis`` selectors, including integer zero. The removed SciPy ``even``
  option is refused explicitly, and SpectroChemPy now requires
  ``scipy>=1.14.1`` (:pr:`1699`).
- SPC interferograms are detected from their original format code without
  inventing a laser frequency. Their raw peak-position header is retained,
  undated files remain readable on Windows, and FFT refuses uncalibrated,
  unitless, dimensionless, or unrelated axes before mutation while accepting
  explicitly calibrated time and optical-path-difference axes (:pr:`1700`).
- Multidimensional interferogram FFTs apply the existing Mertz correction
  independently to every trace, using each trace's own zero path difference;
  the single-interferogram convention is unchanged (:pr:`1701`).
- ``align()`` preserves its existing first-dataset interpolation grid when
  ``interpolate_sampling`` is omitted or ``"auto"`` and explicitly refuses
  numeric or other unsupported target-sampling requests before mutation.
  Call ``NDDataset.interpolate()`` with an explicit coordinate for a different
  target grid (:pr:`1702`).
- ``hamming()`` and ``hann()`` now honor their documented dimension, axis,
  in-place, returned-window, reverse, and inverse options when delegating to
  ``general_hamming()``. ``pk_exp()`` likewise forwards dimension, axis, and
  in-place options to ``pk()`` instead of silently applying the correction on
  the default axis to a copy (:pr:`1682`).
- ``pk()`` and ``pk_exp()`` now reject the unsupported ``inv=True`` option with
  ``NotImplementedError`` before modifying the dataset. Their default and
  explicit ``inv=False`` behavior is unchanged (:pr:`1682`).
- Discrete shifts now move masks with their corresponding values. Circular
  shifts therefore preserve masked statistics, while the zeros introduced by
  left and right shifts are valid, unmasked points. A zero-point left shift no
  longer clears the dataset, a zero-point circular shift with ``neg=True`` no
  longer negates it, and ``cs()`` once again delegates successfully to
  ``roll()`` with a single history entry (:pr:`1683`).
- Harmonized newly generated history messages in core readers, common spectral
  treatments, and analysis results. Individual-file imports identify the
  format and portable filename while the complete source remains in
  ``dataset.filename``; OPUS messages no longer add their own timestamp.
  Histories restored from existing files and opaque vendor histories remain
  unchanged (:pr:`1681`).
- OMNIC SRS imports now retain both useful vendor processing history and the
  import message. Empty vendor blocks are omitted, and the import message no
  longer embeds a second timestamp (:pr:`1681`).
- Corresponding releases of the NMR, PerkinElmer, IRIS, and Tensor plugins
  harmonize their import or analysis-result messages. These plugin-side changes
  require respectively ``spectrochempy-nmr>=0.1.13``,
  ``spectrochempy-perkinelmer>=0.1.6``, ``spectrochempy-iris>=0.1.10``,
  and ``spectrochempy-tensor>=0.1.7``; they are not contained in the core
  package (:pr:`1681`).
- Refused NDDataset in-place arithmetic operations now roll back data, units,
  masks, titles, history, and other trait replacements made by the operation.
  Side effects performed by custom Traitlets observers remain the observer's
  responsibility (:pr:`1668`).
- NDDataset arithmetic now reconstructs dimensions and coordinates after
  positional broadcasting. When a singleton axis expands, the result uses the
  name and coordinate of the operand providing the non-singleton axis.
  Duplicate result dimension names are rejected explicitly (:pr:`1667`).
- Dataset arithmetic now rejects different last-dimension coordinate grids
  carrying the same unit instead of silently accepting a scientifically
  incompatible pairing (:pr:`1665`).


.. section

Dependency Updates
~~~~~~~~~~~~~~~~~~
.. Add here new dependency updates (do not delete this comment)

- The next NMR 0.1.13, PerkinElmer 0.1.6, IRIS 0.1.10, and Tensor 0.1.7
  plugin releases require SpectroChemPy 1.1.0 or later and remain restricted
  to versions below 2. Existing compatible plugin releases remain available
  for installations pinned to SpectroChemPy 1.0.0.


.. section

Breaking Changes
~~~~~~~~~~~~~~~~
.. Add here new breaking changes (do not delete this comment)

- NDDataset arithmetic now rejects two ambiguous pairings that 1.0.0 could
  accept: a broadcast result with duplicate dimension names, and different
  same-unit coordinate grids on a non-expanded final axis. Rename colliding
  dimensions or align the coordinate grids explicitly before the operation
  (:pr:`1665`, :pr:`1667`).
- Conda development builds are no longer published to the ``dev`` label for
  either the core or official plugins. Stable releases remain available from
  the main ``spectrocat`` channel; unreleased versions should be installed from
  a source checkout.
- Direct mutations of the readable ``NDDataset.history`` view now raise
  ``TypeError`` instead of being silently lost. Use ``annotate()``,
  ``replace_history()``, ``clear_history()``, or the supported ``history``
  assignments; ``list(dataset.history)`` remains an ordinary mutable copy
  (:pr:`1681`).
- Assigning a list to ``NDDataset.history`` now retains every supplied entry
  instead of only the first (:pr:`1676`).
- Native ``.scp``/``.pscp`` files containing structured histories use format
  version 3, and portable xarray/NetCDF mappings use version 2. New readers
  continue to accept native version 2 and portable version 1 textual histories;
  reading the new formats with older SpectroChemPy versions is not guaranteed
  (:pr:`1676`, :pr:`1680`).

.. section

Deprecations
~~~~~~~~~~~~
.. Add here new deprecations (do not delete this comment)


.. section

Developer
~~~~~~~~~
.. Add here developer changes (do not delete this comment)

- MAINT: Added the cross-validation building blocks used by the public API:
  helpers for resolving observation dimensions, validating and slicing aligned
  folds, and restoring prediction geometry (:pr:`1653`), plus unfitted cloning
  of ``Pipeline`` templates and their supported steps (:pr:`1654`), and
  per-target regression metric kernels with explicit validity reporting (:pr:`1655`).
- MAINT: Added an internal structured cross-validation result prototype that
  validates complete out-of-fold coverage and records aligned predictions,
  residuals, metrics, fold positions, configuration snapshots, and optional
  fitted fold estimators (:pr:`1656`), followed by a private supervised execution
  engine with fold-local cloning, fitting, prediction, and OOF assembly
  (:pr:`1657`).
- Improved the public units and masks documentation and simplified examples to
  favor SpectroChemPy-native construction, plotting, and arithmetic where that
  keeps the scientific intent clear (:pr:`1661`, :pr:`1662`, :pr:`1663`).
- Corrected development-package version selection so stable core tags sort
  after their older release candidates. Python and Conda builds now derive
  ``1.0.1.devN`` from ``spectrochempy-v1.0.0`` rather than falling back to an
  obsolete RC series (:pr:`1666`).
