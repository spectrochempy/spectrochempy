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

- Added a structured operation history for ``NDDataset``. ``history`` is a
  readable, list-compatible, read-only view; ``history_entries`` returns a
  detached structured copy; and ``annotate()``, ``replace_history()``, and
  ``clear_history()`` provide explicit editing operations. Structured entries
  cover selection and transposition, supported arithmetic and reductions,
  shape operations, shifts and zero filling, apodization, phasing,
  ``mc()``, ``ps()``, ``ht()``, ``dc()``, and numerical integration. Entries
  identify the executed kernel and, where meaningful, requested and effective
  parameters plus the requested and resolved dimension and source axis.
  Existing chronology is preserved by copies, supported persistence,
  ``trapezoid()``, ``simpson()``, and ``snv()``; structured and text-only
  entries coexist normally, with ``snv()`` retaining the single text entry
  ``SNVTransformer applied``. This journal is not an exhaustive provenance or
  replay system (:pr:`1676`, :pr:`1677`, :pr:`1678`, :pr:`1679`, :pr:`1680`,
  :pr:`1681`, :pr:`1703`, :pr:`1704`, :pr:`1705`, :pr:`1706`, :pr:`1707`,
  :pr:`1708`, :pr:`1709`, :pr:`1710`, :pr:`1711`, :pr:`1712`, :pr:`1714`).
- Added bounded supervised cross-validation for PLS and PLS-ending Pipelines.
  ``cross_validate`` fits preprocessing inside each fold and returns an aligned
  ``CrossValidationResult`` with out-of-fold predictions, per-target metrics,
  fold records, and optional fitted estimators. ``KFold``, ``GroupKFold``, and
  ``LeaveOneOut`` adapters are available from the SpectroChemPy namespace,
  with a user guide and Gallery example (:pr:`1658`, :pr:`1659`, :pr:`1660`).

.. section

Bug Fixes
~~~~~~~~~
.. Add here new bug fixes (do not delete this comment)

- ``PLSRegression.predict`` now verifies that masked feature positions match
  those used at fit time, and cross-validation preserves the original feature
  geometry for preprocessors such as MSC (:pr:`1657`).
- Dataset arithmetic rejects incompatible same-unit coordinate grids,
  reconstructs dimensions and coordinates correctly after positional
  broadcasting, refuses duplicate result dimension names, and rolls back data,
  units, masks, titles, history, and other trait replacements made by a refused
  in-place operation. Side effects performed by custom Traitlets observers
  remain the observer's responsibility (:pr:`1665`, :pr:`1667`, :pr:`1668`).
- Spectral-processing wrappers now forward their documented selectors and
  execution options consistently. ``hamming()``, ``hann()``, and ``pk_exp()``
  no longer fall back silently to the default dimension or an out-of-place
  result; unsupported inverse phasing is refused; discrete shifts move masks
  with values and handle zero shifts correctly; and refused zero filling or a
  later kernel failure restores any temporary in-place permutation
  (:pr:`1682`, :pr:`1683`, :pr:`1705`, :pr:`1709`).
- ``ifft(size=...)`` now honors larger and smaller sizes on final and non-final
  dimensions and reconstructs physical time coordinates from the retained
  frequency-bin spacing (:pr:`1694`).
- SPC interferograms are detected from their format code without inventing a
  laser frequency. Their raw peak position is retained, undated files remain
  readable on Windows, and FFT validates calibration before mutation while
  accepting explicitly calibrated time and optical-path-difference axes.
  Multidimensional FFT applies the existing Mertz correction independently to
  each trace and its own zero path difference (:pr:`1700`, :pr:`1701`).
- ``mc()``, ``ps()``, and ``ht()`` honor ``dim``, ``dims``, and ``axis``.
  ``ht()`` uses the selected dimension length when ``N`` is omitted or
  ``None``, supports equal or larger transform sizes on multidimensional data,
  preserves the input shape and real component, and rejects invalid or smaller
  sizes before mutation (:pr:`1709`, :pr:`1710`).
- ``trapezoid()`` and ``simpson()`` exclude masked values from quadrature and
  publish incomplete slices as masked raw ``NaN`` values. They consume all
  supported dimension selectors consistently, including integer zero, and
  explicitly refuse the removed SciPy ``even`` option (:pr:`1698`,
  :pr:`1699`).
- ``concatenate()`` refuses partial coordinate information and shared
  coordinate references whose geometry cannot survive the concatenated size,
  rather than returning an inconsistent dataset (:pr:`1697`).
- ``detrend()`` rejects unsupported keyword arguments, and ``align()`` refuses
  unsupported target-sampling requests instead of silently ignoring them.
  Omitting ``interpolate_sampling`` or using ``"auto"`` preserves alignment on
  the first dataset grid; use ``NDDataset.interpolate()`` for an explicit
  alternative grid (:pr:`1695`, :pr:`1702`).
- ``NDDataset.acquisition_date`` is preserved by copies, out-of-place
  arithmetic, and native ``.scp`` round trips, including timezone offsets
  (:pr:`1688`).
- Reader and exporter metadata are no longer lost in several partial or
  multi-spectrum cases: SPC sub-spectra retain their associated coordinates,
  CSV round trips preserve each recognized title and unit independently,
  JCAMP-DX LINK exports retain labels in column zero, and NMR plugin readers
  honor explicit ``origin`` and ``description`` overrides (:pr:`1691`,
  :pr:`1692`, :pr:`1693`, :pr:`1696`).
- Newly generated histories from core readers, spectral treatments, analysis
  results, and OMNIC SRS imports now use consistent messages without duplicate
  timestamps while preserving useful vendor history. Corresponding official
  plugin releases apply the same message conventions (:pr:`1681`).

.. section

Dependency Updates
~~~~~~~~~~~~~~~~~~
.. Add here new dependency updates (do not delete this comment)

- SpectroChemPy continues to require Python 3.11 or later and now requires
  ``scipy>=1.14.1`` for the supported Simpson integration behavior
  (:pr:`1699`).
- The next NMR 0.1.13, PerkinElmer 0.1.6, IRIS 0.1.10, and Tensor 0.1.7 plugin
  releases require SpectroChemPy 1.1.0 or later and remain restricted to
  versions below 2. Existing compatible plugin releases remain available for
  installations pinned to SpectroChemPy 1.0.0 (:pr:`1684`).

.. section

Breaking Changes
~~~~~~~~~~~~~~~~
.. Add here new breaking changes (do not delete this comment)

- Direct mutations of the readable ``NDDataset.history`` view now raise
  ``TypeError`` instead of being silently lost. Use ``annotate()``,
  ``replace_history()``, ``clear_history()``, or supported ``history``
  assignment; assigning a list now retains every supplied entry
  (:pr:`1676`, :pr:`1681`).
- Native ``.scp``/``.pscp`` files containing structured histories use format
  version 3, and portable xarray/NetCDF mappings use version 2. The new readers
  accept native version 2 and portable version 1 textual histories, but older
  SpectroChemPy versions are not guaranteed to read the new formats
  (:pr:`1676`, :pr:`1680`).
- Arithmetic now refuses ambiguous broadcast results with duplicate dimension
  names and different same-unit coordinate grids on a non-expanded final axis.
  Rename colliding dimensions or align coordinate grids explicitly
  (:pr:`1665`, :pr:`1667`).
- Requests that previously appeared to succeed while being unsupported now
  fail explicitly: ``pk(..., inv=True)``, unknown ``detrend()`` options, and
  non-default ``align(interpolate_sampling=...)`` targets. Use forward
  phasing, supported detrending options, or ``NDDataset.interpolate()`` with an
  explicit grid respectively (:pr:`1682`, :pr:`1695`, :pr:`1702`).
- ``simpson()`` no longer accepts ``even`` because SciPy removed that option.
  These strategies were functional with earlier SciPy versions: omitting
  ``even`` selects the current SciPy behavior and can change results for an
  even number of samples compared with the former ``"avg"``, ``"first"``,
  or ``"last"`` strategies. ``trapezoid()`` never accepted ``even``
  (:pr:`1699`).
- FFT of an interferogram now requires an explicitly calibrated time or
  optical-path-difference coordinate; raw sample indices and unrelated units
  are refused before mutation (:pr:`1700`).
- Conda development builds are no longer published to the ``dev`` label for
  either core or official plugins. Stable releases remain on the main
  ``spectrocat`` channel; install unreleased versions from a source checkout
  (:pr:`1685`).

.. section

Deprecations
~~~~~~~~~~~~
.. Add here new deprecations (do not delete this comment)


.. section

Developer
~~~~~~~~~
.. Add here developer changes (do not delete this comment)

- Added the cross-validation building blocks used by the public API: aligned
  fold validation and slicing, unfitted Pipeline cloning, per-target metrics,
  structured result assembly, and the supervised execution engine
  (:pr:`1653`, :pr:`1654`, :pr:`1655`, :pr:`1656`, :pr:`1657`).
- Improved the public units and masks documentation and simplified examples to
  favor SpectroChemPy-native construction, plotting, and arithmetic where that
  preserves the scientific intent (:pr:`1661`, :pr:`1662`, :pr:`1663`).
- Corrected development-package version selection so stable core tags sort
  after older release candidates, and strengthened documentation builds and
  selector examples used by the release documentation (:pr:`1666`, :pr:`1686`,
  :pr:`1687`, :pr:`1689`, :pr:`1690`).
