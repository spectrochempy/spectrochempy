.. _omnic-reader-mapping:

SpectroChemPy OMNIC reader mapping
==================================

The OMNIC format references (:ref:`spa-format`, :ref:`spg-format`,
:ref:`srs-format`) describe the file formats independently of any software.
This page records how the SpectroChemPy reader
(``src/spectrochempy/core/readers/read_omnic.py``) maps the documented
structures onto ``NDDataset`` objects. It is implementation context for
SpectroChemPy contributors: the reader does not define the binary semantics,
and its behavior may lag behind the format references. The reader, its tests,
and the format references are reconciled through separate reader audits and
implementation changes.

SPA files
---------

* The reader's internal ``firstx`` and ``lastx`` names describe storage
  order (payload element 0 and the final element), not OMNIC's *First X* /
  *Last X* display terminology: ``firstx`` is header ``+16`` (OMNIC
  *Last X*) and ``lastx`` is header ``+20`` (OMNIC *First X*).
* The interferogram optical-path-difference step
  ``sample_spacing / (2 * reference_frequency)`` (header ``+84`` and
  ``+80``) is implemented for native interferograms.
* The offset-296 timestamp is exposed as
  ``dataset.acquisition_date``. Library/retrieved files can store a counter
  or default-like value there, so the field should not be promoted to an
  acquisition date without checking the variant.
* Subtype-``0x79`` Experiment Information is decoded with the fixed-slot
  anchors ``+10``, ``+90``, ``+154`` and ``+413`` and exposed as ``omnic_*``
  metadata; the ``+670`` text and the other ``0x82`` subtypes are not read.
  Only the first decoded block of a file is used.
* Some variant-specific metadata, including the Raman distinction between
  header ``+80`` (reference frequency) and ``+96`` (excitation frequency),
  may not yet be surfaced with native semantics. This metadata limitation
  does not affect the stored Raman X axis.
* The optical velocity is taken from the ``0x6a`` block when it is parsed,
  with header-relative ``+188`` as a fallback (see
  :ref:`omnic-header-overreads`).

SPG files
---------

The reader:

* parses the key table, the per-spectrum ``0x02`` data headers, ``0x03``
  payloads, ``0x6b`` title/timestamp blocks, and ``0x82`` blocks (using the
  first decoded subtype-``0x79`` block of the file);
* sets ``dataset.name`` from the **filesystem filename stem**, while the
  embedded group/title text (offset 30) is stored separately in the dataset
  description;
* exposes the per-spectrum ``0x6b`` timestamps as the y coordinate
  (Unix-seconds values with datetime and title label rows) and as
  ``dataset.acquisition_date = minimum member timestamp`` — a reader
  convention on a heterogeneous container, not a format-level group
  acquisition property;
* sorts the spectra chronologically by those timestamps by default
  (``sortbydate=True``), which changes the original SPG group order;
  ``sortbydate=False`` preserves the stored order;
* reduces the per-spectrum acquisition parameters (collection length,
  reference/laser frequency, optical velocity) to a single global metadata
  value taken from the last parsed spectrum's header; the optical velocity
  is read at header-relative ``+188`` (and an internal, unexposed text
  value at ``+208``), i.e. beyond the ``0x02`` block (see
  :ref:`SPG header offsets <spg-header-overreads>`);
* does not currently expose the ``0x1b`` history blocks (including their
  background references), the ``0x6a`` parameter blocks, the ``0x69``
  blocks, the interferogram-related ``0x64``–``0x67`` structures, or
  per-spectrum Experiment Information;
* rejects spectra with inconsistent X-axis definitions unless
  ``allow_inconsistent_x=True`` is used, in which case it returns one
  dataset per spectrum.

SRS files
---------

:func:`spectrochempy.read_srs` is the public entry point for ``.srs``
files; it delegates to the OMNIC importer with the ``.srs`` file-type
constraint.

* Spectral series are presented with a **descending wavenumber** X axis
  (high to low), matching the convention used by ``read_spa``, whereas the
  raw storage order is ascending (see :ref:`srs-spectral-sample-order`).
  The normalization is applied per spectrum from each record's own
  endpoints.
* Spectral backgrounds are normalized per record to the same descending
  wavenumber grid as the series. Only a subset of background layouts is
  currently decoded: the ``return_bg`` path returns no data for some
  background record shapes.
* Rapid-scan interferograms retain the raw **ascending data-points**
  coordinate. These records are flagged internally (``meta.interferogram``);
  their ZPD index is the data-derived peak index, and the laser frequency is
  set to the reader's standard default rather than derived from these
  records' coordinate or data.
* The reader distinguishes three X-axis cases: a known spectral axis
  (X-unit codes 1/3/4/32), an explicit data-points interferogram (X-unit
  code 2), and an unknown X-axis type (unrecognized code), which is left in
  raw storage orientation with an informational message and is neither
  treated as an interferogram nor spectral-normalized.
* The series name (series-header ``+938``) is split on the first newline.
* The Y coordinate is **relative series time in minutes** (``time_min``
  to the last time, rounded to 3 decimal places; title "Time", units
  "minute"). Earlier reader versions misread header ``+1002`` as a
  collection length and header ``+1010`` (then named ``firsty``) as the
  series minimum.
* The native ``Collected`` timestamp (file offset 296) is accepted only when
  it matches one of its header-relative copies at the documented locations
  (see :ref:`srs-time-representation`). It is then exposed through
  ``dataset.acquisition_date``, and the Y coordinate gains a second label
  column with the per-spectrum absolute datetimes
  ``Collected + timedelta(minutes=time_min + i * step)``, computed in full
  precision from the native float32 fields rather than from the rounded
  public Y coordinate.
* Variants without a valid absolute anchor (GC, reprocessed RapidScan), or
  whose copy locations differ from the documented ones, leave
  ``acquisition_date`` unset and keep single-column names-only Y labels.
* The public ``collection_length`` metadata attribute is derived from
  series-header ``+1006`` (series last time × 60), not from the
  general-header field at ``+68``.
* The historical ``reverse_x`` keyword is a deprecated no-op: spectral
  orientation is handled automatically and supplying the keyword emits a
  ``DeprecationWarning``.
