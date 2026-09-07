.. _srs-format:

OMNIC SRS file format
=====================

.. note::

   This is an independently reverse-engineered, unofficial description of the
   OMNIC SRS format. See :ref:`omnic-file-formats` for provenance,
   limitations, and certainty-level definitions. Contributors with access to
   other OMNIC ``.srs`` variants are encouraged to report differences so that
   this reference can be improved.

.. _srs-tested-files:

Tested files and evidence
-------------------------

The observations below are based on the public SpectroChemPy test fixtures and
on one independently controlled series (see the note at the end of this
section).

.. list-table:: Tested ``.srs`` files
   :header-rows: 1

   * - File
     - Acquisition family
     - Content
     - Format structures exercised
   * - ``rapid_scan.srs``
     - RapidScan
     - interferograms (643 × 4160)
     - repeated-record layout; interferogram (data-points) axis; inter-spectrum
       trailer
   * - ``rapid_scan_reprocessed.srs``
     - RapidScan (reprocessed)
     - spectral (643 × 3734)
     - repeated-record layout; spectral axis; pristine/reprocessed flag
   * - ``high_speed.srs``
     - HighSpeed
     - spectral (897 × 13898)
     - repeated-record layout; spectral axis; 4-occurrence detection signature
   * - ``GC_Demo.srs``
     - TG/GC
     - spectral (788 × 1738)
     - repeated-record layout; background record; spectral axis
   * - ``TGA_demo.srs``
     - TGA
     - spectral (485 × 3630)
     - repeated-record layout; background record; spectral axis
   * - ``TGAIR-unreadable.srs``
     - TG/GC
     - spectral (335 × 1868)
     - repeated-record layout; spectral axis

``[ESTABLISHED]`` The repeated-record layout ``84 + nx*4 + 16`` bytes
reproduces the data arrays exactly in independent binary reconstruction for
all of the files above and all four acquisition families, covering both
spectra and rapid-scan interferograms (see :ref:`Repeated spectrum records
<srs-repeated-records>`).

In addition, **one independently controlled TG/GC series** was used as a
physical validation oracle. Individual spectra of that series were exported by
OMNIC as SPA files and compared against the raw SRS samples: the match is
``correlation = 1.000000`` and ``RMSE = 0.000000`` in the correct physical
order (see :ref:`srs-spectral-sample-order`). The same series also exercised
the trailer's SeriesProfile, Gram-Schmidt, and Area structures. The source
files of that series are **not** distributed with SpectroChemPy; only the
public fixtures listed above are.

Overall file organization
-------------------------

.. code-block:: text

    OMNIC SRS layout (regions currently understood)
    ===================================================

    +------------------+   file header / key-table region:
    | file header /    |   file magic, file-level flags, key table
    | key-table region |   (first 304 bytes + entries)
    +------------------+                  ^
    | series metadata /|------------------+-- series-header base at
    | header region    |                     key[0].ref_pos (152 bytes before
    +------------------+                     the first detection signature)
    | background header |   its own X endpoints
    | / background data |
    +------------------+
    | repeated spectrum records:
    |   [84-byte prefix | nx * 4-byte float32 payload | 16-byte trailer]
    |   ... x ny
    +------------------+
    | post-series profile region (observed in the controlled series):
    |                  Gram-Schmidt-related data
    |                  copied series metadata
    |                  SeriesProfile block(s), one per profile
    +------------------+

The diagram is logical, not to scale: it shows only regions and boundaries
whose relationships are reproducibly observed. The metadata associated with
the series extend well beyond the 152 bytes above the first signature, and
the internal subdivision of that region — like that of the trailer blocks —
is only partially understood. Some boundary positions and the
number/layout of the trailer blocks are version-dependent
(see :ref:`SeriesProfile structures <srs-seriesprofile>`).

Offset conventions
------------------

* File-header / key-table offsets are file-relative (relative to the start of
  the file).
* Series-header offsets are relative to the located *series-header base*
  (at ``key[0].ref_pos``, i.e. the position of the first detection signature
  minus 152; see :ref:`srs-detection-signatures`).
* Repeated-record offsets use the record start as origin; trailer offsets use
  the trailer start; SeriesProfile fields use the block start as origin.

Header and key structures
-------------------------

File header / key-table region (file-relative)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Offset
     - Size
     - Type
     - Meaning
     - Certainty
   * - 0
     - 18
     - ASCII
     - File magic ``"Spectral Exte File"`` (followed by ``\r\n``). The SPA/SPG
       sibling formats use ``"Spectral Data File"`` instead.
     - ``[ESTABLISHED]``
   * - 18
     - 274
     - bytes
     - Reserved / file metadata region.
     - ``[UNKNOWN]``
   * - 292
     - 1
     - UInt8
     - Level-of-processing flag: ``0x27`` pristine, ``0x0f`` reprocessed
       (verified for the RapidScan family; other families show both values
       too).
     - ``[OBSERVED]``
   * - 294
     - 2
     - UInt16
     - Number of key-table entries (28–29 observed).
     - ``[OBSERVED]``
   * - 304
     - n × 16
     - KeyTable[]
     - Key table (see below).
     - ``[OBSERVED]`` structure

Key-table entry (16 bytes; entry-relative offsets)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Offset
     - Size
     - Type
     - Meaning
     - Certainty
   * - 0
     - 2
     - UInt16
     - Key type / role indicator (``0x02`` identifies the series entry).
     - ``[OBSERVED]``
   * - 4
     - 4
     - UInt32
     - Often 0 in the observed files; role unclear.
     - ``[UNKNOWN]``
   * - 8
     - 4
     - UInt32
     - ``ref_pos``: file offset of the referenced section. For the first
       entry, ``ref_pos`` equals the series-header base, which is exactly 152
       bytes before the first detection signature (verified across six files).
     - ``[ESTABLISHED]``
   * - 12
     - 4
     - UInt32
     - Referenced section size in bytes.
     - ``[HYPOTHESIS]``

Series metadata
---------------

In the files examined, the series-header base is located 152 bytes before the
first detection signature (see :ref:`srs-detection-signatures`). Offsets in the
tables below are relative to that base. The metadata associated with the series
extend well beyond the first 152 bytes: the exact subdivision into OMNIC
internal blocks is only partially understood. Only the fields listed below
have been interpreted so far; the rest of the region is unmapped.

.. list-table:: Series-header fields (offsets relative to the series-header base)
   :header-rows: 1

   * - Offset
     - Size
     - Type
     - Meaning
     - Certainty
   * - 0
     - 1
     - UInt8
     - Header type marker (``0x01``/``0x02``/``0x03`` …).
     - ``[OBSERVED]``
   * - 4
     - 4
     - UInt32
     - ``nx`` — points per spectrum.
     - ``[ESTABLISHED]``
   * - 8
     - 1
     - UInt8
     - X-unit code: 1 = cm⁻¹, 2 = data points (interferogram), 3 = nm,
       4 = µm, 32 = Raman shift (cm⁻¹).
     - ``[ESTABLISHED]`` for codes 1 and 2; ``[OBSERVED]`` for the others
   * - 12
     - 1
     - UInt8
     - Y/data-unit code: 0x11 absorbance, 0x10 transmittance (%),
       0x0F single beam, 0x16 detector signal (V), 0x1A photoacoustic,
       0x1F Raman intensity, among others shared with the SPA/SPG readers.
     - ``[OBSERVED]``
   * - 16
     - 4
     - Float32
     - Raw ``firstx`` (stored ordering, not normalized).
     - ``[ESTABLISHED]``
   * - 20
     - 4
     - Float32
     - Raw ``lastx`` (stored ordering, not normalized).
     - ``[ESTABLISHED]``
   * - 24
     - 4
     - Float32
     - Per-file varying, registration-like factor; see
       :ref:`srs-unknown-fields`.
     - ``[HYPOTHESIS]`` / ``[UNKNOWN]``
   * - 28
     - 4
     - UInt32
     - Scan point count.
     - ``[OBSERVED]``
   * - 32
     - 4
     - UInt32
     - ZPD (zero-path-difference) position.
     - ``[OBSERVED]``
   * - 36
     - 4
     - UInt32
     - Number of scans.
     - ``[OBSERVED]``
   * - 52
     - 4
     - UInt32
     - Number of background scans.
     - ``[OBSERVED]``
   * - 56
     - 4
     - Float32
     - 1.0 in non-RapidScan, 0.0 in RapidScan files examined; see
       :ref:`srs-unknown-fields`.
     - ``[OBSERVED]`` (correlated only)
   * - 68
     - 4
     - UInt32
     - General-header *collection length* in 1/100 s. This is the shared
       acquisition-time field of the OMNIC header family; it must **not** be
       conflated with the series minimum/first time at +1002.
     - ``[OBSERVED]``
   * - 80
     - 4
     - Float32
     - Reference (laser) frequency.
     - ``[OBSERVED]``
   * - 84
     - 4
     - Float32
     - 1.0 in TG/GC and HighSpeed, 2.0 in RapidScan files examined; see
       :ref:`srs-unknown-fields`.
     - ``[OBSERVED]`` (correlated only)
   * - 184
     - 4
     - Float32
     - Same value pattern as +84.
     - ``[OBSERVED]`` (correlated only)
   * - 188
     - 4
     - Float32
     - Optical velocity.
     - ``[OBSERVED]``
   * - 208
     - var
     - text
     - Spectrum history text; a record whose text starts with ``Background``
       marks a background header.
     - ``[ESTABLISHED]`` string presence; ``[OBSERVED]`` role
   * - 296
     - 8
     - UInt64
     - Raw file size.
     - ``[HYPOTHESIS]``
   * - 938
     - ≤ 256
     - text
     - Series name (the reader splits on the first newline).
     - ``[OBSERVED]``
   * - 1002
     - 4
     - Float32
     - **Series minimum / first time, in minutes** — the time-axis anchor.
       This field was historically misinterpreted as a "collection length";
       the general-header collection-length field is the one at +68. Some
       implementations expose it as a collection duration in seconds
       (value × 60) for backward compatibility.
     - ``[ESTABLISHED]``
   * - 1006
     - 4
     - Float32
     - Series maximum / last time, in minutes.
     - ``[OBSERVED]``
   * - 1010
     - 4
     - Float32
     - **Regular time step, in minutes** (historically misnamed ``firsty``;
       it is *not* the series minimum). See :ref:`srs-time-representation`.
     - ``[ESTABLISHED]``
   * - 1026
     - 4
     - UInt32
     - ``ny`` — number of spectra in the series.
     - ``[ESTABLISHED]``
   * - 1030
     - 1
     - UInt8
     - Y-unit code; a value of 1 may mean minutes.
     - ``[HYPOTHESIS]``
   * - 1044
     - 2
     - UInt16
     - Gram-Schmidt offset (10 in the tested family); see
       :ref:`srs-gram-schmidt`.
     - ``[OBSERVED]`` value / ``[UNKNOWN]`` semantic
   * - 1046
     - 2
     - UInt16
     - Gram-Schmidt interferogram points (100 in the tested family); see
       :ref:`srs-gram-schmidt`.
     - ``[OBSERVED]`` / ``[HYPOTHESIS]``
   * - 1048
     - 2
     - UInt16
     - 200 in the tested files; see :ref:`srs-unknown-fields`.
     - ``[UNKNOWN]``
   * - 1200
     - var
     - text
     - Initial history text (pristine files). Reprocessed files carry their
       updated history at the end of the file after a 16-byte ``0xff``
       sequence.
     - ``[OBSERVED]``

Background data
---------------

``[OBSERVED]`` Spectral backgrounds have their own header, located through the
second detection signature (signature position − 152), with their own X
endpoints. In the tested background records where both headers were decoded,
the **endpoint ordering can differ from that of the series header**: spectral
background headers store raw ``firstx < lastx`` (ascending) while the
corresponding spectral series headers store raw ``firstx > lastx``
(descending). No single raw-order rule has been observed to apply to the
whole file; any normalization must be applied per record.

The present reader normalizes spectral backgrounds per record to the same
public descending-wavenumber grid as the series. Only a subset of background
layouts is currently decoded (the reader's ``return_bg`` path returns no data
for some background record shapes).

.. _srs-repeated-records:

Repeated spectrum records
-------------------------

``[ESTABLISHED]`` In the tested SRS series, each spectrum record has the
following layout. The same repeated-record structure is also observed for the
rapid-scan interferograms in ``rapid_scan.srs``, so the description below is
generic to both spectral series and interferogram records:

.. code-block:: text

    data record
    ├── 84-byte prefix
    ├── nx × 4-byte float32 intensity payload
    └── 16-byte trailer

* ``[ESTABLISHED]`` The **spectrum name is null-terminated inside the
  84-byte prefix**; the prefix also carries binary metadata (per-file mostly
  constant fields and a per-spectrum minimum-Y value).
* ``[ESTABLISHED]`` The payload boundaries reproduce the data arrays exactly
  in independent binary reconstruction (stride
  `84 + nx·4 + 16` bytes per record).
* ``[ESTABLISHED]`` The trailer size (16 bytes) holds for all tested files.

.. list-table:: 84-byte record prefix (record-relative offsets)
   :header-rows: 1

   * - Offset
     - Size
     - Type
     - Meaning
     - Certainty
   * - 0 .. null
     - var
     - ASCII
     - Spectrum name, null-terminated (e.g. ``Linked spectrum at 0.025 min.``).
     - ``[ESTABLISHED]``
   * - 22 .. 75
     - var
     - bytes
     - Mostly per-file constant metadata; semantics not fully interpreted.
     - ``[OBSERVED]`` values / ``[UNKNOWN]`` semantics
   * - 76
     - 4
     - Float32
     - Spectrum minimum Y value.
     - ``[OBSERVED]``

.. _srs-inter-spectrum-trailer:

Inter-spectrum trailer (16 bytes; trailer-relative offsets)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``[ESTABLISHED]`` Tested records (both spectral series and rapid-scan
interferograms) are followed by a 16-byte trailer.

``[OBSERVED]`` One uint32 field in this trailer behaves as a **cumulative
time counter in centiseconds** (values increase with spectrum index and track
the series time axis).

``[UNKNOWN]`` The remaining trailer fields are not fully interpreted.

.. list-table::
   :header-rows: 1

   * - Offset
     - Size
     - Type
     - Meaning
     - Certainty
   * - 0
     - 4
     - UInt32
     - ``0x20`` (32) in the tested files.
     - ``[OBSERVED]``
   * - 4
     - 4
     - UInt32
     - Cumulative elapsed-time counter in centiseconds for the **next**
       spectrum (see :ref:`srs-time-representation`).
     - ``[OBSERVED]``
   * - 8
     - 8
     - UInt64
     - Zeros in the tested files.
     - ``[OBSERVED]``

.. _srs-time-representation:

Time representation
-------------------

Four quantities describe the series time axis; they must not be conflated:

* **Series minimum / first time** — header +1002 (Float32, minutes). This is
  the time of the first spectrum and the anchor of the axis.
* **Last / max time** — header +1006 (Float32, minutes).
* **Regular time step** — header +1010 (Float32, minutes). In the
  independently controlled series this equals ``(last time − first time) /
  (ny − 1)`` exactly to float32 precision, which confirms it is a step, not a
  minimum (the historical ``firsty`` misreading).
* **Number of spectra** — ``ny`` (header +1026).

``[OBSERVED]`` the time axis follows the regular model

.. code-block:: text

    T[i] = time_min + i * step        (i = 0 .. ny-1)

where ``time_min`` reads from +1002 and ``step`` from +1010.

The trailer's centisecond counter provides an independent confirmation:
``[OBSERVED]`` the counter stored in the trailer of spectrum *i* holds the
elapsed time of spectrum *i+1*, quantized to integer centiseconds; dividing by
6000 (centiseconds per minute) reproduces the time axis and matches the
3-decimal times embedded in the spectrum names. In the RapidScan test file the
per-spectrum increment equals the collection period; in the independently
controlled series the increment is the regular time step above.

.. _srs-spectral-sample-order:

Spectral sample order
---------------------

``[OBSERVED]``

In the SRS files examined, spectral intensity samples are stored in
**ascending-wavenumber physical order** (sample 0 = lowest wavenumber). This
ordering was independently confirmed for one controlled series against
individual spectra exported by OMNIC as SPA files. It should not yet be
treated as a guaranteed invariant for every SRS producer/version.

The independent comparison established:

* correlation = 1.000000
* RMSE = 0.000000

for matching raw SRS samples against the OMNIC-exported SPA data in the
correct physical order (the reversed orientation does not match).

This page describes the **raw storage order**. Presentational conventions of
particular software are separate: SpectroChemPy, for instance, presents SRS
spectral datasets with a *descending* wavenumber axis (matching its SPA
convention) while the raw file order is ascending (see
:ref:`srs-implementation-references`).

Rapid-scan interferograms
-------------------------

``rapid_scan.srs`` contains rapid-scan interferograms. ``[OBSERVED]`` These
interferogram records carry an X-axis of type "data points" (x-unit code 2,
no physical wavenumber unit), stored with ascending data-points coordinates.

Do **not** equate "no xunit code" with "interferogram": a record whose X-unit
code is unrecognized also lacks a physical xunit, yet is not a data-points
interferogram. The current reader therefore distinguishes three cases:

* a known spectral axis (x-unit codes 1/3/4/32);
* an explicit data-points interferogram (x-unit code 2);
* an unknown X-axis type (unrecognized code), which is neither treated as an
  interferogram nor spectral-normalized.

This three-way distinction is presented as the current observed evidence for
the files examined, not as a universal OMNIC invariant.

.. _srs-seriesprofile:

SeriesProfile structures
------------------------

``[HYPOTHESIS]`` A recurring trailer structure interpreted as an OMNIC
**SeriesProfile** block has been identified in the tested files. The block
boundaries and value shapes are reproducible; their full meaning is not.

.. list-table:: SeriesProfile block (block-relative offsets)
   :header-rows: 1

   * - Offset
     - Size
     - Type
     - Meaning
     - Certainty
   * - 0
     - 4
     - UInt32
     - ``ny`` marker equal to the series spectrum count — a reliable block
       boundary marker.
     - ``[ESTABLISHED]`` boundary; ``[OBSERVED]`` value
   * - 4
     - 4
     - UInt32
     - Profile-type code: 6 for "peak area of one peak" (Area), 0 for
       Chemigram, in the observed files.
     - ``[HYPOTHESIS]`` (higher confidence for code 6, medium for code 0)
   * - 8
     - 4
     - UInt32
     - Gram-Schmidt offset, copied unchanged from the series metadata into
       every block examined.
     - ``[OBSERVED]`` value / ``[UNKNOWN]`` semantic
   * - 12
     - 4
     - Float32
     - Series minimum time, copied into every block examined.
     - ``[OBSERVED]``
   * - 16
     - 4
     - Float32
     - Series maximum time, copied into every block examined.
     - ``[OBSERVED]``
   * - 20
     - ~26
     - ASCII
     - Human-readable profile label, e.g. ``Area [lo, hi]`` or
       ``Chemigram: ...``. No numeric copy of the integration limits has been
       identified in the structures examined; in the tested blocks, the limits
       are present in the label.
     - ``[ESTABLISHED]`` string; ``[OBSERVED]`` content
   * - after header
     - var
     - padding
     - Header padding to a fixed block-relative data offset in the tested
       version.
     - ``[OBSERVED]``
   * - data offset
     - ny × 4
     - Float32[]
     - Per-spectrum profile values vector.
     - ``[ESTABLISHED]`` vector shape; semantics ``[HYPOTHESIS]``

``[OBSERVED]`` In the tested files the block boundaries, the per-spectrum
vector shape, and the correspondence between the ASCII label and the vector
are reproducible, and a series defines one block per profile. The exact byte
stride and header/data grouping are **version-dependent**: a fixed stride
observed in one OMNIC-written file was absent in a file re-saved by a
different OMNIC version (headers packed together, data vectors grouped
separately). A parser must therefore discover the profile blocks dynamically
(e.g. via the ``ny`` markers and the ASCII labels) rather than assume a fixed
count or stride.

Gram-Schmidt / Chemigram / Area structures
------------------------------------------

.. _srs-gram-schmidt:

Gram-Schmidt data
~~~~~~~~~~~~~~~~~

``[OBSERVED]`` In the independently controlled series examined, the
post-series trailer/profile region begins with a Gram-Schmidt-related region:
a small header of two uint32 values (``ny`` and the number of Gram-Schmidt
points) followed by a matrix of ``ny × gs_points`` float32 values (5331 × 100
in that series). Whether every SRS family places this region first in the
trailer has not been verified.

``[OBSERVED]`` The number of Gram-Schmidt interferogram points also appears
as a series-header field at +1046 (UInt16 = 100 in the tested family), and
the Gram-Schmidt offset value (header +1044, UInt16 = 10) is copied into every
SeriesProfile block examined. ``[UNKNOWN]`` The exact meaning of the offset and the
reconstruction role of the matrix are not yet established.

Chemigram / Area observations (experimental)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This subsection records reverse-engineering evidence, not a formal binary
definition.

``[OBSERVED]`` Profile vectors whose areas were recomputed from the spectral
data reproduce the stored vectors: one region matched exactly, the others
within a ≤ 0.6 % affine residual (linear correlation ~1.00000).

``[HYPOTHESIS]`` The stored "Area" profile values follow a
baseline-subtracted integral of the intensity over the labelled region
(``Area = ∫(I − baseline) dx`` using the ascending-wavenumber raw order, with
the baseline joining the region-endpoint intensities). The small residual is
consistent with a slight difference in the boundary-point selection at the
integration limits. No numeric copy of the integration limits has been
identified in the structures examined; in the tested blocks, the limits are
present in the ASCII label.

.. _srs-detection-signatures:

Detection signatures and positioning observations
-------------------------------------------------

``[OBSERVED]`` In the files examined, each acquisition family is preceded by a
recognizable 10–16 byte signature. The tested files show:

* **RapidScan** — ``\x02\x00\x00\x00\x18\x00\x00\x00\x00\x00\x48\x43\x00\x50\x43\x47``,
  occurring 3 times;
* **HighSpeed** — ``\x02\x00\x00\x00\x18\x00\x00\x00\x00\x00\x48\x43\x00\xc8\xaf\x47``,
  occurring 4 times;
* **TG/GC** — ``\x02\x00\x00\x00\x18\x00\x00\x00\x00\x00`` (first 10 bytes),
  occurring 3 times; the following bytes vary between files.

Positioning relationships (all values verified in the tested corpus):

* the **first** signature occurs 152 bytes after the series-header base
  (``key[0].ref_pos`` points at the same header — ``[ESTABLISHED]`` for
  ``key[0]`` across the six public files; ``[OBSERVED]`` as a general rule);
* the **second** signature occurs 152 bytes before the background header;
* the **last** signature occurs 60 bytes before the series spectral-data
  start (data = signature position + 60);
* in HighSpeed files, the third occurrence's role is not understood and the
  fourth is the data-position one.

Careful wording is intentional here: these are **observed** positioning
relationships, not guaranteed layout invariants. The exact significance of the
60-byte offset and the exact roles of the third/fourth signatures remain only
partially understood, and the signatures themselves could differ in other
OMNIC versions.

.. _srs-unknown-fields:

Unknown fields
--------------

The following fields are published for usefulness to future
reverse-engineers even though their meaning is not established. They are
listed as correlated observations only; no unsupported semantic is assigned.

.. list-table::
   :header-rows: 1

   * - Location
     - Size
     - Type
     - Observed pattern
     - Certainty
   * - Series header +24
     - 4
     - Float32
     - Nonzero, per-file varying (≈ 0.001–0.023) in non-RapidScan files, 0.0
       in the RapidScan files examined; registration-like.
     - ``[HYPOTHESIS]`` / ``[UNKNOWN]``
   * - Series header +56
     - 4
     - Float32
     - 1.0 in non-RapidScan, 0.0 in RapidScan files examined.
     - ``[OBSERVED]`` (correlated only)
   * - Series header +84
     - 4
     - Float32
     - 1.0 in TG/GC and HighSpeed, 2.0 in RapidScan files examined.
     - ``[OBSERVED]`` (correlated only)
   * - Series header +184
     - 4
     - Float32
     - Same pattern as +84.
     - ``[OBSERVED]`` (correlated only)
   * - Series header +1048
     - 2
     - UInt16
     - 200 in the tested files.
     - ``[UNKNOWN]``
   * - Key-table entry +4
     - 4
     - UInt32
     - Often 0.
     - ``[UNKNOWN]``
   * - Record prefix +22..75
     - var
     - bytes
     - Mostly per-file constant metadata.
     - ``[OBSERVED]`` values / ``[UNKNOWN]`` semantics
   * - Trailer, all fields except the centisecond counter
     - 12
     - mixed
     - ``0x20`` constant and a zero uint64 in the tested files.
     - ``[OBSERVED]`` values / ``[UNKNOWN]`` semantics

Confound warning: in the current corpus, RapidScan is the only family
observed with a distinct value pattern for the fields below; the HighSpeed and
TG/GC samples (``TGA_demo.srs`` shares the TG/GC structure) all share the
same pattern. These fields therefore "correlate" with the acquisition family
in this sample without being independently confirmed as family markers. They
are more plausibly measurement-mode or axis-registration fields, but nothing
beyond the correlation is established.

Open questions
--------------

* Is ascending-wavenumber raw storage universal for every SRS producer and
  OMNIC version? Only one independently controlled series has SPA ground
  truth so far.
* What are the trailer fields other than the centisecond counter?
* What is the complete SeriesProfile type-code enum (only Area = 6 and
  Chemigram = 0 observed)?
* What is the meaning of the Gram-Schmidt offset (= 10) and what role does the
  Gram-Schmidt matrix play?
* Which boundary-point selection does OMNIC use at the area integration
  limits?
* What is the exact significance of the +60 data-position offset and of the
  third HighSpeed signature occurrence?
* Do the detection signatures and the −152 / +60 positioning relationships
  generalize across OMNIC versions?
* What are the key-table fields at +4 and +12?
* What is the structure and role of the trailer's copied "series metadata"
  region?

.. _srs-implementation-references:

SpectroChemPy implementation references
---------------------------------------

SpectroChemPy implements the knowledge above in
``src/spectrochempy/core/readers/read_omnic.py``:

* :func:`spectrochempy.read_srs` is the public entry point for ``.srs`` files;
* the internal functions ``_read_srs``, ``_read_header`` and
  ``_read_srs_spectra`` apply the record layout, header decoding, and
  normalization described in this page.

The public presentation conventions of the reader are distinct from the raw
storage order documented here:

* spectral records are presented with a **descending** wavenumber axis
  (matching ``read_spa``), with data matched to that axis;
* interferogram records keep the raw ascending data-points coordinate;
* records with an unknown X-axis type are left in raw storage orientation
  with a warning;
* the historical ``reverse_x`` option is deprecated and is a no-op.

The public regression tests in ``tests/test_core/test_readers/test_read_omnic.py``
and the public test-data files listed in :ref:`Tested files and evidence
<srs-tested-files>` serve as reference evidence for the interpretations on
this page. The independently controlled validation series is not distributed.
