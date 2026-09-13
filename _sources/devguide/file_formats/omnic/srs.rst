.. _srs-format:

OMNIC SRS file format
=====================

.. note::

   This is an unofficial interoperability reference for the OMNIC ``.srs``
   format. It is based on independent controlled binary and oracle analysis
   across several observed SRS variants. OMNIC producers and versions may use
   different subsets of these structures. The certainty tags below describe
   the strength and scope of each claim; they do not turn this page into an
   official vendor specification. See :ref:`omnic-file-formats` for
   provenance and the definitions of the certainty levels. Contributors with
   access to other OMNIC ``.srs`` variants are encouraged to report
   differences so that this reference can be improved.

Evidence and scope
------------------

This reference separates structural evidence from semantic interpretation:

* ``[ESTABLISHED]`` means that a structure or relationship is supported by
  independent binary/oracle evidence, controlled behavior, or exact
  arithmetic in multiple native cases.
* ``[OBSERVED]`` means that a reproducible structure or correlation was seen
  in the analyzed variants but is not established as universal.
* ``[HYPOTHESIS]`` means that a semantic interpretation is plausible but not
  demonstrated.
* ``[UNKNOWN]`` means that the position or structure may be known while its
  meaning remains unresolved.

The evidence derives from independent controlled binary and oracle analysis
across several observed SRS variants, spanning the RapidScan, HighSpeed and
TG/GC acquisition families: ordinary spectral series, rapid-scan
interferogram series, reprocessed series, and background-containing variants.
One independently controlled TG/GC series served as a controlled OMNIC
export oracle: its individual spectra were exported by OMNIC as SPA files and
compared against the raw SRS samples (see :ref:`srs-spectral-sample-order`),
and the same series also exercised the trailer's SeriesProfile, Gram-Schmidt
and Area structures (see :ref:`srs-seriesprofile`). These descriptions are
generic; they do not depend on particular proprietary files or on the
availability of OMNIC-distributed examples. Specimen-level provenance and the
detailed audit records behind these claims are maintained outside this
reference.

Overall file organization
-------------------------

A ``.srs`` file is organized as a fixed file header followed by a counted
table of 16-byte key records (the same key-table model as the SPA/SPG
families). The first key record references the series metadata region; the
spectral data are stored as a sequence of repeated fixed-stride records.

.. code-block:: text

   file header / key-table region
   ├── file magic at file offset 0
   ├── level-of-processing flag at file offset 292
   ├── nlines (number of key records) at file offset 294
   ├── native series-level 'Collected' timestamp at file offset 296
   └── key records beginning at file offset 304

   referenced blocks
   ├── series metadata region (referenced by the first key record)
   │     └── series-header base: key[0] position, 152 bytes before the first
   │         detection signature
   ├── background header / background data (when a background is present)
   ├── repeated spectrum records: [84-byte prefix | nx * 4-byte float32
   │     payload | 16-byte trailer] repeated ny times
   ├── post-series profile region: Gram-Schmidt-related data, copied series
   │     metadata, SeriesProfile block(s)
   └── other variant-dependent blocks

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
  (at the position referenced by the ``key[0]`` entry — entry offset +2 —
  i.e. the position of the first detection signature minus 152; see
  :ref:`srs-detection-signatures`).
* Repeated-record offsets use the record start as origin; trailer offsets use
  the trailer start; SeriesProfile fields use the block start as origin.

Header and key structures
-------------------------

File header / key-table region (file-relative)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 10 6 6 58 20
   :header-rows: 1

   * - Offset
     - Size
     - Type
     - Meaning
     - Certainty
   * - 0
     - 18
     - ASCII
     - File magic ``"Spectral Exte File"`` (followed by ``\r\n``). The
       SPA/SPG sibling formats use ``"Spectral Data File"`` instead.
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
       in the observed variants; other values also occur.
     - ``[OBSERVED]``
   * - 294
     - 2
     - UInt16
     - Number of key-table entries (24–33 in the observed variants).
     - ``[OBSERVED]``
   * - 296
     - 4
     - UInt32
     - Native series-level acquisition timestamp (``Collected``): seconds
       since the OMNIC epoch (1899-12-31 00:00:00 UTC), unsigned
       little-endian. This is the canonical series-absolute anchor used
       by the reader (see :ref:`srs-time-representation`). Zeroed in
       reprocessed files; holds unrelated bytes in GC variants.
     - ``[ESTABLISHED]`` for RapidScan / HighSpeed / TGA;
       ``[OBSERVED]`` absence for GC
   * - 304
     - n × 16
     - KeyTable[]
     - Key table (see below).
     - ``[OBSERVED]`` structure

Key-table entry (16 bytes; entry-relative offsets)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``[ESTABLISHED]`` Each key record occupies 16 bytes and follows the generic
OMNIC key-record layout also used by the SPA/SPG families (see
:ref:`spa-format`): a 1-byte key, a reserved/variant byte, a 4-byte
file-relative block position, a 4-byte block length and 6
trailing/variant-dependent bytes. Only the first entry — the **series**
entry — is interpreted here; the remaining entries are unresolved.

.. list-table::
   :widths: 10 6 6 58 20
   :header-rows: 1

   * - Offset
     - Size
     - Type
     - Meaning
     - Certainty
   * - 0
     - 1
     - UInt8
     - Key / role indicator (``0x02`` identifies the series entry;
       ``nlines`` at offset 294 counts the entries).
     - ``[ESTABLISHED]`` for the first entry
   * - 1
     - 1
     - UInt8
     - Reserved / variant-dependent byte (0 in the observed variants).
     - ``[OBSERVED]``
   * - 2
     - 4
     - UInt32
     - File-relative position of the referenced block. For the first
       entry this is the series-header base, exactly 152 bytes before the
       first detection signature (identical in all observed variants).
     - ``[ESTABLISHED]`` for the first entry
   * - 6
     - 4
     - UInt32
     - Referenced block length in bytes (0 for the first entry in the
       observed variants).
     - ``[OBSERVED]`` value / ``[UNKNOWN]`` role
   * - 10
     - 6
     - bytes
     - Trailing / variant-dependent data; ``8c 00 00 00 01 00`` for the
       first entry in all observed variants.
     - ``[OBSERVED]``

Series metadata
---------------

In the observed variants, the series-header base is located 152 bytes before
the first detection signature (see :ref:`srs-detection-signatures`). Offsets
in the tables below are relative to that base (so the ``+296`` row below is a
series-header-relative field, unrelated to the file-relative offset 296 —
the native ``Collected`` timestamp — of the header table above). The metadata
associated with the series extend well beyond the first 152 bytes: the exact
subdivision into OMNIC internal blocks is only partially understood. Only the
fields listed below have been interpreted so far; the rest of the region is
unmapped.

Signal and acquisition fields
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 10 6 6 58 20
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
     - ``[ESTABLISHED]`` for codes 1 and 2; ``[OBSERVED]`` for the
       others
   * - 12
     - 1
     - UInt8
     - Y/data-unit code: 0x11 absorbance, 0x10 transmittance (%),
       0x0F single beam, 0x16 detector signal (V), 0x1A photoacoustic,
       0x1F Raman intensity, among others shared with the SPA/SPG
       readers.
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
     - Per-variant varying, registration-like factor; see
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
     - Interferogram peak position (matches OMNIC's series-info field
       in all observed variants). Equivalence with the physical
       zero-path-difference position is not established.
     - ``[ESTABLISHED]``
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
     - Value pattern that differs between RapidScan and other observed
       variants; see :ref:`srs-unknown-fields`.
     - ``[OBSERVED]`` (correlated only)
   * - 68
     - 4
     - UInt32
     - General-header *collection length* in 1/100 s. This is the
       shared acquisition-time field of the OMNIC header family; it must
       **not** be conflated with the series minimum/first time at +1002
       (nor with the public SRS ``collection_length`` derived from
       +1006).
     - ``[OBSERVED]``
   * - 80
     - 4
     - Float32
     - Reference (laser) frequency.
     - ``[OBSERVED]``
   * - 84
     - 4
     - Float32
     - OMNIC **sample spacing** (matches the series-info field of the
       same name in all observed variants). Varies with acquisition
       configuration, not with the acquisition family.
     - ``[ESTABLISHED]``
   * - 184
     - 4
     - Float32
     - Not covered by OMNIC's reported series-info fields; in some
       observed variants its value differs from that of +84; see
       :ref:`srs-unknown-fields`.
     - ``[OBSERVED]`` value / ``[UNKNOWN]`` semantic
   * - 188
     - 4
     - Float32
     - Optical velocity.
     - ``[OBSERVED]``
   * - 208
     - var
     - text
     - Spectrum history text; a record whose text starts with
       ``Background`` marks a background header.
     - ``[ESTABLISHED]`` string presence; ``[OBSERVED]`` role

Series identity and time-model fields
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 10 6 6 58 20
   :header-rows: 1

   * - Offset
     - Size
     - Type
     - Meaning
     - Certainty
   * - 296
     - 8
     - UInt64
     - Raw file size (series-header-relative; not to be confused with the
       file-relative offset 296 ``Collected`` timestamp above).
     - ``[HYPOTHESIS]``
   * - 938
     - ≤ 256
     - text
     - Series name (the reader splits on the first newline).
     - ``[OBSERVED]``
   * - 1002
     - 4
     - Float32
     - **Series minimum / first time, in minutes** — the time-axis
       anchor. Historically misinterpreted as a "collection length";
       the general-header collection-length field is the one at +68 (not
       to be conflated either). The public ``collection_length`` for SRS
       series is derived from +1006, not from this field.
     - ``[ESTABLISHED]``
   * - 1006
     - 4
     - Float32
     - Series maximum / last time, in minutes. Converted to seconds
       (× 60) it is the OMNIC "Total collection time" and the public
       ``collection_length`` exposed by the reader for SRS series.
     - ``[ESTABLISHED]``
   * - 1010
     - 4
     - Float32
     - **Regular time step, in minutes** (historically misnamed
       ``firsty``; it is *not* the series minimum). See
       :ref:`srs-time-representation`.
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
   * - 1036
     - 4
     - Float32
     - Flow-cell temperature (°C; TG/GC variants only). Matches
       OMNIC's reported series-info value in the TG/GC variants; 0.0
       in the other variants.
     - ``[ESTABLISHED]`` (TG/GC variants)
   * - 1040
     - 4
     - Float32
     - Transfer-line temperature (°C; TG/GC variants only). Matches
       OMNIC's reported series-info value in the TG/GC variants; 0.0
       in the other variants.
     - ``[ESTABLISHED]`` (TG/GC variants)
   * - 1044
     - 2
     - UInt16
     - Gram-Schmidt offset; see :ref:`srs-gram-schmidt`.
     - ``[OBSERVED]`` value / ``[UNKNOWN]`` semantic
   * - 1046
     - 2
     - UInt16
     - Gram-Schmidt interferogram points; see :ref:`srs-gram-schmidt`.
     - ``[OBSERVED]`` / ``[HYPOTHESIS]``
   * - 1048
     - 2
     - UInt16
     - Constant value in the observed variants; see
       :ref:`srs-unknown-fields`.
     - ``[UNKNOWN]``
   * - 1200
     - var
     - text
     - Initial history text (pristine variants). Reprocessed variants
       carry their updated history at the end of the file after a
       16-byte ``0xff`` sequence.
     - ``[OBSERVED]``

Background data
---------------

``[OBSERVED]`` Spectral backgrounds have their own header, located through
the second detection signature (signature position − 152), with their own X
endpoints. In the observed variants where both headers were decoded, the
**endpoint ordering can differ from that of the series header**: spectral
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

``[ESTABLISHED]`` In the observed SRS series, each spectrum record has the
following layout. The same repeated-record structure is also observed for the
rapid-scan interferogram records, so the description below is generic to both
spectral series and interferogram records:

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
  ``84 + nx·4 + 16`` bytes per record).
* ``[ESTABLISHED]`` The trailer size (16 bytes) holds for all observed
  variants.

84-byte record prefix (record-relative offsets)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 12 6 6 56 20
   :header-rows: 1

   * - Offset
     - Size
     - Type
     - Meaning
     - Certainty
   * - 0 .. null
     - var
     - ASCII
     - Spectrum name, null-terminated.
     - ``[ESTABLISHED]``
   * - 22 .. 75
     - var
     - bytes
     - Mostly per-file constant metadata; semantics not fully
       interpreted.
     - ``[OBSERVED]`` values / ``[UNKNOWN]`` semantics
   * - 76
     - 4
     - Float32
     - Spectrum minimum Y value.
     - ``[OBSERVED]``

.. _srs-inter-spectrum-trailer:

Inter-spectrum trailer (16 bytes; trailer-relative offsets)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``[ESTABLISHED]`` Observed records (both spectral series and rapid-scan
interferograms) are followed by a 16-byte trailer.

``[OBSERVED]`` One uint32 field in the trailer of the *non-final* records
behaves as a **cumulative time counter in centiseconds** (values increase
with spectrum index and track the series time axis). The final record's
trailer is instead repurposed for series statistics.

``[UNKNOWN]`` The remaining trailer fields are not fully interpreted.

.. list-table::
   :widths: 10 6 6 58 20
   :header-rows: 1

   * - Offset
     - Size
     - Type
     - Meaning
     - Certainty
   * - 0
     - 4
     - UInt32
     - ``0x20`` (32) in the observed variants.
     - ``[OBSERVED]``
   * - 4
     - 4
     - UInt32
     - Cumulative elapsed-time counter in centiseconds for the **next**
       spectrum (non-final records only; see
       :ref:`srs-time-representation`).
     - ``[OBSERVED]``
   * - 8
     - 8
     - UInt64
     - Zeros in the observed variants.
     - ``[OBSERVED]``

.. _srs-detection-signatures:

Detection signatures and positioning observations
-------------------------------------------------

``[OBSERVED]`` In the observed variants, each acquisition family is preceded
by a recognizable 10–16 byte signature:

* **RapidScan** — ``\x02\x00\x00\x00\x18\x00\x00\x00\x00\x00\x48\x43\x00\x50\x43\x47``,
  occurring 3 times;
* **HighSpeed** — ``\x02\x00\x00\x00\x18\x00\x00\x00\x00\x00\x48\x43\x00\xc8\xaf\x47``,
  occurring 4 times;
* **TG/GC** — ``\x02\x00\x00\x00\x18\x00\x00\x00\x00\x00`` (first 10 bytes),
  occurring 3 times; the following bytes vary between files.

Positioning relationships (all values verified in the observed variants):

* the **first** signature occurs 152 bytes after the series-header base
  (the position field of the ``key[0]`` entry — entry offset +2 — points at
  the same header: ``[ESTABLISHED]`` for ``key[0]`` across the observed
  variants; ``[OBSERVED]`` as a general rule);
* the **second** signature occurs 152 bytes before the background header;
* the **last** signature occurs 60 bytes before the series spectral-data
  start (data = signature position + 60);
* in HighSpeed variants, the fourth signature occurrence is the
  data-position one.

These are **observed** positioning relationships, not guaranteed layout
invariants.

.. _srs-seriesprofile:

SeriesProfile structures
------------------------

``[HYPOTHESIS]`` A recurring trailer structure interpreted as an OMNIC
**SeriesProfile** block has been identified in the observed variants. The
block boundaries and value shapes are reproducible; their full meaning is
not.

.. list-table::
   :widths: 12 6 6 56 20
   :header-rows: 1

   * - Offset
     - Size
     - Type
     - Meaning
     - Certainty
   * - 0
     - 4
     - UInt32
     - ``ny`` marker equal to the series spectrum count — a reliable
       block boundary marker.
     - ``[ESTABLISHED]`` boundary; ``[OBSERVED]`` value
   * - 4
     - 4
     - UInt32
     - Profile-type code: 6 for "peak area of one peak" (Area), 0 for
       Chemigram, in the observed variants.
     - ``[HYPOTHESIS]`` (higher confidence for code 6, medium for
       code 0)
   * - 8
     - 4
     - UInt32
     - Gram-Schmidt offset, copied unchanged from the series metadata
       into every block examined.
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
       ``Chemigram: ...``.
     - ``[ESTABLISHED]`` string; ``[OBSERVED]`` content
   * - after header
     - var
     - padding
     - Header padding to a fixed block-relative data offset in the
       observed version.
     - ``[OBSERVED]``
   * - data offset
     - ny × 4
     - Float32[]
     - Per-spectrum profile values vector.
     - ``[ESTABLISHED]`` vector shape; semantics ``[HYPOTHESIS]``

``[OBSERVED]`` In the observed variants the block boundaries, the
per-spectrum vector shape, and the correspondence between the ASCII label
and the vector are reproducible, and a series defines one block per profile.
The exact byte stride and header/data grouping are **version-dependent**: a
fixed stride observed in one OMNIC-written file was absent in a file re-saved
by a different OMNIC version (headers packed together, data vectors grouped
separately). A parser must therefore discover the profile blocks dynamically
(e.g. via the ``ny`` markers and the ASCII labels) rather than assume a fixed
count or stride.

.. _srs-time-representation:

Time representation
-------------------

Four quantities describe the series time axis; they must not be conflated:

* **Series minimum / first time** — header +1002 (Float32, minutes). This is
  the time of the first spectrum and the anchor of the axis.
* **Last / max time** — header +1006 (Float32, minutes).
* **Regular time step** — header +1010 (Float32, minutes). In the
  independently controlled series this equals ``(last time − first time) /
  (ny − 1)`` exactly to float32 precision, which confirms it is a step, not
  a minimum (the historical ``firsty`` misreading).
* **Number of spectra** — ``ny`` (header +1026).

``[OBSERVED]`` the time axis follows the regular model

.. code-block:: text

   T[i] = time_min + i * step        (i = 0 .. ny-1)

where ``time_min`` reads from +1002 and ``step`` from +1010.

The trailer's centisecond counter provides an independent confirmation:
``[OBSERVED]`` for non-final records, the counter stored in the trailer of
spectrum *i* holds the elapsed time of spectrum *i+1*, quantized to integer
centiseconds; dividing by 6000 (centiseconds per minute) reproduces the time
axis and matches the 3-decimal times embedded in the spectrum names. The
final record's trailer is instead repurposed for series statistics. In the
RapidScan variant the per-spectrum increment equals the collection period; in
the independently controlled series the increment equals the regular time
step above.

Absolute series anchor (``Collected``) and derived datetimes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``[ESTABLISHED]`` for the RapidScan, HighSpeed and TGA samples examined, the
series additionally carries a native absolute series timestamp at
**file-relative offset 296**: a UInt32 OMNIC-epoch timestamp (seconds since
1899-12-31 00:00:00 UTC, unsigned little-endian) following the same
convention as the SPA/SPG timestamps. It equals the series ``Collected``
timestamp reported by OMNIC.  The per-spectrum timestamps OMNIC writes when
exporting the series spectra as SPA files follow Model A below: the first
exported spectrum's timestamp is ``Collected + time_min`` (≈ ``time_min`` ≈
4.96 s after ``Collected`` in the controlled TGA series), **not**
``Collected`` itself.

The field has header-relative copies that the reader verifies against before
trusting it (the offsets differ between families, and are relative to the
series-header base):

* series-header base + 836 for RapidScan / HighSpeed;
* series-header base + 368 and series-header base + 828 for TG/GC-family
  TGA files.

``[OBSERVED]`` The copy-check prevents false positives: GC files, whose
offset 296 holds unrelated bytes (ASCII text / assorted values), and
reprocessed RapidScan files, whose field is zeroed, yield *no* absolute
anchor (``None``) rather than a fabricated date. The reader only accepts the
evidence-backed copy locations listed above; a layout with different copy
offsets is unsupported and yields ``None``.

Combined with the regular time model above, the per-spectrum absolute
datetimes follow Model A

.. code-block:: text

    datetime[i] = Collected + timedelta(minutes=time_min + i * step)

computed in full precision from the native float32 ``time_min`` (+1002) and
``step`` (+1010) fields, preserving the full precision represented by the
native float32 timing fields rather than the already-rounded three-decimal
public Y coordinate. The derived datetimes agree with the OMNIC-exported
SPA timestamps at their serialization precision (whole seconds), while
SpectroChemPy retains the full precision derived from the native series
fields.
The reader exposes the anchor through the standard ``acquisition_date``
convention and the derived datetimes as an additional Y-label column (see
:ref:`srs-implementation-references`).  Which acquisition event (integration
start, midpoint, end, ...) the OMNIC timestamp corresponds to is not
experimentally established (see the Open questions section); no physical
start/end interpretation is implied.

.. _srs-spectral-sample-order:

Spectral sample order
---------------------

``[OBSERVED]``

In the SRS variants examined, spectral intensity samples are stored in
**ascending-wavenumber physical order** (sample 0 = lowest wavenumber). This
ordering was independently confirmed for one controlled series against
individual spectra exported by OMNIC as SPA files: the raw SRS samples match
the OMNIC-exported SPA data exactly in the correct physical order, while the
reversed orientation does not match. It should not yet be treated as a
guaranteed invariant for every SRS producer/version.

This page describes the **raw storage order**. Presentational conventions of
particular software are separate: SpectroChemPy, for instance, presents SRS
spectral datasets with a *descending* wavenumber axis (matching its SPA
convention) while the raw file order is ascending (see
:ref:`srs-implementation-references`).

Rapid-scan interferograms
-------------------------

RapidScan variants contain rapid-scan interferogram records. ``[OBSERVED]``
These interferogram records carry an X-axis of type "data points" (x-unit
code 2, no physical wavenumber unit), stored with ascending data-points
coordinates.

Do **not** equate "no xunit code" with "interferogram": a record whose X-unit
code is unrecognized also lacks a physical xunit, yet is not a data-points
interferogram. The current reader therefore distinguishes three cases:

* a known spectral axis (x-unit codes 1/3/4/32);
* an explicit data-points interferogram (x-unit code 2);
* an unknown X-axis type (unrecognized code), which is neither treated as
  an interferogram nor spectral-normalized.

This three-way distinction is presented as the current observed evidence for
the variants examined, not as a universal OMNIC invariant.

Gram-Schmidt / Chemigram / Area structures
------------------------------------------

.. _srs-gram-schmidt:

Gram-Schmidt data
~~~~~~~~~~~~~~~~~

``[OBSERVED]`` In the independently controlled series examined, the
post-series trailer/profile region begins with a Gram-Schmidt-related
region: a small header of two uint32 values (``ny`` and the number of
Gram-Schmidt points) followed by a matrix of ``ny × gs_points`` float32
values. Whether every SRS family places this region first in the trailer
has not been verified.

``[OBSERVED]`` The number of Gram-Schmidt interferogram points also appears
as a series-header field at +1046 (UInt16), and the Gram-Schmidt offset
value (header +1044, UInt16) is copied into every SeriesProfile block
examined. ``[UNKNOWN]`` The exact meaning of the offset and the
reconstruction role of the matrix are not yet established.

Chemigram / Area observations (experimental)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``[OBSERVED]`` Profile vectors whose areas were recomputed from the spectral
data closely reproduce the stored vectors, with small residuals where
applicable.

``[HYPOTHESIS]`` The stored "Area" profile values follow a
baseline-subtracted integral of the intensity over the labelled region
(``Area = ∫(I − baseline) dx`` using the ascending-wavenumber raw order,
with the baseline joining the region-endpoint intensities). The small
residual is consistent with a slight difference in the boundary-point
selection at the integration limits. No numeric copy of the integration
limits has been identified in the structures examined; in the examined
blocks, the limits are present in the ASCII label.

.. _srs-unknown-fields:

Unknown fields
--------------

The following fields are published for usefulness to future
reverse-engineers even though their meaning is not established. They are
listed as correlated observations only; no unsupported semantic is assigned.

.. list-table::
   :widths: 18 6 6 50 20
   :header-rows: 1

   * - Location
     - Size
     - Type
     - Observed pattern
     - Certainty
   * - Series header +24
     - 4
     - Float32
     - Nonzero and per-file varying in non-RapidScan variants, 0.0 in
       the RapidScan variants examined; registration-like.
     - ``[HYPOTHESIS]`` / ``[UNKNOWN]``
   * - Series header +56
     - 4
     - Float32
     - 0.0 in RapidScan variants, 1.0 in the other variants examined.
     - ``[OBSERVED]`` (correlated only)
   * - Series header +184
     - 4
     - Float32
     - 2.0 in RapidScan variants, 1.0 in the other variants examined;
       not covered by OMNIC's reported series-info fields.
     - ``[OBSERVED]`` (correlated only)
   * - Series header +1048
     - 2
     - UInt16
     - Constant value in the observed variants.
     - ``[UNKNOWN]``
   * - Key-table entries other
       than the first
     - 16 each
     - record
     - Key values, referenced positions, lengths and trailing data not
       yet interpreted.
     - ``[UNKNOWN]``
   * - Record prefix +22..75
     - var
     - bytes
     - Mostly per-file constant metadata.
     - ``[OBSERVED]`` values / ``[UNKNOWN]`` semantics
   * - Trailer, all fields
       except the centisecond
       counter
     - 12
     - mixed
     - ``0x20`` constant and a zero uint64 in the observed variants.
     - ``[OBSERVED]`` values / ``[UNKNOWN]`` semantics

Confound warning: in the observed variants, RapidScan is the only family
with a distinct value pattern for the correlated fields listed above
(``+24``, ``+56``, ``+184``); the HighSpeed and TG/GC samples all share the
same pattern. These fields therefore "correlate" with the acquisition family
in this sample without being independently confirmed as family markers;
nothing beyond the correlation is established.

Open questions and limitations
------------------------------

* Which boundary-point selection does OMNIC use at the area integration
  limits? The computed Area values reproduce the stored values within a
  small residual, but the exact boundary-point treatment at the
  integration limits has not been isolated.
* Which acquisition event (integration start, midpoint, end, ...) does the
  per-spectrum OMNIC timestamp correspond to physically? The arithmetic
  mapping (Model A) is established, but the physical semantics of the
  native timestamp are not.

.. _srs-implementation-references:

SpectroChemPy implementation
----------------------------

The reader is implemented in ``src/spectrochempy/core/readers/read_omnic.py``.
:func:`spectrochempy.read_srs` is the public entry point for ``.srs``
files; it delegates to the OMNIC importer with the ``.srs`` file-type
constraint.

Format-to-public-behaviour mapping:

* Spectral series are presented with a **descending wavenumber** X axis
  (high to low), matching the convention used by ``read_spa``. The
  normalization is applied per-spectrum from each record's own
  ``firstx``/``lastx`` endpoints.
* Rapid-scan interferograms retain the raw **ascending data-points**
  coordinate. These records are flagged internally (``meta.interferogram``);
  their ZPD index is the data-derived peak index, and the laser frequency is
  set to the reader's standard default rather than derived from these
  records' coordinate or data.
* Records with an unrecognized X-unit code are left in raw storage
  orientation with an informational message; they are neither treated as
  interferograms nor spectral-normalized.
* The Y coordinate is **relative series time in minutes** (``time_min``
  to ``lasty``, rounded to 3 decimal places; title "Time", units
  "minute").
* When a validated native ``Collected`` timestamp exists at file offset
  296 (see :ref:`srs-time-representation`), the reader exposes it through
  ``dataset.acquisition_date`` (the standard SPA convention). The Y
  coordinate then gains a second label column with the per-spectrum
  absolute datetimes derived as
  ``Collected + timedelta(minutes=time_min + i * step)`` in full
  precision from the native series fields.
* Variants without a valid absolute anchor (GC, reprocessed RapidScan)
  leave ``acquisition_date`` unset and keep single-column
  names-only Y labels.
* The public ``collection_length`` metadata attribute is derived from
  header +1006 (series last time × 60), not from the general-header
  field at +68.
* The historical ``reverse_x`` keyword is a deprecated no-op:
  spectral orientation is handled automatically and supplying the
  keyword emits a ``DeprecationWarning``.
