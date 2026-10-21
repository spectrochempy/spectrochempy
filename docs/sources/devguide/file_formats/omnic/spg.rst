.. _spg-format:

OMNIC SPG file format
=====================

.. note::

   This is an unofficial interoperability reference for the OMNIC ``.spg``
   format. It is based on independent analysis of lawfully obtained files.
   OMNIC producers and versions may use different subsets of these
   structures. The certainty tags below describe the strength and scope of
   each claim; they do not turn this page into an official vendor
   specification. See :ref:`fileformats-provenance` for provenance and
   :ref:`fileformats-evidence` for the definitions of the certainty levels
   and evidence methods.

Nature of the container
-----------------------

An OMNIC ``.spg`` file is a **container/group of spectra and is not
inherently a temporal series**. Spectra saved in one SPG may:

* come from different acquisitions, with arbitrary acquisition times;
* have different experimental provenance or processing history;
* have been assembled into the group after the fact;
* appear in an order chosen in the OMNIC interface rather than in
  acquisition order;
* be derived or processed spectra mixed with directly collected spectra.

Consequently, this reference does not infer temporal continuity, regular
time spacing, chronological meaning, or causal relationships from common
SPG membership. Any temporal structure across the spectra of a particular
SPG must be demonstrated for that file.

Evidence and scope
------------------

The certainty levels and evidence methods are defined in
:ref:`fileformats-evidence`. The evidence derives from binary inspection of
a small set of observed SPG variants: directly collected spectral groups and
groups of derived/processed (subtraction-result) spectra, including variants
that embed interferogram-related blocks and variants without them. This
evidence base is currently narrower than the one behind the SPA and SRS
references: no comparison with the vendor software and no controlled
acquisition have yet been applied to SPG. Where a structure is shared with
the standalone-spectrum (SPA) family, the SPA reference (:ref:`spa-format`)
carries the stronger evidence, and the rule for claims shared between
formats (:ref:`fileformats-evidence`) applies.

The public test files include three distinct SPG files holding 76 spectra
in total (2, 19, and 55 spectra per file).

Overall file organization
-------------------------

An SPG file is organized as a fixed file header followed by a counted table
of 16-byte key records (the same key-table model as the SPA/SRS families).
The records point to blocks elsewhere in the file; the blocks are not
required to be contiguous. The spectra are represented by per-spectrum sets
of referenced blocks.

.. code-block:: text

   fixed file region
   ├── signature at file offset 0
   ├── embedded group/title text at file offset 30
   ├── nlines (number of key records) at file offset 294
   ├── file-level timestamp-like uint32 at file offset 296
   ├── value of unknown role at file offset 300
   └── key table beginning at file offset 304

   referenced blocks (mostly per-spectrum; arrangement is variant-dependent)
   ├── 0x02  per-spectrum data header
   ├── 0x03  spectral intensity payload
   ├── 0x6b  spectrum title + native timestamp block
   ├── 0x1b  history/acquisition text block
   ├── 0x69, 0x6a, 0x82   auxiliary/parameter/experiment-info blocks
   ├── 0x64..0x67  interferogram-related blocks (content-dependent)
   └── other variant-dependent blocks

The diagram is logical, not to scale: it shows only regions whose
relationships are reproducibly observed.

Offset conventions
------------------

The common OMNIC conventions apply (see :ref:`omnic-conventions`):
file-relative offsets are plain decimal numbers, and ``+N`` offsets are
relative to the start of the block referenced by the key record.

File header and key-record table
--------------------------------

``[ESTABLISHED]`` The signature begins with the ASCII text ``Spectral Data
File``, shared with the SPA family. The embedded group/title text begins at
offset 30 (a text region of up to 256 bytes), ``nlines`` is a little-endian
``uint16`` at offset 294 counting the key records, and the first key record
is at offset 304.

.. list-table:: File-header fields (file-relative)
   :header-rows: 1

   * - Offset
     - Size
     - Type
     - Meaning
     - Certainty
   * - 0
     - 18
     - ASCII
     - File magic ``"Spectral Data File"`` (the SRS format uses
       ``"Spectral Exte File"`` instead).
     - ``[ESTABLISHED]``
   * - 18
     - 12
     - bytes
     - Unmapped bytes.
     - ``[UNKNOWN]``
   * - 30
     - 256
     - text
     - Embedded group/title text. This is the name under which the group
       was saved; it may differ from the current filesystem name if the file
       was renamed outside OMNIC.
     - ``[ESTABLISHED]`` location; interpretation as the saved group name is
       ``[OBSERVED]``
   * - 286
     - 8
     - bytes
     - Not assigned.
     - ``[UNKNOWN]``
   * - 294
     - 2
     - UInt16
     - ``nlines`` — number of key-table records.
     - ``[ESTABLISHED]``
   * - 296
     - 4
     - UInt32
     - File-level timestamp-like value (same unsigned little-endian OMNIC
       encoding as the per-spectrum timestamps; epoch
       1899-12-31 00:00:00 UTC). In every inspected SPG it has matched the
       minimum of the per-spectrum timestamps (see
       :ref:`spg-file-level-timestamp`). Its semantic purpose remains
       unresolved.
     - ``[OBSERVED]`` value relationship;
       ``[UNKNOWN]`` semantics
   * - 300
     - 4
     - UInt32
     - Value of unknown role (distinct small integer values were observed
       across files).
     - ``[OBSERVED]`` presence; ``[UNKNOWN]`` semantics
   * - 304
     - n × 16
     - KeyTable[]
     - Key table (below).
     - ``[ESTABLISHED]``

Key-record entry
~~~~~~~~~~~~~~~~

``[ESTABLISHED]`` Each key record occupies 16 bytes and follows the common
OMNIC key-record layout (see :ref:`omnic-key-table`). ``nlines`` counts the
active records; the table region may be followed by variant-dependent padding
or terminators, to which no universal interpretation is assigned.

Recognized key associations
---------------------------

A key identifies a block in the inspected variants; it does not by itself
determine the signal interpretation of that block. Blocks associated with
one spectrum appear as consecutive key-table runs in the inspected files,
but the key order within a run varies, and trailing variant-dependent
arrangements also occur; record order is not a semantic guarantee.

.. list-table:: SPG key associations
   :header-rows: 1

   * - Key
     - Generic association
     - Certainty and scope
   * - ``0x02``
     - Per-spectrum data header (see :ref:`omnic-data-header` for the field
       map).
     - ``[ESTABLISHED]`` association; one block per spectrum in all
       inspected variants.
   * - ``0x03``
     - Spectral intensity payload.
     - ``[ESTABLISHED]`` association in the inspected variants.
   * - ``0x1b``
     - Per-spectrum history/acquisition text block.
     - ``[ESTABLISHED]`` association where present; content and length vary
       by spectrum and processing history.
   * - ``0x64`` / ``0x65``
     - 24-byte companion blocks accompanying ``0x66`` / ``0x67`` payloads.
     - ``[ESTABLISHED]`` presence and boundaries; pairing ``[OBSERVED]``;
       semantics ``[UNKNOWN]``.
   * - ``0x66`` / ``0x67``
     - Large payload blocks interpreted as sample/background interferograms
       by working analogy with the SPA acquired-pair variant.
     - ``[ESTABLISHED]`` presence; interferogram interpretation is
       ``[HYPOTHESIS]`` for SPG (not independently verified in this format).
   * - ``0x69``
     - Recurring 12-byte auxiliary block.
     - ``[ESTABLISHED]`` presence and boundaries; semantics ``[UNKNOWN]``.
   * - ``0x6a``
     - Spectrometer/acquisition parameter block, structurally matching the
       SPA ``0x6a`` layout.
     - ``[ESTABLISHED]`` presence and 56-byte boundaries; structural match
       ``[OBSERVED]``; SPA field relationships are working assumptions.
   * - ``0x6b``
     - Spectrum title + native timestamp block (SPG-specific).
     - ``[ESTABLISHED]`` association; see below.
   * - ``0x82``
     - Experiment Information blocks (multiple per spectrum).
     - ``[ESTABLISHED]`` association; subtype semantics scoped below.

The set of keys present for a spectrum is content-dependent: directly
collected spectra and derived/processed spectra in the inspected variants
carry different subsets (for example, interferogram-related blocks occurred
only in some variants). No universal per-spectrum block set is claimed.

Spectrum grouping model
-----------------------

``[ESTABLISHED]`` The number of spectra in an inspected SPG equals the number
of ``0x02`` key occurrences, and each spectrum carries **its own** ``0x02``
header — the header is per-spectrum, not stored once per file.

``[OBSERVED]`` In the inspected variants, every spectrum also carries an
``0x03`` payload block and an ``0x6b`` title/timestamp block, with counts
equal to the ``0x02`` count. No universal rule is claimed that every SPG
must contain exactly these keys for every spectrum.

``[OBSERVED]`` The storage order of the spectra in the file need not be
chronological: inspected variants were stored in non-chronological order,
including fully reverse-chronological order and mixed content types. The
order is compatible with user-selected group assembly and is not a
format-level time axis.

Per-spectrum data header ``0x02``
---------------------------------

The ``0x02`` block (140 bytes in the inspected variants) holds the shared
OMNIC data header documented field-by-field in :ref:`omnic-data-header`;
the
same relative field map applies. The fields relevant to SPG are:

.. list-table:: Data-header fields used by SPG (block-relative)
   :header-rows: 1

   * - Offset
     - Type
     - Meaning
     - Certainty
   * - ``+4``
     - ``uint32``
     - Stored point count.
     - ``[ESTABLISHED]`` (shared data header).
   * - ``+8``
     - ``uint8``
     - X-unit code (see :ref:`omnic-unit-codes`).
     - ``[ESTABLISHED]`` field; mapping scope as in the SPA reference.
   * - ``+12``
     - ``uint8``
     - Y/data-unit code (see :ref:`omnic-unit-codes`).
     - ``[ESTABLISHED]`` field; mapping scope as in the SPA reference.
   * - ``+16`` / ``+20``
     - ``float32``
     - OMNIC ``Last X`` / ``First X`` endpoints.
     - ``[ESTABLISHED]`` (shared data header); see
       :ref:`spg-payload-orientation`.
   * - ``+28``
     - ``uint32``
     - Scan points.
     - ``[ESTABLISHED]`` field location; interpretation is a working
       assumption inherited from the shared SPA header.
   * - ``+32``
     - ``uint32``
     - Interferogram peak-position field where applicable.
     - ``[ESTABLISHED]`` field location; SPA interpretation is a working
       assumption for SPG and physical ZPD semantics remain ``[UNKNOWN]``.
   * - ``+36``
     - ``uint32``
     - Sample scans.
     - ``[OBSERVED]`` (as in the SPA reference); interpretation is a working
       assumption inherited from the shared SPA header.
   * - ``+44``
     - ``uint32``
     - FFT points.
     - ``[OBSERVED]`` (as in the SPA reference); transform interpretation is
       a working assumption inherited from the shared SPA header.
   * - ``+52``
     - ``uint32``
     - Background scans.
     - ``[OBSERVED]`` (as in the SPA reference); interpretation is a working
       assumption inherited from the shared SPA header.
   * - ``+68``
     - ``uint32``
     - Collection duration multiplied by 100.
     - ``[ESTABLISHED]`` (shared data header).
   * - ``+80``
     - ``float32``
     - Reference/HeNe-class frequency.
     - ``[ESTABLISHED]`` (shared data header).

Scan-related fields (``+28``, ``+32``, ``+36``, ``+52``, …) exist at the
same relative offsets as in the SPA header; their SPA-scoped interpretations
apply as working assumptions only.

.. _spg-header-overreads:

**Offsets beyond the 140-byte block.** As in SPA (see
:ref:`omnic-header-overreads`), header-relative offsets such as ``+188`` and
``+208`` lie **outside** the 140-byte ``0x02`` block.
``[OBSERVED]`` In every inspected SPG, the ``0x6a`` block (56 bytes) is stored
immediately after the ``0x02`` block and the ``0x6b`` block immediately after
``0x6a``, so that:

* header-relative ``+188`` coincides with ``0x6a`` ``+48`` (the optical
  velocity field of the parameter block);
* header-relative ``+208`` coincides with ``0x6b`` ``+12``, inside the
  spectrum title/text region (which explains why its content is a rendered
  datetime for directly collected spectra and an operation name for derived
  spectra).

These values are therefore not ``0x02`` fields: they depend on the
contiguous placement of the following blocks, which is not a format
guarantee. The canonical sources are the ``0x6a`` and ``0x6b`` blocks
located through their own key records.

``[OBSERVED]`` In the inspected SPG files, reference frequency (``+80``)
and optical velocity (``0x6a`` ``+48``) were identical across spectra within
each file, while collection duration (``+68``) differed by a few hundredths
of a second in two files. None of these fields is intrinsically
group-global: each value is stored per spectrum, and files whose members
come from different acquisitions may legitimately differ. Format consumers
must read the value from the relevant spectrum's own blocks.

.. _spg-payload-orientation:

Spectral payload ``0x03`` and X-axis orientation
------------------------------------------------

``[ESTABLISHED]`` The ``0x03`` block contains the stored spectral intensity
payload: little-endian ``float32`` values whose count agrees with the
``+4`` point count of the same spectrum's header.

*Working assumption (maintainer decision).* The individual spectral payload and
the X-axis convention in SPG are treated as following the same OMNIC
conventions as standalone SPA spectra (see :ref:`spa-payload-orientation`): for
spectral payloads, the native storage is descending-wavenumber ordered, with
payload element 0 corresponding to header ``+16`` (``Last X``) and the final
element corresponding to ``+20`` (``First X``). This is a strong working
assumption based on the shared header and payload structure, not an
independently proven byte-for-byte identity claim for SPG; contradictory
SPG-specific evidence would reopen it. It has not been contradicted by any
inspected SPG variant.

.. _spg-title-timestamp:

Spectrum title and timestamp block ``0x6b``
--------------------------------------------

``[ESTABLISHED]`` Each spectrum carries one ``0x6b`` block (264 bytes in the
inspected variants) with the following block-relative layout:

.. list-table:: ``0x6b`` block fields (block-relative)
   :header-rows: 1

   * - Offset
     - Size
     - Type
     - Meaning
     - Certainty
   * - ``+0``
     - 256
     - text
     - Spectrum title/text region (NUL-padded). Observed conventions: a
       human-readable spectrum name, for directly collected spectra often
       followed by a rendered local datetime; operation/result names for
       derived spectra.
     - ``[ESTABLISHED]`` region; naming conventions are ``[OBSERVED]``.
   * - ``+256``
     - 4
     - UInt32
     - Native timestamp, unsigned little-endian, decoded as
       ``1899-12-31 00:00:00 UTC + raw seconds`` (the OMNIC epoch shared
       with the SPA/SRS timestamps).
     - ``[ESTABLISHED]`` storage/decoding.
   * - ``+260``
     - 4
     - bytes
     - Trailing bytes of unknown role.
     - ``[UNKNOWN]``.

**Timestamp semantics.** The storage and decoding of the ``+256`` timestamp
are established. The **physical event represented by the timestamp remains
unknown**: it must not be described as the acquisition start, midpoint,
acquisition end, or save time without independent evidence. See
:ref:`spg-timestamp-relationship` for what is currently observable.

.. _spg-timestamp-relationship:

Textual datetime and collection-length relationship
---------------------------------------------------

``[OBSERVED]`` For **directly collected** spectra in the currently studied
corpus, the native ``0x6b`` timestamp, the textual datetime rendered in the
spectrum's title/header text, and the ``+68`` collection length of the same
spectrum's header satisfy approximately:

.. code-block:: text

   native_timestamp ≈ textual_datetime + collection_length + small residual

The residual is small and tightly clustered in the inspected
directly-collected variants. The relationship is **not** claimed to be
universal, and it is **clearly different** for some processed/derived
spectra, where the title text and the timestamp plausibly refer to different
events. Neither of the two candidate interpretations — the textual datetime
as the acquisition start, or the native timestamp as an
acquisition-completion/save-like event — is established; the physical
semantics of both fields remain explicitly unresolved. Detailed residual
statistics and per-variant observations are maintained outside this
reference.

.. _spg-file-level-timestamp:

File-level timestamp-like field at offset 296
---------------------------------------------

``[OBSERVED]`` The unsigned little-endian ``uint32`` at file offset 296
decodes with the OMNIC epoch like the per-spectrum timestamps, and in every
currently inspected SPG file its decoded value has matched the **minimum**
of the per-spectrum ``0x6b`` timestamps.

This is recorded strictly as an observed binary relationship. The semantic
purpose of the field is **unknown**: this reference deliberately does not
name it a group acquisition date, group creation date, "group collected"
field, or first acquisition time, because no independent evidence
establishes any of those meanings. It is also not implied to be a group
"start" in any sense — an SPG is a heterogeneous container, and the
minimum member timestamp is not intrinsically a property of the group as a
whole.

History blocks ``0x1b``
-----------------------

``[ESTABLISHED]`` Per-spectrum ``0x1b`` text blocks exist and may contain
scientifically useful textual metadata. ``[OBSERVED]`` content classes seen
in the inspected variants include:

* acquisition information (e.g. resolution and X range, instrument/bench
  identifiers);
* **background references** — the background file path and the background
  collection datetime, where a background was used;
* processing history for processed/derived spectra (the block is longer and
  accumulates operation records).

No stable formal grammar is claimed for the block; the text is read as
NUL-terminated lines whose structure and completeness vary by save operation
and processing history. These blocks are format content: whether and how a
consumer exposes them is an implementation choice, not a format fact.

Spectrometer and acquisition parameter block ``0x6a``
-----------------------------------------------------

``[ESTABLISHED]`` Each spectrum in the inspected variants carries a 56-byte
``0x6a`` block whose size and byte layout are consistent with the SPA
spectrometer/acquisition parameter block (see :ref:`omnic-parameter-block`).
The mature
SPA field relationships (``+16`` digitizer bits, ``+20``/``+24`` filters,
``+44`` sample gain, ``+48`` optical velocity, and the instrument-family
codes) are therefore treated as **working assumptions** for SPG: those field
relationships have not been independently re-validated on SPG data, and
names inherited from the SPA analogy must not be presented as established
SPG semantics.

Auxiliary block ``0x69``
------------------------

``[ESTABLISHED]`` A recurring 12-byte ``0x69`` block accompanies each
spectrum in the inspected variants. Its semantics remain ``[UNKNOWN]`` (see
:ref:`omnic-auxiliary-blocks`).

Experiment Information blocks ``0x82``
---------------------------------------

``[ESTABLISHED]`` Multiple ``0x82`` Experiment Information blocks occur per
spectrum (two or three in the inspected variants), and all ``0x82`` blocks
must not be assumed to represent the same structure:

* subtype ``0x79`` — fixed-slot layout, established and shared with the SPA
  family (see :ref:`omnic-experiment-information`); observed length
  constant in the inspected variants;
* subtype ``0x9e`` — observed preceding the ``0x79`` blocks, with
  variant-dependent length; semantics ``[UNKNOWN]``;
* subtype ``0xa6`` — a small block observed in one inspected content type;
  semantics ``[UNKNOWN]``.

Only the semantics of subtype ``0x79`` are currently described; the other
subtypes are structurally recognized but unresolved.

Interferogram-related blocks ``0x64``–``0x67``
----------------------------------------------

``[ESTABLISHED]`` The inspected SPG variants contain large payload blocks
referenced by ``0x66`` (and ``0x67`` in some variants), each accompanied by
a 24-byte ``0x64`` (or ``0x65``) companion block. Their lengths are
compatible with arrays of 4-byte values, but the element type and signal
identity have not been independently established for SPG.

The interpretation of ``0x66``/``0x67`` as **sample/background
interferograms** follows the validated SPA acquired-pair variant by
analogy. For SPG this remains a **working interpretation (hypothesis)**: it
has not been independently verified in this format, and no consumer should
treat those blocks as interferograms without acknowledging that boundary.
The roles of the 24-byte companion blocks are ``[UNKNOWN]``. The presence
and pairing of these blocks are content-dependent.

Unit codes
----------

SPG uses the shared OMNIC X-unit and Y/data-unit code tables
(:ref:`omnic-unit-codes`). The inspected variants used wavenumber X units and
absorbance data units; the broader shared mappings apply with the SPA
reference's scoping.

Open questions and limitations
------------------------------

The following are useful format-research landmarks; they must not be
assigned unsupported meanings:

* the physical event represented by the ``0x6b`` timestamp and by the
  textual datetimes (start/midpoint/end/save/other) — unresolved;
* the semantic purpose of the file-level ``uint32`` at offset 296 and of the
  value at offset 300;
* the file-header bytes 18–29 and 286–293;
* the semantics of ``0x69``, of the ``0x82`` subtypes ``0x9e`` and
  ``0xa6``, and of the ``0x64``/``0x65`` companion blocks;
* independent verification of the ``0x66``/``0x67`` interferogram
  interpretation for SPG;
* the trailing bytes of the ``0x6b`` block (``+260..+263``);
* the universality of the observed per-spectrum block sets and record
  ordering across OMNIC producers and versions;
* the behavior of variants whose spectra carry inconsistent X-axis
  definitions.
