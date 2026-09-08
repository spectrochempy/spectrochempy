.. _spa-format:

OMNIC SPA file format
=====================

.. note::

   This is a Phase 1, implementation-informed and explicitly unofficial
   description of the OMNIC SPA format. See :ref:`omnic-file-formats` for
   provenance, limitations, and certainty-level definitions. The page records
   apparent binary structures without treating the current SpectroChemPy
   reader as a normative format specification.

Tested files and evidence
-------------------------

The Phase 1 evidence basis is narrower than for the SRS reference. It consists
primarily of the open-source SpectroChemPy SPA reader, its comments, and its
existing tests. No new binary reverse engineering or controlled native-file
comparison was performed for this page. The existing tests use external OMNIC
files when available and use synthetic bytes for the Experiment Information
decoder.

.. list-table:: Existing SPA test coverage
   :header-rows: 1

   * - File or input
     - Content or purpose
     - Structures exercised
   * - ``7_CZ0-100_Pd_101.SPA``
     - SPA path/bytes equivalence test
     - normal spectrum; bytes-content import
   * - ``7_CZ0-100_Pd_21.SPA``
     - normal SPA spectrum
     - spectral header; intensity block; missing-IFG behavior
   * - ``2-BaSO4_0.SPA``
     - sample and background interferogram retrieval
     - the two interferogram key associations
   * - synthetic ``0x79`` blocks
     - decoder unit tests
     - sequential text-field hypothesis; subtype rejection; short blocks

``[OBSERVED]`` The Phase-1 implementation evidence and existing tests
consistently support the following apparent relationships: a key-record
sequence begins at file offset 304; key ``0x02`` supplies a spectral-header
location; key ``0x03`` supplies a spectrum-data location; keys ``0x66`` and
``0x67`` are associated with the two interferogram paths; and key ``0x82`` is
associated with an Experiment Information block. These are evidence-backed
associations, not yet universal SPA format invariants.

The following remain without independent native-file validation: the timestamp
epoch and timezone, key semantics and table termination, the physical meaning
of unit codes, spectral storage orientation, interferogram identity and
scaling, ZPD semantics, and the Experiment Information layout.

Overall file organization
-------------------------

The currently understood organization is pointer-based rather than a claim
that the data blocks are contiguous:

.. code-block:: text

    OMNIC SPA layout (logical regions currently understood)
    ========================================================

    +----------------------------------------------------------+
    | fixed file-header region                                |
    |   name/title field at 0x1e                              |
     |   time/date-related field at 0x128                      |
    +----------------------------------------------------------+
    | apparent key-record sequence, beginning at 0x130        |
    |   [key | referenced position | referenced length | ???] |
    +----------------------------------------------------------+
       |          |          |          |          |          |
       v          v          v          v          v          v
    spectral   spectrum   comments   history   sample/bg   experiment /
    header     payload    / text     text      IFG data    custom/unknown
    (0x02)     (0x03)     (0x04)     (0x1b)    (0x66/67)   blocks

The diagram shows relationships supported by the Phase-1 evidence. It does not
establish physical ordering, block boundaries beyond the referenced lengths,
or the existence of a declared key-table count. The regions not reached by a
recognized key remain unmapped here.

Offset conventions
------------------

* Offsets 30, 296, and 304 are file-relative, measured from the first byte of
  the SPA file.
* The apparent key-record offsets in the key-table section are entry-relative:
  key byte at +0, referenced position at +2, and referenced length at +6.
* Spectral-header offsets are relative to the position referenced by key
  ``0x02``.
* Positions referenced by keys ``0x03``, ``0x04``, ``0x1b``, ``0x66``,
  ``0x67``, and ``0x82`` are file-position candidates in the Phase-1
  reconstruction. This pointer interpretation is implementation evidence; the
  complete entry schema and pointer rules remain ``[UNKNOWN]``.

File header
-----------

The following candidate fixed fields occur at the indicated file-relative
locations in the Phase-1 reconstruction. The table reports each raw field and
the historical interpretation separately. In particular, the date
interpretation is not an established SPA format fact.

.. list-table:: File-relative fields
   :header-rows: 1

   * - Offset
     - Size and type
     - Apparent meaning
     - Certainty
   * - 30 (``0x1e``)
     - up to 256 bytes
     - Null-padded text candidate for an original OMNIC name/title. A source
       comment calls it the filename under which the spectrum was saved.
     - ``[OBSERVED]`` location in the Phase-1 reconstruction; filename versus
       title semantics are ``[HYPOTHESIS]``.
   * - 296 (``0x128``)
     - 4-byte ``uint32``
     - Time/date-related value historically interpreted as seconds after
       ``1899-12-31 00:00 UTC``.
     - ``[OBSERVED]`` raw field location; epoch, timezone, units, and exact
       semantic role are ``[UNKNOWN]``.
   * - 304 (``0x130``)
     - start of an apparent sequence of 16-byte records
     - Beginning of the apparent key-record area used to locate referenced
       blocks.
     - ``[OBSERVED]`` starting position in Phase-1 evidence; record count and
       physical extent are ``[UNKNOWN]``.

The first 18 bytes are also examined by the shared header decoder. The byte
sequence ``Spectral Data File`` selects the SPA/SPG family in that decoder,
while ``Spectral Exte File`` selects SRS. This signature distinction is shared
decoder logic rather than an independent SPA validation result.

Acquisition timestamp
~~~~~~~~~~~~~~~~~~~~~

The raw value at +296 is represented in the Phase-1 reconstruction as a native
unsigned 32-bit integer. The historical interpretation can be written as:

.. code-block:: text

    date = 1899-12-31 00:00 UTC + raw_value seconds

``[HYPOTHESIS]`` This epoch and timezone may reflect the original author's
interpretation rather than a documented OMNIC convention. The raw type and
location should be validated first against OMNIC-reported acquisition dates;
the historical conversion is not independent evidence for the epoch.

Key-table structure
-------------------

``[OBSERVED]`` Beginning at file offset 304, the Phase-1 evidence is consistent
with a sequence of apparent 16-byte records. This page does not call it a
fixed-size table: no count or total length is established.

.. code-block:: text

   entry offset | size/type          | apparent role                 | status
   ------------ | ------------------ | ----------------------------- | ----------------
   +0           | uint8              | key value                     | [OBSERVED]
   +2           | uint32             | referenced block position    | [OBSERVED] association
   +6           | uint32             | referenced block length      | [OBSERVED] association
   +10 .. +15   | 6 bytes            | unmapped entry data           | [UNKNOWN]

The +2 and +6 fields are pointer and length candidates for recognized records;
their universal applicability and units remain ``[UNKNOWN]``.

The apparent sequence advances by 16 bytes in the Phase-1 reconstruction and stops when the
key byte is ``0x00`` or ``0x01``. ``[HYPOTHESIS]`` These values may be table
terminators, but their roles and universality have not been established. The
source comments also mention preceding ``0x01`` or ``0x0a`` bytes and occasional
``0x01`` records; those observations have not been reconciled into a formal
key-table structure.

Recognized and mentioned keys
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The associations below distinguish properties supported by Phase-1 evidence
from semantic interpretations that remain provisional.

.. list-table:: Key values in the Phase 1 evidence
   :header-rows: 1

   * - Key
     - Association in current evidence
     - Format status
   * - ``0x02``
     - Referenced block associated with the apparent spectral-header region.
     - ``[OBSERVED]`` header association; semantic role is not independently
       validated.
   * - ``0x03``
     - Referenced position and length associated with the apparent spectral
       float payload.
     - ``[OBSERVED]`` spectral-data association; payload role needs native-file
       validation.
   * - ``0x04``
     - Referenced position and length associated with one or more user/custom
       text blocks.
     - ``[OBSERVED]`` text association; complete key meaning is ``[UNKNOWN]``.
   * - ``0x1b``
     - Referenced position and length associated with processing-history text.
     - ``[OBSERVED]`` history association; variants are ``[UNKNOWN]``.
   * - ``0x53``
     - Not decoded. A source comment says it is probably present for a
       retrieved library spectrum.
     - ``[HYPOTHESIS]`` only.
   * - ``0x64``
     - Not decoded.
     - ``[UNKNOWN]``; the source comment gives no semantic interpretation.
   * - ``0x66``
     - Referenced float-data block associated by SpectroChemPy with the
       sample-IFG path.
     - ``[OBSERVED]`` association in the Phase-1 evidence; sample identity is
       ``[HYPOTHESIS]`` pending physical validation.
   * - ``0x67``
     - Referenced float-data block associated by SpectroChemPy with the
       background-IFG path.
     - ``[OBSERVED]`` association in the Phase-1 evidence; background identity
       is ``[HYPOTHESIS]`` pending physical validation.
   * - ``0x69``
     - Not decoded.
     - ``[UNKNOWN]``.
   * - ``0x6a``
     - Not decoded.
     - ``[UNKNOWN]``.
   * - ``0x80``
     - Not decoded.
     - ``[UNKNOWN]``.
   * - ``0x82``
     - First qualifying referenced block is associated with Experiment
       Information.
     - ``[OBSERVED]`` block association; subtype and field layout remain
       ``[HYPOTHESIS]``.
   * - ``0x92``
     - Not decoded. A source comment calls it custom information.
     - ``[UNKNOWN]``.

Spectral header
---------------

The apparent spectral header begins at the position referenced by key ``0x02``.
The following candidate fields occur at offsets relative to that position. They are
presented as fields of the apparent shared header, not as a guarantee that all
SPA versions use the same map.

.. code-block:: text

   offset | type       | candidate interpretation       | certainty
   ------ | ---------- | -------------------------------- | --------------------------
   +4     | uint32     | spectral point count (``nx``)   | [OBSERVED] association
   +8     | uint8      | X-unit code                     | [OBSERVED] location
   +12    | uint8      | data/Y-unit code                | [OBSERVED] location
   +16    | float32    | ``firstx`` endpoint             | [OBSERVED]; meaning [UNKNOWN]
   +20    | float32    | ``lastx`` endpoint              | [OBSERVED]; meaning [UNKNOWN]
   +28    | uint32     | candidate scan-point count     | [OBSERVED]; meaning [UNKNOWN]
   +32    | uint32     | historically ZPD-associated    | [OBSERVED]; relation [UNKNOWN]
   +36    | uint32     | candidate scan count           | [OBSERVED]; meaning pending
   +52    | uint32     | candidate background scans     | [OBSERVED]; meaning pending
   +68    | uint32     | candidate collection duration  | [HYPOTHESIS]
   +80    | float32    | candidate reference frequency  | [HYPOTHESIS]
   +188   | float32    | candidate optical velocity     | [HYPOTHESIS]
   +208   | variable   | candidate processing history   | [OBSERVED]; extent [UNKNOWN]

X-unit codes
~~~~~~~~~~~~

The Phase-1 evidence supplies the following candidate mappings. The numeric
codes are observed in the Phase-1 evidence; the semantic labels remain
provisional unless otherwise noted.

.. code-block:: text

   code   | candidate interpretation       | status
   ------ | ------------------------------ | ------------------------------
   0x01   | wavenumbers, cm^-1             | [OBSERVED] candidate mapping
   0x02   | data points, no physical unit  | [OBSERVED] candidate mapping
   0x03   | wavelength, nm                 | [OBSERVED] candidate mapping
   0x04   | wavelength, um                 | [OBSERVED] candidate mapping
   0x20   | Raman shift, cm^-1             | [OBSERVED] candidate mapping
   other  | unknown X axis                 | [UNKNOWN]

Data/Y-unit codes
~~~~~~~~~~~~~~~~~

.. code-block:: text

   code   | candidate interpretation       | status
   ------ | ------------------------------ | ------------------------------
   0x11   | absorbance                     | [OBSERVED] candidate mapping
   0x10   | transmittance, percent         | [OBSERVED] candidate mapping
   0x0b   | reflectance, percent           | [OBSERVED] candidate mapping
   0x0c   | log(1/R)                       | [OBSERVED]; comment disagrees
   0x0f   | single beam                    | [OBSERVED] candidate mapping
   0x14   | Kubelka--Munk                 | [OBSERVED]; comment disagrees
   0x15   | reflectance, unitless          | [OBSERVED]; meaning [UNKNOWN]
   0x16   | detector signal, V             | [OBSERVED] candidate mapping
   0x1a   | photoacoustic                  | [OBSERVED] candidate mapping
   0x1f   | Raman intensity                | [OBSERVED] candidate mapping
   other  | intensity                      | [UNKNOWN]

Spectrum data
-------------

The block referenced by key ``0x03`` is a candidate sequence of numeric values.
In the Phase-1 reconstruction, the apparent key-record length divided by four
gives the number of referenced ``float32`` values. No explicit byte-order
marker, separate payload header, or independently validated scaling operation
has been identified.

``[OBSERVED]`` The Phase-1 reconstruction associates the payload length with
the candidate header point count ``nx`` and reconstructs a linear X coordinate
from ``firstx`` to ``lastx``. No reversal or numeric transformation is part of
that reconstruction. This is evidence about the present interpretation, not
proof of raw physical storage orientation.

The following distinctions must remain separate:

* the raw ordering of the referenced float payload;
* the meanings and order of the header endpoints;
* a linearly reconstructed coordinate; and
* any presentation convention applied by downstream software.

In particular, the SpectroChemPy public X presentation must not be used as
independent evidence that SPA values are physically stored in either ascending
or descending wavenumber order. Phase 2 should compare raw values,
header endpoints, OMNIC-reported First X/Last X/Data spacing, and an external
physical or exported-spectrum reference.

Interferogram structures
------------------------

Two key records are associated with interferogram payloads in the current
Phase-1 evidence:

* ``0x66`` is used for the request called ``sample``;
* ``0x67`` is used for the request called ``background``.

These labels are ``[OBSERVED]`` associations in the Phase-1 evidence,
not independently established key semantics. Both payloads have the same
apparent position/length and float32 structure as the spectrum. The shared
header also contains candidate scan-point, ZPD, scan-count, background-scan,
and reference-frequency fields, but their relationship to these blocks is not
validated.

The current evidence does not establish:

* whether the two blocks always represent sample and background data;
* whether their values are volts or another detector-domain quantity;
* whether payloads are scaled or transformed;
* whether the header ZPD is expressed in the same index space as the payload;
* how the physical OPD/time coordinate is encoded; or
* whether separate acquisition metadata exist for the background block.

The data-point coordinate, maximum-based ZPD, and laser-frequency handling in
the current reconstruction are intentionally not used here as format
definitions.

Text, comments, and processing history
--------------------------------------

The apparent text-bearing structures are:

* the fixed file-relative name/title field at +30;
* one or more key ``0x04`` referenced text blocks, associated in the current
  Phase-1 evidence with user/custom comments;
* a key ``0x1b`` referenced history block; and
* the variable text associated with spectral-header offset +208, which the
  shared decoder associates with SPA/SPG processing history.

``[OBSERVED]`` The Phase-1 reconstruction treats fixed-length text as null-padded and
tries UTF-8 and Latin-1 decoding. This supports the existence of text-bearing
regions in the current interpretation, but does not establish a universal
encoding, null convention, or complete history layout. Keys ``0x92`` and
possibly ``0x53`` may identify additional custom/library text or metadata, but
their structures are ``[UNKNOWN]``.

Experiment Information
----------------------

Key ``0x82`` is associated with an Experiment Information block. Two competing
descriptions are present in the Phase 1 evidence:

1. **Sequential-field hypothesis.** The executable Phase-1 parser accepts a
   block of at least 50 bytes whose first byte is subtype ``0x79``. Under this
   hypothesis, the first 10 bytes form a header and the remaining bytes split
   at null separators into up to four fields: experiment path, experiment
   filename, accessory name, and experiment title.
2. **Fixed-slot hypothesis.** An older source comment describes fields at
   offsets +10, +90, +254, and +413 and mentions custom text.

The synthetic unit tests establish only that the sequential decoder behaves as
designed for synthetic blocks, including unsupported subtypes and missing
fields. They do not establish that native OMNIC SPA blocks use that layout.
The fixed-slot description is comment-based and likewise remains unvalidated.

Consequently the subtype, header size, field order, field offsets, padding,
encoding, repeat behavior, and custom-text location are ``[UNKNOWN]`` or
``[HYPOTHESIS]``. Native files with varied experiment settings are required to
choose between these descriptions.

Unknown and unmapped fields
---------------------------

The initial unmapped inventory is:

* the six trailing bytes of each apparent key record;
* the key-table count, extent, and meaning of the ``0x00``/``0x01`` stop values;
* unknown key values ``0x64``, ``0x69``, ``0x6a``, and ``0x80``;
* the complete roles of ``0x53`` and ``0x92``;
* the timestamp epoch, timezone, and precision;
* the semantic relationship among ``nx``, scan points, and FFT points;
* unlisted spectral-header fields between and beyond the decoded offsets;
* payload byte order, block headers, scaling, and integrity checks;
* physical spectral orientation and endpoint semantics;
* interferogram identity, scaling, ZPD, and coordinate representation;
* text encoding and history variants; and
* the native Experiment Information layout and custom-text field.

Open questions
--------------

Phase 2 should validate the format itself, rather than merely reproduce the
current reconstruction:

1. What epoch, timezone, and precision does the +296 timestamp use?
2. What bounds or count govern the apparent key-record sequence, and what do
   the six unhandled entry bytes contain?
3. Are the key associations for ``0x02``, ``0x03``, ``0x66``, ``0x67``, and
   ``0x82`` stable across SPA families and OMNIC versions?
4. Which unit meanings correspond to the numeric X/data codes, including the
   ``0x0c``/``0x14`` discrepancy?
5. Is the spectral payload little-endian float32, and is its raw physical order
   the same as the order implied by the header endpoints?
6. How do ``nx``, scan points, FFT points, first/last X, and data spacing
   relate?
7. Do keys ``0x66`` and ``0x67`` always identify sample/background
   interferograms, and what are their scaling, ZPD, and coordinate semantics?
8. Is Experiment Information sequential or fixed-slot, and where are accessory,
   title, experiment filename, and custom text stored?
9. What are the structures and encodings of history, comments, library-related,
   and custom-information blocks?
10. Do these structures vary with detector, beamsplitter, resolution, zero
    filling, scan settings, processing history, or OMNIC version?

Implementation notes
--------------------

The open-source ``read_spa`` implementation is the principal Phase 1 evidence
source. Its current handling is useful for locating candidate fields, but its
parser mechanics, NDDataset metadata mapping, IFG API labels, and robustness
limitations are maintained separately in the Phase 1 audit. They should not be
read as additional format evidence. No production implementation change is
part of this reference.
