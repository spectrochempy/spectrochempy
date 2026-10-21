.. _omnic-common:

Common OMNIC structures
=======================

The SPA, SPG and SRS formats share a common container model and several
block layouts. This page documents them once; the format pages
(:ref:`spa-format`, :ref:`spg-format`, :ref:`srs-format`) describe only what
is specific to each format. See :ref:`fileformats-evidence` for the
certainty levels.

Unless stated otherwise, the evidence for the shared structures comes from
SPA files, which form by far the largest corpus. Their use in SPG and SRS
follows the rule for claims shared between formats: a structure is
``[ESTABLISHED]`` for SPG or SRS only where the corresponding format page
says so. Scopes such as *the validated spectral family* or *the validated
Raman variant* refer to the SPA variants listed in the SPA evidence corpus.

.. _omnic-conventions:

Conventions
-----------

* **Byte order.** ``[ESTABLISHED]`` All multi-byte integers and floats are
  little-endian.
* **Types.** ``uint8``, ``uint16``, ``uint32`` and ``uint64`` are unsigned
  integers; ``float32`` is an IEEE 754 single-precision value.
* **Offsets.** File-relative offsets are written as plain decimal numbers
  (``offset 294``). Offsets inside a block are written ``+N`` (decimal) and
  are relative to the block start, that is, the position stored in the
  block's key record. Ranges ``+a..+b`` include both ends.
* **Keys and codes.** Key values, subtypes and unit codes are written in
  hexadecimal (``0x02``).
* **Text.** Strings are NUL-terminated and, in fixed-size regions,
  NUL-padded. Multi-line text uses CR LF line separators, and some history
  lines are indented with tab characters. ``[OBSERVED]`` Non-ASCII text is
  stored in a single-byte Windows code page (for example ``é`` as ``0xe9``,
  consistent with Windows-1252) and never as UTF-8 in the inspected files.
  Text content, such as history entries and date formats, follows the
  language of the OMNIC installation that wrote it.

.. _omnic-file-header:

File header
-----------

Every OMNIC file starts with a fixed header followed by a counted table of
16-byte key records. The records point to blocks elsewhere in the file; the
blocks are not required to be contiguous, and their order in the file is not
a format guarantee.

.. list-table:: Shared file-header fields (file-relative)
   :header-rows: 1

   * - Offset
     - Size
     - Type
     - Meaning
     - Certainty
   * - 0
     - 18
     - ASCII
     - Signature: ``Spectral Data File`` for SPA and SPG, ``Spectral Exte
       File`` for SRS.
     - ``[ESTABLISHED]``
   * - 30
     - 256
     - text
     - Saved name/title text (SPA, SPG).
     - ``[ESTABLISHED]`` location
   * - 294
     - 2
     - ``uint16``
     - ``nlines``: number of key records.
     - ``[ESTABLISHED]``
   * - 296
     - 4
     - ``uint32``
     - Timestamp-like value in the OMNIC time encoding (see
       :ref:`omnic-timestamps`); its meaning is format- and
       variant-dependent.
     - See the format pages
   * - 304
     - n × 16
     - records
     - Key table.
     - ``[ESTABLISHED]``

The remaining header bytes are not assigned meanings here; the format pages
list the fields observed for each format. A recognizable signature does not
imply that every producer uses the same complete header map.

.. _omnic-key-table:

Key table
---------

``[ESTABLISHED]`` Each key record occupies 16 bytes:

.. list-table:: Key record (record-relative)
   :header-rows: 1

   * - Offset
     - Type
     - Role
   * - ``+0``
     - ``uint8``
     - Key: identifies the kind of referenced block.
   * - ``+1``
     - ``uint8``
     - Reserved/variant-dependent byte.
   * - ``+2``
     - ``uint32``
     - File-relative position of the referenced block.
   * - ``+6``
     - ``uint32``
     - Length of the referenced block in bytes.
   * - ``+10..+15``
     - bytes
     - Trailing/variant-dependent data.

``nlines`` counts the active records, not every byte between the table and
the first block. A key identifies a kind of block; it does not by itself
determine the signal interpretation of that block. A file may contain
several records with the same key (one per spectrum in SPG, for example).

.. _omnic-timestamps:

Timestamps
----------

``[ESTABLISHED]`` OMNIC timestamps are unsigned little-endian ``uint32``
counts of seconds since 1899-12-31 00:00:00 UTC:

.. code-block:: text

   instant = 1899-12-31 00:00:00 UTC + raw seconds

The decoded value is an absolute UTC instant; files do not store a timezone.
The encoding is shared, but the event a timestamp represents (acquisition,
save, or another event) is format- and field-dependent and is documented on
the format pages. A ``uint32`` count from this epoch overflows on
2036-02-06.

.. _omnic-x-endpoints:

X-axis endpoints and sample order
---------------------------------

The data header stores two ``float32`` X endpoints, at ``+16`` and
``+20``. This reference names them after the OMNIC display terms: ``+16`` is
**Last X** and ``+20`` is **First X**. The X coordinate is linear between
the two endpoints over the stored point count (``+4``).

Which endpoint corresponds to the first stored sample is format-dependent:

* SPA and SPG spectra: ``[ESTABLISHED]`` for SPA, the payload is stored in
  descending wavenumber order, sample 0 corresponding to ``+16`` (Last X)
  (see :ref:`spa-format`);
* SRS series: ``[OBSERVED]`` the samples are stored in ascending coordinate
  order, sample 0 corresponding to the smaller of the two endpoints (see
  :ref:`srs-format`).

Readers should therefore derive the sample order from the format and the
endpoint values rather than from the field names alone.

.. _omnic-data-header:

Data header ``0x02``
--------------------

This reference calls the block referenced by key ``0x02`` the *data
header*: it describes the data stored in the payload (point count, X and Y
units, X endpoints) together with acquisition and transform settings, and it
is used for spectra and interferograms alike. OMNIC's internal name for the
block is not known.

The data-header layout is shared by SPA and SPG spectra and by the SRS
series header (see :ref:`srs-format`). The following offsets are relative to
the block referenced by ``0x02``. The
table deliberately separates mature meanings from observations and unresolved
fields.

.. list-table:: Data-header fields (``0x02``)
   :header-rows: 1

   * - Offset
     - Type
     - Meaning
     - Scope/certainty
   * - ``+4``
     - ``uint32``
     - Stored point count.
     - ``[ESTABLISHED]`` for observed spectral and Raman headers.
   * - ``+8``
     - ``uint8``
     - X-unit code.
     - ``[ESTABLISHED]`` field; mappings below are scoped by variant.
   * - ``+12``
     - ``uint8``
     - Y/data-unit code.
     - ``[ESTABLISHED]`` field; mappings below are scoped by variant.
   * - ``+16``
     - ``float32``
     - OMNIC ``Last X``.
     - ``[ESTABLISHED]`` for the validated spectral family; see
       :ref:`omnic-x-endpoints`.
   * - ``+20``
     - ``float32``
     - OMNIC ``First X``.
     - ``[ESTABLISHED]`` for the validated spectral family; see :ref:`omnic-x-endpoints`.
   * - ``+28``
     - ``uint32``
     - Scan points.
     - ``[ESTABLISHED]`` field location; applicability and relation to FFT points
       are variant-dependent.
   * - ``+32``
     - ``uint32``
     - OMNIC interferogram peak position.
     - ``[ESTABLISHED]`` in validated native IFGs; formal physical ZPD meaning
       remains ``[UNKNOWN]``.
   * - ``+36``
     - ``uint32``
     - Sample scans.
     - ``[OBSERVED]`` field and interpretation in acquired variants.
   * - ``+40``
     - ``float32``
     - Duplicate numeric peak position.
     - ``[ESTABLISHED]`` match to ``+32`` in validated native IFGs; broader
       applicability is ``[OBSERVED]``.
   * - ``+44``
     - ``uint32``
     - FFT points.
     - ``[OBSERVED]`` transform-geometry field.
   * - ``+48``
     - ``uint32``
     - Transform/trailing geometry field.
     - ``[ESTABLISHED]`` as ``N_stored - P`` in validated native IFGs; exact
       native semantic name remains ``[UNKNOWN]``.
   * - ``+52``
     - ``uint32``
     - Background scans.
     - ``[OBSERVED]`` field and interpretation in acquired variants.
   * - ``+56``
     - ``float32``
     - Background gain where applicable.
     - ``[OBSERVED]`` in the validated acquired family; generality is ``[UNKNOWN]``.
   * - ``+68``
     - ``uint32``
     - Collection duration multiplied by 100.
     - ``[ESTABLISHED]`` for the data header.
   * - ``+80``
     - ``float32``
     - Reference/HeNe-class frequency.
     - ``[ESTABLISHED]`` in validated native cases; distinct from Raman ``+96``.
   * - ``+84``
     - ``float32``
     - Sample-spacing factor.
     - ``[ESTABLISHED]`` in validated IFG cases; used in the native OPD relation.
   * - ``+92``
     - ``float32``
     - Aperture where applicable.
     - ``[OBSERVED]`` for the acquired variant; not universal.
   * - ``+96``
     - ``float32``
     - Raman excitation/laser frequency.
     - ``[ESTABLISHED]`` for the validated Raman variant; not a general IR field.

Other numeric fields in the header are intentionally not assigned meanings by
this reference. In particular, derived transform quantities must not be
mistaken for stored fields.

.. _omnic-header-overreads:

Offsets beyond the 140-byte block
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``[ESTABLISHED]`` The ``0x02`` block is 140 bytes long in every inspected file.
Header-relative offsets from ``+140`` onwards therefore lie outside the block
and read the beginning of whichever block is stored next. ``[OBSERVED]`` In the
SPA corpus, the block stored immediately after ``0x02`` is:

* ``0x80`` (128 zero bytes) in the newer-layout files;
* ``0x6a`` in a minority of files, in which case header-relative ``+188``
  coincides with ``0x6a`` ``+48`` (optical velocity);
* ``0x03`` (the data payload) in the library/retrieved files.

Values read at ``+140..+188`` may therefore look like a copy of ``0x6a``
or be blank, depending on the file; in both cases they belong to the next
block (``0x6a`` or the zero-filled ``0x80``). These offsets are not
``0x02`` fields: their content depends on block placement, which is not a
format guarantee. ``0x6a``, located through its own key record, is the
canonical source of the parameters.


.. _omnic-parameter-block:

Spectrometer and acquisition parameters ``0x6a``
-------------------------------------------------

``[ESTABLISHED]`` The recurring ``0x6a`` block is a 56-byte
spectrometer/acquisition parameter block in the validated variants. The
following relationships are mature enough to document generically:

.. list-table:: Mature ``0x6a`` fields
   :header-rows: 1

   * - Relative offset
     - Meaning
     - Certainty/scope
   * - ``+0..+12``
     - Instrument-family and acquisition codes.
     - ``[OBSERVED]`` values and combinations; individual code semantics remain
       ``[UNKNOWN]``.
   * - ``+16``
     - Digitizer-bit field.
     - ``[ESTABLISHED]`` in the validated acquired variants.
   * - ``+20``
     - High-pass filter.
     - ``[ESTABLISHED]`` where independently matched to acquisition reports.
   * - ``+24``
     - Low-pass filter.
     - ``[ESTABLISHED]`` where independently matched to acquisition reports.
   * - ``+44``
     - Sample gain.
     - ``[ESTABLISHED]`` where independently matched to acquisition reports.
   * - ``+48``
     - Optical velocity.
     - ``[ESTABLISHED]`` in validated acquired/IFG variants.

Values read at header-relative offsets ``+140`` and beyond coincide with
these parameters only when ``0x6a`` happens to be stored right after
``0x02`` (see :ref:`omnic-header-overreads`). A format consumer should locate
``0x6a`` through its key record.


.. _omnic-unit-codes:

Unit codes
----------

X-unit codes
~~~~~~~~~~~~

.. list-table:: Validated X-unit mappings
   :header-rows: 1

   * - Code
     - Meaning
     - Certainty/scope
   * - ``0x01``
     - Wavenumbers, ``cm^-1``.
     - ``[ESTABLISHED]`` in the validated spectral family.
   * - ``0x02``
     - Data points.
     - ``[ESTABLISHED]`` for the validated standalone interferogram context;
       it is not itself a universal interferogram marker without context.
   * - ``0x20``
     - Raman shift, ``cm^-1``.
     - ``[ESTABLISHED]`` for the validated Raman variant.
   * - ``0x03`` / ``0x04``
     - Wavelength, nm / wavelength, micrometres.
     - ``[OBSERVED]`` mappings from the shared OMNIC data header; broader SPA
       coverage is not established here.

Y/data-unit codes
~~~~~~~~~~~~~~~~~

.. list-table:: Validated Y/data-unit mappings
   :header-rows: 1

   * - Code
     - Meaning
     - Certainty/scope
   * - ``0x10``
     - Percent transmittance.
     - ``[ESTABLISHED]`` in the validated library/retrieved variant.
   * - ``0x11``
     - Absorbance.
     - ``[ESTABLISHED]`` in the validated spectral family.
   * - ``0x16``
     - Volts-labelled detector signal.
     - ``[ESTABLISHED]`` for the standalone saved-IFG representation;
       this is a label, not proof of calibrated voltage.
   * - ``0x17``
     - Transmittance.
     - ``[OBSERVED]`` in the shared OMNIC data header.
   * - ``0x1f``
     - Raman intensity.
     - ``[ESTABLISHED]`` for the validated Raman variant.
   * - ``0x0b`` / ``0x0c`` / ``0x0f`` / ``0x14`` / ``0x15`` / ``0x1a``
     - Reflectance, log(1/R), single beam, Kubelka--Munk, reflectance, and
       photoacoustic mappings respectively.
     - ``[OBSERVED]`` shared-header mappings; applicability and exact physical
       calibration are variant-dependent.


.. _omnic-experiment-information:

Experiment Information ``0x82``
-------------------------------

Key ``0x82`` identifies Experiment Information blocks. Native subtype ``0x79``
has a fixed-anchor layout supported by multiple independent native blocks:

.. list-table:: Native subtype ``0x79`` anchors
   :header-rows: 1

   * - Relative anchor
     - Generic content
     - Certainty
   * - ``+10``
     - Experiment path/file text begins.
     - ``[ESTABLISHED]``
   * - ``+90``
     - Experiment title/name text.
     - ``[ESTABLISHED]``
   * - ``+154``
     - Descriptive or custom text.
     - ``[ESTABLISHED]``
   * - ``+413``
     - Accessory-related text anchor.
     - ``[ESTABLISHED]``
   * - ``+670``
     - Duplicate or prefixed path-like text.
     - ``[OBSERVED]``; exact role remains unresolved.

The strings are NUL-terminated and padded between the observed anchors.
``+413..+670`` also contains unresolved numeric or padding content.

``[OBSERVED]`` Other subtypes occur in ``0x82`` blocks: ``0x9e`` (the most
frequent, usually alongside a ``0x79`` block), ``0xa6`` (a small block), and,
rarely, ``0x9d``. They are structurally recognized but their semantics are
``[UNKNOWN]``; none of them is mandatory before a subtype ``0x79`` block.
The subtypes must not be collapsed into one universal record layout.

.. _omnic-auxiliary-blocks:

Auxiliary block ``0x69`` and history text ``0x1b``
--------------------------------------------------

``[ESTABLISHED]`` ``0x69`` is a recurring 12-byte block whose boundaries are
known in the acquired variants; its semantics are ``[UNKNOWN]``.

``[ESTABLISHED]`` ``0x1b`` references a processing/acquisition history text
block where present. Its content, length and presence vary with the save
operation and the processing history; no stable formal grammar is claimed.
