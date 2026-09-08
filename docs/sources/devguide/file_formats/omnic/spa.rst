.. _spa-format:

OMNIC SPA file format
=====================

.. note::

   This is an unofficial interoperability reference for the OMNIC ``.spa``
   format. It is based on independent controlled binary and oracle analysis
   across several observed SPA variants. OMNIC producers and versions may use
   different subsets of these structures. The certainty tags below describe
   the strength and scope of each claim; they do not turn this page into an
   official vendor specification. See :ref:`omnic-file-formats` for
   provenance and the definitions of the certainty levels.

Evidence and scope
------------------

The reference separates structural evidence from semantic interpretation:

* ``[ESTABLISHED]`` means that a structure or relationship is supported by
  independent binary/oracle evidence, controlled behavior, or exact
  arithmetic in multiple native cases.
* ``[OBSERVED]`` means that a reproducible structure or correlation was seen
  in the analyzed variants but is not established as universal.
* ``[HYPOTHESIS]`` means that a semantic interpretation is plausible but not
  demonstrated.
* ``[UNKNOWN]`` means that the position or structure may be known while its
  meaning remains unresolved.

The evidence covers ordinary acquired spectra, processed acquired spectra,
library/retrieved spectra, acquired spectra with paired interferograms,
standalone saved interferograms, a Raman variant, Experiment-Information-
bearing variants, and a newer-layout variant. These descriptions are generic;
they do not depend on particular proprietary files or on the availability of
OMNIC-distributed examples.

Overall file organization
-------------------------

An SPA file is organized as a fixed file header followed by a counted table of
16-byte key records. The records point to blocks elsewhere in the file; the
blocks are not required to be contiguous.

.. code-block:: text

   fixed file region
   ├── signature at file offset 0
   ├── saved name/title text at file offset 30
   ├── nlines (number of key records) at file offset 294
   ├── raw acquisition-time value at file offset 296
   └── key table beginning at file offset 304

   referenced blocks
   ├── 0x02  general/spectral header
   ├── 0x03  primary data payload
   ├── 0x04  comments or user text
   ├── 0x1b  processing/history text
   └── other variant-dependent blocks

``[ESTABLISHED]`` The signature begins with the ASCII text ``Spectral Data
File`` for the SPA/SPG family. The saved name/title field begins at offset 30,
``nlines`` is a little-endian ``uint16`` at offset 294, the raw timestamp is at
offset 296, and the first key record is at offset 304.

The remaining fixed-header bytes are not assigned meanings here unless stated
below. In particular, a recognizable file signature does not imply that every
producer uses the same complete header map.

Key-record table
----------------

``[ESTABLISHED]`` Each key record occupies 16 bytes in the observed SPA
families. The generic record layout is:

.. code-block:: text

   relative offset   type       role
   +0                uint8      key
   +1                uint8      reserved/variant-dependent byte
   +2                uint32     referenced block position
   +6                uint32     referenced block length
   +10..+15          bytes      trailing/variant-dependent data

Positions and lengths are file-relative for the referenced blocks. ``nlines``
counts key records, not every byte between the table and the first payload.

``[OBSERVED]`` Ordinary/acquired families use a ``0x00`` terminator slot and
zero padding after the active records. Observed library-derived variants use a
grid of 16-byte-stride ``0x01`` slots before the first main block. The grid's
boundaries and structural role are known in those variants, but its slot
semantics are ``[UNKNOWN]``. Terminator and following-region details remain
variant-dependent; no universal interpretation is assigned to every trailing
record byte.

Recognized key associations
---------------------------

The following table distinguishes an established association from a universal
semantic guarantee. A key identifies a block in the observed variants; it does
not by itself determine the signal interpretation of that block.

.. list-table:: SPA key associations
   :header-rows: 1

   * - Key
     - Generic association
     - Certainty and scope
   * - ``0x02``
     - General/spectral header block.
     - ``[ESTABLISHED]`` association; field applicability remains variant-dependent.
   * - ``0x03``
     - Primary data payload.
     - ``[ESTABLISHED]`` association. It may contain a wavenumber spectrum or a
       standalone interferogram; header/unit context determines the signal type.
   * - ``0x04``
     - Comment or user-text block.
     - ``[ESTABLISHED]`` text association where present; complete text conventions
       are variant-dependent.
   * - ``0x1b``
     - Processing/acquisition history text.
     - ``[ESTABLISHED]`` association where present; history content and presence
       vary by save operation.
   * - ``0x53``
     - Library/retrieval-associated saved-spectrum text.
     - ``[OBSERVED]`` association; full structure and semantics are ``[UNKNOWN]``.
   * - ``0x64`` / ``0x65``
     - Companion interferogram-related blocks.
     - ``[OBSERVED]`` in the acquired-pair variant; general semantics remain
       ``[UNKNOWN]``.
   * - ``0x66``
     - Sample interferogram in a validated acquired-pair variant.
     - ``[ESTABLISHED]`` for that variant through independent Fourier-transform
       reconstruction; not a universal SPA rule.
   * - ``0x67``
     - Background interferogram in the same validated acquired-pair variant.
     - ``[ESTABLISHED]`` for that variant through independent Fourier-transform
       reconstruction; standalone saved interferograms may use ``0x03`` instead.
   * - ``0x69``
     - Recurring 12-byte auxiliary block outside the observed library variant.
     - Block boundaries are ``[ESTABLISHED]``; semantics are ``[UNKNOWN]``.
   * - ``0x6a``
     - Spectrometer/acquisition parameter block.
     - ``[ESTABLISHED]`` block association and several field relationships;
       some family-code semantics remain ``[UNKNOWN]``.
   * - ``0x80``
     - 128-byte all-zero block in one newer-layout/writer-associated family.
     - ``[OBSERVED]`` correlation only. It is not established as a writer
       generation marker; its semantic role is ``[UNKNOWN]``.
   * - ``0x82``
     - Experiment Information block.
     - ``[ESTABLISHED]`` association. Subtype ``0x79`` has a supported native
       layout; subtype ``0x9d`` remains ``[UNKNOWN]`` and is not mandatory
       before every ``0x79`` occurrence.
   * - ``0x92``
     - Custom-information association.
     - ``[OBSERVED]`` association in some variants; structure and semantics are
       ``[UNKNOWN]``.

The presence or absence of a key is itself variant-dependent. In particular,
``0x03`` must not be described universally as ``the spectrum``: its X-unit
context distinguishes a spectral payload from an explicit data-points
interferogram in the validated examples.

General ``0x02`` header
------------------------

The following offsets are relative to the block referenced by ``0x02``. The
table deliberately separates mature meanings from observations and unresolved
fields.

.. list-table:: ``0x02`` header fields
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
     - ``[ESTABLISHED]`` for the validated spectral family; see orientation below.
   * - ``+20``
     - ``float32``
     - OMNIC ``First X``.
     - ``[ESTABLISHED]`` for the validated spectral family; see orientation below.
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
     - ``[ESTABLISHED]`` for the general header family.
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
   * - ``+140..+188``
     - mixed
     - Mirror of the beginning of ``0x6a`` in some variants.
     - ``[OBSERVED]`` byte-for-byte mirror in some files; blank in another
       newer-layout family. Use ``0x6a`` as the canonical parameter source.
   * - ``+188``
     - ``float32``
     - Optical velocity mirror.
     - ``[ESTABLISHED]`` as a mirror relationship where populated; applicability
       is variant-dependent.

Other numeric fields in the header are intentionally not assigned meanings by
this reference. In particular, derived transform quantities must not be
mistaken for stored fields.

Spectrometer and acquisition parameters (``0x6a``)
---------------------------------------------------

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

The ``+140..+188`` header region can mirror these parameters, but the mirror is
blank in an observed newer-layout family. A reader or format consumer should
therefore treat ``0x6a`` as the canonical parameter block and the header mirror
as variant-dependent.

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
     - ``[OBSERVED]`` mappings from the shared OMNIC header family; broader SPA
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
     - ``[OBSERVED]`` in the shared OMNIC header family.
   * - ``0x1f``
     - Raman intensity.
     - ``[ESTABLISHED]`` for the validated Raman variant.
   * - ``0x0b`` / ``0x0c`` / ``0x0f`` / ``0x14`` / ``0x15`` / ``0x1a``
     - Reflectance, log(1/R), single beam, Kubelka--Munk, reflectance, and
       photoacoustic mappings respectively.
     - ``[OBSERVED]`` shared-header mappings; applicability and exact physical
       calibration are variant-dependent.

Spectrum payload and orientation
--------------------------------

The primary payload is referenced by key ``0x03``. ``[ESTABLISHED]`` in the
validated spectral family, it consists of little-endian ``float32`` intensity
values whose count agrees with ``+4`` and whose X coordinate is linear between
the two header endpoints.

For the validated spectral family:

* header ``+16`` is OMNIC **Last X**;
* header ``+20`` is OMNIC **First X**;
* the native payload is stored in descending wavenumber order;
* payload element 0 corresponds to ``+16`` and the final element corresponds
  to ``+20``.

The historical internal reader names ``firstx`` and ``lastx`` describe storage
order rather than OMNIC's display terminology. This orientation is not
independently generalized here to every library/retrieved variant.

The key association ``0x03`` does not by itself imply a spectrum. A header with
X-unit code ``0x02`` can use the same key for a standalone saved interferogram
with a data-points X coordinate.

Interferograms
--------------

In the validated acquired-pair variant, ``0x66`` is the sample interferogram
and ``0x67`` is the background interferogram. ``[ESTABLISHED]`` This identity
was supported by independent Fourier reconstruction, not merely by the
reader's parameter names. It is scoped to that acquired-pair variant: a
standalone saved interferogram may place its primary data in ``0x03`` instead.

The native IFG payload is stored as ``float32`` values. A standalone saved IFG
has an OMNIC Volts label, but ``[UNKNOWN]`` whether this implies a calibrated
absolute voltage scale. Do not generalize that label automatically to every
``0x66``/``0x67`` block, and do not infer a universal amplitude normalization
from the stored values.

Peak and transform geometry
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the validated native IFG cases:

* ``+32`` is a ``uint32`` and ``+40`` is a ``float32``;
* both match OMNIC's interferogram peak position;
* both match the raw signal ``argmax``;
* this establishes the peak/centerburst index relationship, but not a formal
  physical zero-path-difference (ZPD) definition.

``[ESTABLISHED]`` In the same native cases:

.. code-block:: text

   +48 = N_stored - P

where ``N_stored`` is the native stored IFG payload length and ``P`` is the
peak position. The exact OMNIC semantic name or counting rationale for ``+48``
remains ``[UNKNOWN]``. Keep the following quantities distinct:

* native stored IFG length ``N_stored``;
* peak position ``P``;
* stored field ``+48``;
* derived transform base ``N_hat = 2 * (+48)``;
* FFT length ``+44``.

``N_hat`` is a derived quantity, not a stored native field and not a universal
name for a native OMNIC quantity.

Zero filling
~~~~~~~~~~~~

Where the validated transform relationship applies:

.. code-block:: text

   FFT = N_hat * 2**zero_fill_level
   N_hat = 2 * (+48)       (derived)

This is an arithmetic relationship between stored geometry and FFT length. The
zero-filling level is directly supported for some acquired cases and inferred
from the arithmetic in another native IFG case; it must not be treated as a
universal interpretation of every SPA header.

Native IFG coordinate sampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``[ESTABLISHED]`` in the validated native IFG cases, the physical optical-path
difference step is:

.. code-block:: text

   Delta_OPD = sample_spacing / (2 * reference_frequency)

where ``reference_frequency`` is header ``+80`` and ``sample_spacing`` is
header ``+84``. The peak/argmax supplies the origin used by the validated
reconstruction. Formal physical ZPD semantics remain ``[UNKNOWN]``.

This relationship is now implemented by the public reader for the corrected
sample-spacing path, but the format statement is based on native evidence
rather than on reader behavior alone.

Acquisition time and timezone
-----------------------------

For ordinary acquired SPA variants, ``[ESTABLISHED]`` the raw little-endian
``uint32`` at file-relative offset ``+296`` represents:

.. code-block:: text

   1899-12-31 00:00:00 UTC + raw seconds

The result is an absolute UTC acquisition instant. A 32-bit rollover is
observed and must be handled when interpreting values near the counter limit.

The displayed GMT/local offset is not an intrinsic SPA timezone field in the
validated files. ``[ESTABLISHED]`` OMNIC renders the stored absolute instant
using the timezone configuration and rules of the Windows system displaying
the file; changing that viewing-system timezone changes the displayed offset
without changing the SPA bytes.

This interpretation is variant-dependent. Library/retrieved variants can use
``+296`` for a counter or default-like value that is not a valid acquisition
date. A reader must not promote the field to ``acquisition_date`` without
checking the variant.

Experiment Information
----------------------

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
     - ``[ESTABLISHED]``; the historical ``+254`` comment is contradicted.
   * - ``+413``
     - Accessory-related text anchor.
     - ``[ESTABLISHED]``
   * - ``+670``
     - Duplicate or prefixed path-like text.
     - ``[OBSERVED]``; exact role remains unresolved.

The strings are NUL-terminated and padded between the observed anchors.
``+413..+670`` also contains unresolved numeric or padding content. Subtype
``0x9d`` is structurally recognized but its semantics are ``[UNKNOWN]``; it is
not mandatory before every subtype ``0x79`` block. The two subtypes must not be
collapsed into one universal record layout.

The current public reader's sequential decoder is an implementation detail and
does not define this fixed-slot native layout. A future reader audit may map
these anchors into metadata, but no reader change is part of this reference.

Raman variant
-------------

The validated Raman variant uses:

* X-unit code ``0x20`` for Raman shift in ``cm^-1``;
* Y/data-unit code ``0x1f`` for Raman intensity;
* header ``+80`` for the reference/HeNe-class frequency;
* header ``+96`` for the Raman excitation/laser frequency.

``[ESTABLISHED]`` These are distinct physical quantities. The Raman ``+96``
interpretation is scoped to the validated Raman variant and should not be
promoted to a general-purpose field for all SPA files. A reader metadata issue
must not be used to claim that the stored Raman X axis is shifted or otherwise
incorrect.

Library/retrieved variants
--------------------------

Library/retrieved spectra retain the key-table pointer model but can differ
structurally from ordinary acquired spectra:

* the ``0x02`` block can occur at a different physical file location while
  retaining the relevant relative header fields;
* acquisition-only fields can be zero, default-like, or not applicable;
* the raw ``+296`` value must not automatically become an acquisition date;
* a 16-byte-stride ``0x01`` grid can occur before the first main block;
* ``0x53`` is associated with saved-spectrum/library text;
* the primary payload remains located through the key table.

The grid boundaries and structural role are ``[OBSERVED]`` in these variants,
but its slot semantics are ``[UNKNOWN]``. The full semantics of ``0x53`` and
the library-specific timestamp are also ``[UNKNOWN]``. No complete library
format is claimed here.

Other unresolved structures and limitations
--------------------------------------------

The following structures are useful format-research landmarks but should not
be assigned unsupported meanings:

* ``0x69`` — recurring 12-byte auxiliary block; semantics ``[UNKNOWN]``.
* ``0x80`` — 128-byte all-zero block associated with an observed newer-layout
  family; writer correlation is ``[OBSERVED]`` and the role is ``[UNKNOWN]``.
* ``0x9d`` — recognized Experiment Information subtype; semantics ``[UNKNOWN]``.
* ``0x01`` grid — observed library/retrieved layout and boundaries; slot
  semantics ``[UNKNOWN]``.
* ``0x53`` — saved-spectrum/library-associated text; complete encoding and
  semantics ``[UNKNOWN]``.
* instrument-family code meanings within ``0x6a``;
* the formal physical meaning of ZPD and the exact native semantic name of
  ``+48``;
* general absolute IFG scaling and calibrated-voltage semantics;
* generality of paired-IFG identities beyond the validated acquired-pair
  variant;
* variant-specific timestamp meanings and Experiment Information tail fields.

Implementation notes and limitations
------------------------------------

This page describes format evidence, not a complete critique of
``read_omnic.py``. The current implementation may not expose every established
field, may use variant-specific fallbacks, and may still need a separate
field-by-field audit against this reference. In particular:

* the corrected ``+84`` sample-spacing relation is implemented in the public
  reader;
* native subtype-``0x79`` Experiment Information is fixed-slot, whereas the
  current decoder uses a sequential strategy;
* some variant-specific metadata, including the Raman distinction between
  ``+80`` and ``+96``, may not yet be surfaced with native semantics.

No production reader behavior is changed by this document. The public reader,
tests, and this reference should be reconciled in a separate reader-audit and
implementation sequence.
