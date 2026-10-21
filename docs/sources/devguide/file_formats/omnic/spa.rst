.. _spa-format:

OMNIC SPA file format
=====================

.. note::

   This is an unofficial interoperability reference for the OMNIC ``.spa``
   format. It is based on independent controlled binary and oracle analysis
   across several observed SPA variants. OMNIC producers and versions may use
   different subsets of these structures. The certainty tags below describe
   the strength and scope of each claim; they do not turn this page into an
   official vendor specification. See :ref:`fileformats-provenance` for
   provenance and :ref:`fileformats-evidence` for the definitions of the
   certainty levels and evidence methods.

Evidence and scope
------------------

The certainty levels and evidence methods are defined in
:ref:`fileformats-evidence`. The evidence covers ordinary acquired spectra,
processed acquired spectra, library/retrieved spectra, acquired spectra with
paired interferograms, standalone saved interferograms, a Raman variant,
Experiment-Information-bearing variants, and a newer-layout variant.

.. list-table:: SPA evidence corpus (distinct files)
   :header-rows: 1

   * - Variant
     - Files
     - Methods
   * - Spectra (acquired or processed)
     - 5849
     - Binary inspection; comparison with the vendor software; controlled
       acquisition (timestamp and timezone)
   * - Acquired spectra with paired interferograms (``0x66``/``0x67``)
     - 81
     - Binary inspection; independent recomputation (Fourier transform)
   * - Library/retrieved spectra
     - 5
     - Binary inspection
   * - Standalone saved interferogram
     - 1
     - Binary inspection; comparison with the vendor software; independent
       recomputation
   * - Raman spectrum
     - 1
     - Binary inspection; comparison with the vendor software

Across these 5937 files, 5904 use the newer layout with a ``0x80`` block and
91 carry subtype-``0x79`` Experiment Information.

Overall file organization
-------------------------

An SPA file follows the common OMNIC container model: a fixed file header,
a counted table of 16-byte key records, and the blocks they reference (see
:ref:`omnic-file-header` and :ref:`omnic-key-table`). The shared block
layouts (data header ``0x02``, parameter block ``0x6a``, unit codes,
Experiment Information ``0x82``) are documented in :ref:`omnic-common`.

.. code-block:: text

   fixed file region
   ├── signature at file offset 0
   ├── saved name/title text at file offset 30
   ├── nlines (number of key records) at file offset 294
   ├── raw acquisition-time value at file offset 296
   └── key table beginning at file offset 304

   referenced blocks
   ├── 0x02  data header
   ├── 0x03  primary data payload
   ├── 0x04  comments or user text
   ├── 0x1b  processing/history text
   └── other variant-dependent blocks

In SPA files, offset 296 holds the acquisition timestamp of ordinary
acquired spectra (see :ref:`spa-acquisition-time`).

Key-table layout in SPA files
-----------------------------

The key records follow the common layout (:ref:`omnic-key-table`).
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
     - Data header (description of the data payload and acquisition
       settings).
     - ``[ESTABLISHED]`` association (see :ref:`omnic-data-header`);
       field applicability remains variant-dependent.
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
     - ``[ESTABLISHED]`` block association and several field relationships
       (see :ref:`omnic-parameter-block`); some family-code semantics remain
       ``[UNKNOWN]``.
   * - ``0x80``
     - 128-byte all-zero block in one newer-layout/writer-associated family.
     - ``[OBSERVED]`` correlation only. It is not established as a writer
       generation marker; its semantic role is ``[UNKNOWN]``.
   * - ``0x82``
     - Experiment Information block.
     - ``[ESTABLISHED]`` association. Subtype ``0x79`` has a supported native
       layout; subtypes ``0x9e``, ``0x9d`` and ``0xa6`` remain ``[UNKNOWN]``
       (see :ref:`omnic-experiment-information`).
   * - ``0x92``
     - Custom-information association.
     - ``[OBSERVED]`` association in some variants; structure and semantics are
       ``[UNKNOWN]``.

The presence or absence of a key is itself variant-dependent. In particular,
``0x03`` must not be described universally as ``the spectrum``: its X-unit
context distinguishes a spectral payload from an explicit data-points
interferogram in the validated examples.

.. _spa-payload-orientation:

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

This orientation is not independently generalized here to every
library/retrieved variant.

The key association ``0x03`` does not by itself imply a spectrum. A header with
X-unit code ``0x02`` can use the same key for a standalone saved interferogram
with a data-points X coordinate.

Interferograms
--------------

In the validated acquired-pair variant, ``0x66`` is the sample interferogram
and ``0x67`` is the background interferogram. ``[ESTABLISHED]`` This identity
was supported by independent Fourier reconstruction. It is scoped to that
acquired-pair variant: a standalone saved interferogram may place its primary
data in ``0x03`` instead.

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

.. _spa-acquisition-time:

Acquisition time and timezone
-----------------------------

For ordinary acquired SPA variants, ``[ESTABLISHED]`` the ``uint32`` at
file offset 296 is the acquisition instant, in the OMNIC time encoding
(see :ref:`omnic-timestamps`). A 32-bit rollover is
observed and must be handled when interpreting values near the counter limit.

The displayed GMT/local offset is not an intrinsic SPA timezone field in the
validated files. ``[ESTABLISHED]`` OMNIC renders the stored absolute instant
using the timezone configuration and rules of the Windows system displaying
the file; changing that viewing-system timezone changes the displayed offset
without changing the SPA bytes.

This interpretation is variant-dependent. Library/retrieved variants can use
offset 296 for a counter or default-like value that is not a valid acquisition
date. A format consumer must not interpret the field as an acquisition date
without checking the variant.

Raman variant
-------------

The validated Raman variant uses:

* X-unit code ``0x20`` for Raman shift in ``cm^-1``;
* Y/data-unit code ``0x1f`` for Raman intensity;
* header ``+80`` for the reference/HeNe-class frequency;
* header ``+96`` for the Raman excitation/laser frequency.

``[ESTABLISHED]`` These are distinct physical quantities. The Raman ``+96``
interpretation is scoped to the validated Raman variant and should not be
promoted to a general-purpose field for all SPA files.

Library/retrieved variants
--------------------------

Library/retrieved spectra retain the key-table pointer model but can differ
structurally from ordinary acquired spectra:

* the ``0x02`` block can occur at a different physical file location while
  retaining the relevant relative header fields;
* acquisition-only fields can be zero, default-like, or not applicable;
* the raw offset-296 value must not automatically become an acquisition date;
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
* ``0x9e``, ``0xa6``, ``0x9d`` — recognized Experiment Information subtypes;
  semantics ``[UNKNOWN]``.
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
