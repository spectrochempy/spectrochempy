.. _fileformats:

File format references
======================

This section contains technical references for binary and proprietary
file formats supported by SpectroChemPy.

These references describe independently reconstructed format knowledge
where official specifications are unavailable, and may distinguish
established facts, observations, hypotheses, and unresolved fields. They
describe the formats themselves and are intended to be usable by any
developer, independently of SpectroChemPy.

.. _fileformats-provenance:

Provenance
----------

The format references are based on independent observations of lawfully
obtained files and are provided solely to support interoperability and the
preservation of scientific data. They do not constitute official
specifications from the respective vendors. No proprietary source code or
object code has been incorporated into this documentation.

Because the formats are undocumented and the analysis is based on a limited
set of files, some interpretations may be incomplete or inaccurate. The
descriptions are generic: they do not depend on particular proprietary files
or on the availability of vendor-distributed examples.

Contributors with access to other variants of a documented format are
encouraged to report differences through the
`SpectroChemPy issue tracker <https://github.com/spectrochempy/spectrochempy/issues>`_
so that these references can be improved.

.. _fileformats-evidence:

Evidence and certainty levels
-----------------------------

Every statement of evidence in the format references is tagged with a
certainty level:

* ``[ESTABLISHED]`` — supported strongly enough within the current evidence
  corpus to be relied upon, through at least one of the methods below
  applied to files of the format concerned: repeated binary reconstruction
  across multiple files, comparison with the vendor software, a controlled
  acquisition, or an independent recomputation.
* ``[OBSERVED]`` — reproducibly seen in the files examined, but not
  established as a universal invariant across all producers and versions.
* ``[HYPOTHESIS]`` — a plausible semantic interpretation of a reproducible
  binary structure, not demonstrated.
* ``[UNKNOWN]`` — a field or structure whose position, type, or presence may
  be known but whose meaning remains unclear.

Structural certainty is reported separately from semantic certainty where
needed: a block may have an ``[ESTABLISHED]`` boundary or size while its
meaning is only ``[HYPOTHESIS]``, and a field may have ``[OBSERVED]`` value
patterns with ``[UNKNOWN]`` semantics.

Evidence methods
~~~~~~~~~~~~~~~~

The certainty levels rely on the following methods:

* **binary inspection** — repeated decoding of the same structure across
  several files, with consistent boundaries, sizes, and value patterns;
* **comparison with the vendor software** (also called an *oracle*) —
  agreement between a decoded value and the value displayed or exported by
  the vendor software for the same file (for example, a series member
  exported by the vendor software as a standalone spectrum file);
* **controlled acquisition** — files produced with known acquisition or
  processing parameters, so that a decoded field can be matched to a
  deliberately chosen value;
* **independent recomputation** — a stored quantity reproduced from other
  stored data by an independent calculation (for example, a Fourier
  transform of a stored interferogram, an integrated area, or an exact
  arithmetic relationship between fields).

Claims shared between formats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Related formats from the same vendor often share structures. When a
reference reuses a structure or an interpretation documented for a sibling
format without verifying it on files of its own format, the claim is stated
as a working assumption and never carries a stronger certainty level than in
the source reference.

.. toctree::
   :maxdepth: 2

   omnic/index
