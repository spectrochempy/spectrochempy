.. _omnic-file-formats:

OMNIC file formats
==================

This documentation is based on independent observations of lawfully obtained
files and is provided solely to support interoperability and the preservation
of scientific data. It does not constitute an official specification from
Thermo Fisher Scientific. OMNIC is a trademark of its respective owner. No
proprietary source code or object code has been incorporated into this
documentation.

The information presented here was obtained through independent analysis of
the binary structure and observable behavior of OMNIC files. Because the
formats are undocumented and the analysis is based on a limited set of files,
some interpretations may be incomplete or inaccurate.

Evidence and certainty levels
-----------------------------

Every statement of evidence in the format references below is tagged with a
certainty level:

* ``[ESTABLISHED]`` — confirmed strongly enough within the current evidence
  corpus to be relied upon structurally, through repeated binary
  reconstruction, multiple files, or independent validation.
* ``[OBSERVED]`` — reproducibly seen in the files examined, but not
  established as a universal invariant across all OMNIC versions/producers.
* ``[HYPOTHESIS]`` — a plausible semantic interpretation of a reproducible
  binary structure.
* ``[UNKNOWN]`` — a field or structure whose position/type/presence may be
  known but whose meaning remains unclear.

Structural certainty is reported separately from semantic certainty where
needed: a block may have an ``[ESTABLISHED]`` boundary or size while its
meaning is only ``[HYPOTHESIS]``, and a field may have ``[OBSERVED]`` value
patterns with ``[UNKNOWN]`` semantics.

.. toctree::
   :maxdepth: 2

   srs
