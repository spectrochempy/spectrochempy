.. _official-plugin-architecture:

===================================
Official plugin architecture notes
===================================

SpectroChemPy's plugin architecture is still stabilizing. The current system is
intended to support both official project-maintained plugins and third-party
extensions, while preserving a simple public API for users.

Current direction
=================

The project may progressively distinguish:

* a technical core containing shared infrastructure such as ``NDDataset``,
  coordinates, units, plotting, generic processing foundations, and plugin
  registration;
* official domain plugins for coordinated scientific domains such as IR, NMR,
  Raman, IRIS, and simulation;
* advanced or specialized plugins for heavier optional workflows.

This is an architectural direction rather than a completed migration plan.
APIs, packaging details, and plugin boundaries may still evolve.

Official plugins
================

Official plugins should:

* expose modern APIs through namespaces such as ``scp.nmr`` or ``scp.iris``;
* keep dataset accessors for operations that use an existing dataset;
* avoid importing heavy optional dependencies at ``import spectrochempy`` time;
* use the public plugin API from ``spectrochempy.api.plugins``;
* keep examples in the central gallery, clearly marked when a plugin is
  required.

Limited root-level compatibility aliases may be provided for official plugins
when they preserve important existing user workflows. These aliases should be
explicit, documented, and preferably emit a deprecation or future warning when
appropriate. Reader functions such as ``scp.read_*`` may remain top-level when
that is the established and clearest public API.

Future IR plugin
================

Infrared support remains central to SpectroChemPy. A future official IR plugin
could provide readers such as OMNIC and OPUS internally while still being part
of the standard installation experience. Such a change would be an internal
modularization step, not a requirement for infrared users to manually load a
plugin.

The project should avoid moving established infrared workflows out of the
standard user experience until packaging, documentation, and compatibility
layers are ready.

