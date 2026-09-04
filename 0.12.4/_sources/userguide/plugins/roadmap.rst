.. _plugin-roadmap:

==============
Plugin roadmap
==============

SpectroChemPy is moving selected optional and domain-specific features into
official plugins. For users, the goal is to keep workflows natural:

* install the plugin for the domain you need;
* use the documented namespace, such as ``scp.nmr`` or ``scp.iris``;
* keep core-only scripts free from optional plugin dependencies;
* find plugin examples in the gallery with their requirements stated clearly.

Official plugins may evolve faster than the core package when they depend on
specialized libraries or domain-specific conventions. Compatibility aliases may
exist during transitions, but new examples should use plugin namespaces.

This page describes the user-facing direction only. It is not the maintainer
architecture roadmap.

The detailed architecture roadmap for maintainers and plugin authors lives in
:ref:`plugin-dev-roadmap`.
