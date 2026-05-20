.. _plugin-roadmap:

============================
Modular architecture roadmap
============================

SpectroChemPy may progressively evolve toward a more modular architecture. This
page describes a possible future direction; it is not a finalized migration
plan, and it does not change the current installation or public API.

The immediate goal is to keep the standard user experience simple while making
the internal architecture easier to maintain.

Why modularity?
===============

A modular architecture can help SpectroChemPy:

* keep a lighter technical core;
* isolate optional or heavy dependencies;
* let domain-specific features evolve at their own pace;
* make official extensions easier to test, publish, and maintain;
* support third-party extensions without mixing them into the core package.

This should not make everyday usage more complicated. Standard workflows such
as reading common infrared files or running established chemometric analyses
should remain natural for users. In particular, modularity is intended to make
the ecosystem easier to organize and maintain while keeping familiar workflows
available through the standard SpectroChemPy experience.

Conceptual categories
=====================

Future work may distinguish several categories.

Core infrastructure
-------------------

The technical core is expected to contain foundations shared by all domains,
for example:

* ``NDDataset``;
* coordinates;
* units;
* plotting;
* generic processing foundations;
* plugin discovery and registration infrastructure.

Official domain plugins
-----------------------

Official domain plugins are project-maintained extensions for scientific
domains or coordinated workflows. Candidate domains include:

* IR;
* NMR;
* Raman;
* IRIS;
* Cantera / simulation.

These domains may progressively be organized as plugins, but they can still be
part of the standard SpectroChemPy experience.

Advanced and specialized plugins
--------------------------------

Some workflows may be better suited as optional specialized plugins, especially
when they require heavy dependencies, specialized external software, GPU
backends, or narrow domain assumptions.

Standard distribution and minimal core
======================================

One possible future distribution model is:

* a minimal technical package with the core infrastructure;
* a richer default SpectroChemPy installation that includes official plugins
  needed for common workflows.

No package split is implemented here. The current package name and installation
behavior remain unchanged.

In such a future, "installed" would simply mean that the relevant official
plugin package is present in the environment. "Discovered" would mean
SpectroChemPy has found it through entry points. "Lazy-loaded" would mean the
implementation is imported only when a user actually accesses that feature.

Infrared workflows
==================

Infrared users remain the primary target audience for the standard
SpectroChemPy experience.

Readers such as ``scp.read_omnic()`` and ``scp.read_opus()`` and common
infrared workflows should remain naturally available in standard
installations. A future official IR plugin could internally provide these
capabilities while still being installed automatically in the default
SpectroChemPy distribution.

In that scenario, pluginization would primarily be an internal modularization
strategy. Infrared users should not have to manage extra plugin loading steps
for normal usage.

Official and third-party plugins
================================

Official plugins are maintained with the SpectroChemPy project and may receive
coordinated documentation, tests, examples, and compatibility aliases.

Third-party plugins use the same plugin mechanism but are not listed as built-in
knowledge in the core package. They should declare their own contributions
through the plugin entry point system.

Examples
========

The gallery should remain centralized and organized scientifically. Examples
that require a plugin should be marked with the required plugin, while
beginner-oriented pages and quickstart material should continue to work with
the standard SpectroChemPy environment.
