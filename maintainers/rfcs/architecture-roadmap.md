# SpectroChemPy — Architecture Roadmap

**Status:** Maintainer discussion document

## Purpose

This document records the current understanding of SpectroChemPy architectural priorities following the CoordSet Storage Redesign and the subsequent architecture reviews.

It is intentionally lightweight.

It is not a formal RFC.

It is not a commitment.

It is a living document intended to:

* preserve architectural context;
* record major conclusions;
* avoid repeating completed investigations;
* help maintainers prioritize future work.

This document should evolve as the project evolves.

---

# Decision Log

## 2026-06

### CoordSet Storage Redesign

Decision:

* redesign considered complete;
* `_storage` is the runtime coordinate container;
* lifecycle behavior is clarified;
* same-dim semantics are stabilized;
* serialization compatibility is preserved.

Future work is limited to maintenance and follow-up improvements.

---

### Metadata Propagation

Decision:

* semantic fragmentation was identified as the primary issue;
* Metadata Contract RFC was drafted;
* behavior characterization was completed.

No immediate redesign work is planned.

Future work should be driven by concrete metadata issues.

---

### Coordinate Arithmetic

Decision:

* current arithmetic semantics were reviewed;
* no compelling reason was found to move toward full coordinate-aware alignment;
* current NumPy-style broadcasting with spectroscopy-oriented coordinate validation remains acceptable.

Further redesign is deferred.

---

### Dataset-vs-Coord Arithmetic

Decision:

* Coord is considered an axis/support object;
* NDDataset is considered a signal/data object;
* arithmetic between Coord and NDDataset was classified as a semantic bug;
* support was removed.

Issue closed unless regressions appear.

---

### NDMath

Decision:

* semantics appear substantially more coherent than initially expected;
* test coverage is stronger than expected;
* no urgent refactoring need has been identified.

Future work should focus on maintainability rather than semantics.

NDMath refactoring is deferred until a concrete maintenance problem appears.

---

# Recently Completed Topics

## CoordSet Storage Redesign

Status: Completed

Major outcomes:

* runtime storage unification;
* lifecycle clarification;
* removal of legacy storage assumptions;
* stabilization of same-dim behavior;
* improved architectural consistency.

---

## Metadata Contract

Status: Drafted

Major outcomes:

* metadata propagation semantics documented;
* preservation and recomputation rules clarified;
* architectural direction established.

Current priority: Low.

---

## Coordinate Arithmetic Review

Status: Completed

Major outcomes:

* current behavior documented;
* design space explored;
* future options identified.

Current priority: Low.

---

## Dataset-vs-Coord Semantics

Status: Completed

Major outcomes:

* object-model ambiguity identified;
* semantic bug corrected;
* object responsibilities clarified.

Current priority: Closed.

---

## Display Architecture (#843)

Status: Completed

Refer to [`maintainers/display-architecture.md`](../display-architecture.md)
for the final architecture.

Key outcomes: Coord, CoordSet, NDDataset, and Project now use the
semantic HTML path (`_repr_sections()` → `_render_sections()`). The
authoritative reference is `display-architecture.md`.

---

# Near-Term Architectural Candidates

The following topics appear to offer the highest architectural value relative to their expected complexity.

---

## Project Object Review

Motivation:

The long-term role of `Project` remains less clearly defined than other core objects.

Questions include:

* object responsibilities;
* persistence model;
* relationship with datasets;
* relationship with workflows;
* interaction with future ecosystem developments.

This topic likely requires an architectural audit before any implementation work.

---

## User-Facing Improvements

Architectural work should not prevent progress on user-facing capabilities.

Examples include:

* OPUS improvements;
* SPG support;
* PerkinElmer support;
* CSDM integration;
* plugin ecosystem improvements;
* documentation improvements.

User-facing improvements should generally take precedence when they provide immediate value.

---

# Longer-Term Candidates

The following topics remain interesting but are currently considered lower priority.

## NDMath Maintainability

Current assessment:

* behavior is largely understood;
* semantics appear stable;
* coverage is stronger than expected.

Future work may focus on:

* responsibility separation;
* readability;
* maintenance cost reduction.

No active work planned.

---

## Result Assembly Architecture

Potential future topic.

Motivation:

Several audits revealed multiple result-construction paths throughout the codebase.

Current evidence does not justify immediate work.

This topic should be revisited only if future maintenance difficulties emerge.

---

## Coordinate Arithmetic Evolution

Current assessment:

The existing model remains acceptable.

Future changes should be driven by demonstrated scientific requirements rather than architectural preference.

No active work planned.

---

# General Principles

Recent architecture reviews repeatedly converged toward the same conclusion:

The largest risks in SpectroChemPy are not mathematical correctness problems.

The largest risks are:

* implicit semantics;
* architectural fragmentation;
* duplicated logic;
* undocumented design decisions.

Future architectural work should therefore favor:

* clarification;
* simplification;
* consistency;
* explicit contracts;

over large-scale redesigns.

---

# Revisit Policy

Topics marked as completed or deferred should not automatically trigger new audits.

A new investigation should generally require at least one of:

* a concrete maintenance problem;
* a user-facing limitation;
* repeated bug reports;
* significant new requirements;
* evidence that previous assumptions were incorrect.

This document exists to preserve context and avoid rediscovering the same conclusions repeatedly.
