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

### Integration Semantics Characterization

- Decision: PR8 characterization is complete for `trapezoid()` / `simpson()`.
- Main conclusion: integration is reduction-like in geometry, but is the
  clearest current example of a derived scientific quantity generated through
  reduction.
- Architectural reading:
  - CoordSet category: Reduce
  - result assembly: Reduction Assembly with local unit / metadata rewrite
  - provenance: rewritten, not appended
  - identity: derived scientific object / derived quantity
- Follow-up note: current mask handling during integration warrants separate
  scientific review, because masked values appear to contribute numerically to
  the computed integral while mask information survives on the returned object.

### Analysis Output Semantics Characterization

- Decision: PR9 characterization is complete for representative decomposition
  analysis outputs.
- Scope covered: `PCA`, `SVD`, `EFA`, `NMF`, `MCRALS`.
- Main conclusion: analysis methods do not produce one semantic class of
  output.
- Architectural reading:
  - no new top-level taxonomy node required yet
  - latent outputs: derived analysis objects
  - diagnostic outputs: derived diagnostic / model-summary objects
  - reconstructed outputs: same scientific object in a modeled /
    reconstructed representation
  - current assembly behavior is largely driven by
    `_wrap_ndarray_output_to_nddataset()` / `_set_output()`
- Exception note: `SVD` currently exposes diagnostic vectors and raw factor
  arrays but does not implement the generic `transform()` reduction API, so it
  should not be forced into the same contract as `PCA` / `NMF`.

### Modeldata Removal

- Decision: remove orphaned `NDDataset.modeldata` from the runtime array model.
- Rationale: no production writers, accidental stale propagation, and only one
  legacy plotting reader.
- Compatibility posture: keep loading legacy serialized `modeldata` fields, but
  ignore them on read.

### ROI Removal

- Decision: remove orphaned `roi` support from the runtime array model
  (`NDArray` / `Coord` / `NDDataset`).
- Rationale: inconsistent stale propagation, no stable global semantic
  contract, and no active documented user-facing feature.
- Compatibility posture: keep loading legacy serialized `roi` fields, but
  ignore them on read.

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

### Display Architecture and Hypercomplex Display Follow-Up

Decision:

* semantic display architecture work is considered complete for Coord, CoordSet,
  NDDataset, and Project;
* the later hypercomplex display review identified RR/RI/IR/II quaternion
  display as plugin-owned display semantics;
* the intended ownership boundary is core dispatch/fallback plus plugin-owned
  hypercomplex formatting.

Future display work should preserve this plugin/core separation.

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

## Integration Semantics

Status: Characterized

Major outcomes:

* single-axis integration behavior documented for `trapezoid()` and `simpson()`;
* unit transformation behavior documented as central operation semantics;
* integration classified as reduction-like in geometry but derived in
  scientific identity;
* mask handling flagged for separate scientific review.

Current priority: Low, except for possible mask-handling follow-up.

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

Refer to [`maintainers/architecture/display-architecture.md`](../architecture/display-architecture.md)
for the final architecture.

Key outcomes: Coord, CoordSet, NDDataset, and Project now use the
semantic HTML path (`_repr_sections()` → `_render_sections()`). The
authoritative reference is `display-architecture.md`.

---

# Active Draft RFCs

## Array Class Responsibility

Status: Draft RFC

Reference:
[`maintainers/architecture/array-class-responsibility.md`](../architecture/array-class-responsibility.md)

Purpose:

* documents where responsibilities currently live across `NDArray`,
  `NDComplexArray`, `Coord`, `NDDataset`, `NDMath`, and `NDIO`;
* identifies responsibility mismatches around labels, masks, math/result
  assembly, complex/hypercomplex support, metadata, and serialization;
* treats `NDLabelled` or an equivalent layer as a candidate future direction,
  not a decision.

Current recommendation:

Clarify mathematical semantics before any hierarchy redesign.

---

## Mathematical Semantics and Metadata Propagation

Status: Draft RFC / Audit in progress

Reference:
[`maintainers/architecture/mathematical-semantics-and-metadata-propagation.md`](../architecture/mathematical-semantics-and-metadata-propagation.md)

Purpose:

* documents how operations should behave across arithmetic, ufuncs,
  reductions, shape operations, combination operations, labels, masks, units,
  and metadata;
* separates behavior questions from class hierarchy questions;
* connects operation semantics to the Metadata Contract.

Current recommendation:

Establish and characterize the math semantics contract before introducing a
responsibility split such as `NDLabelled`.

---

# Near-Term Architectural Candidates

The following topics appear to offer the highest architectural value relative to their expected complexity.

---

## Mathematical Semantics Clarification

Motivation:

The current math layer is usable, but operation semantics are still partly
implicit and path-dependent.

Questions include:

* metadata propagation by operation category;
* direct labels versus coordinate labels;
* coordinate compatibility policy;
* mask behavior for geometry-changing operations;
* result assembly consistency across core, processing, analysis, and plugins.

This topic should be addressed before any class hierarchy split.

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

Active related work:

* mathematical semantics and metadata propagation RFC;
* array class responsibility RFC.

Implementation refactoring remains deferred until the behavior contract is
clearer.

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
