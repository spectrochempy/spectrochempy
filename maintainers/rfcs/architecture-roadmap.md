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

### Analysis and Fit Result Architecture Audit

- Decision: audit completed; conceptual RFC drafted; Result Object campaign now completed.
- Main conclusion: current result representation is object-owned but
  semantically fragmented.
- The Result Object contract has since been implemented for the core campaign.
- Remaining work is no longer per-estimator migration work; it is limited to
  deferred infrastructure questions such as serialization, Project
  integration, provenance enrichment, display integration, and caching.
- Reference: `maintainers/rfcs/analysis-fit-result-architecture.md`
- Implementation reference: `maintainers/architecture/result-object-contract-rfc.md`
- Campaign summary: `maintainers/architecture/result-object-migration-roadmap.md`

### Units and Dimensional Semantics Audit

- Decision: audit completed.
- Main conclusion: the unit system (Pint-based) is architecturally coherent
  but has a working-hybrid model (metadata vs scientific quantity) that is
  implicit rather than documented.
- Follow-up note: the previously identified `var()` unit-squaring bug
  (issue #1191) has now been fixed.
- Known scope gap: multi-coordinate dimension concatenation lacks unit
  conversion (documented in PR #1117).
- Minor inconsistency: `PLSRegression.coef` is unitless while
  `LinearRegressionAnalysis.coef` computes explicit `Y/X` unit ratios.
- No immediate redesign recommended.

### History and Provenance Semantics Characterization

- Decision: recent characterization and follow-up fixes materially narrowed the
  provenance contract.
- Main conclusion: local shape-like operations should append history rather
  than silently preserve stale entries, and provenance behavior is now better
  understood as its own semantic axis rather than a byproduct of metadata copy.
- Architectural reading:
  - history is not purely decorative metadata
  - provenance behavior varies by operation family and must be characterized
    explicitly
  - append-vs-rewrite remains a meaningful contract boundary

### Interpolation Semantics Characterization

- Decision: characterization is complete enough for roadmap purposes.
- Main conclusion: interpolation is the clearest current example of the same
  scientific object preserved on a changed coordinate grid.
- Architectural reading:
  - identity: same scientific object
  - geometry/support: locally rebuilt along the interpolated axis
  - provenance: appended rather than rewritten
  - result assembly: copy-first with local coordinate, mask, and label rebuild

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

### Script Removal

- Decision: remove the entire Script subsystem from the codebase.
- Removed: `Script` class, `run_script`, `run_all_scripts`, `%addscript` IPython
  magic, `makescript` decorator, all Script integration from `Project`.
- Rationale: the Script feature predates modern SpectroChemPy workflows
  (notebooks, Python scripts, packages, plugins) and was never widely
  documented or exercised. Removing it simplifies the Project data model and
  eliminates a maintenance burden.
- Compatibility posture: keep loading legacy serialized `_scripts` fields in
  `.pscp` files, but ignore them on read.

### CoordSet Storage Redesign

Decision:

* redesign considered complete;
* `_storage` is the runtime coordinate container;
* lifecycle behavior is clarified;
* same-dim semantics are stabilized;
* serialization compatibility is preserved.

Future work is limited to maintenance and follow-up improvements.

Campaign status: Completed.

---

### Metadata Propagation

Decision:

* semantic fragmentation was identified as the primary issue;
* Metadata Contract RFC was drafted;
* behavior characterization was completed.

No immediate redesign work is planned.

Future work should be driven by concrete metadata issues.

Campaign status: Active contract-consolidation topic, not an open migration
campaign.

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

Campaign status: Completed.

---

### NDMath

Decision:

* semantics appear substantially more coherent than initially expected;
* test coverage is stronger than expected;
* no urgent refactoring need has been identified.

Future work should focus on maintainability rather than semantics.

NDMath refactoring is deferred until a concrete maintenance problem appears.

Campaign status: Deferred / future candidate.

---

# Completed Campaigns

## CoordSet Storage Redesign

Status: Completed

Major outcomes:

* runtime storage unification;
* lifecycle clarification;
* removal of legacy storage assumptions;
* stabilization of same-dim behavior;
* improved architectural consistency.

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

## Result Object Architecture

Status: Completed for core estimators

Refer to
[`maintainers/architecture/result-object-contract-rfc.md`](../architecture/result-object-contract-rfc.md)
and
[`maintainers/architecture/result-object-migration-roadmap.md`](../architecture/result-object-migration-roadmap.md)
for the implemented contract and campaign summary.

Major outcomes:

* stable Result object contract implemented;
* core estimator migration completed;
* ownership, provenance boundary, serialization boundary, and display scope
  clarified.

Current priority: Deferred infrastructure follow-up only.

---

## Project Invariants

Status: Completed

Major outcomes:

* single-parent ownership clarified and enforced;
* cycle protection clarified;
* duplicate rejection clarified;
* key/name identity clarified.

Current priority: Closed except for future broader `Project` role questions.

---

## Project Copy Semantics

Status: Completed

Major outcomes:

* `Project.copy()` contract documented and implemented;
* detached recursive copy model clarified;
* stale-parent behavior removed from the maintained contract.

Current priority: Closed.

---

## Portable xarray / NetCDF Persistence

Status: Implemented core portable subset; follow-up extensions remain optional

Major outcomes:

* canonical RFCs drafted for xarray mapping and NetCDF persistence;
* portable same-dimension coordinates support merged;
* portable string-label subset support merged;
* portable scientific identity fields now preserved, including `name`, `title`,
  and `description`;
* portable provenance fields now preserved, including `author`, `origin`,
  `created`, `modified`, `acquisition_date`, and textual `history`;
* a tracked maintainer reference now exists under
  `maintainers/architecture/portable-persistence-model.md`.

Current priority: Closed as a baseline implementation campaign; future work is
limited to optional portable extensions and broader xarray/NetCDF RFC
synchronization.

---

## Security / Trusted Persistence

Status: Completed security hardening; active contract clarification

Major outcomes:

* native persistence security hardening completed;
* safe-writer and trust-boundary work merged;
* trusted-versus-portable persistence framing documented as the current
  architectural direction.

Current priority: Documentation and boundary clarification rather than urgent
implementation work.

---

# Active Documentation and Contract Topics

## Metadata Contract

Status: Conceptual phase completed; implementation alignment remains future work

Major outcomes:

* metadata propagation semantics documented;
* preservation and recomputation rules clarified;
* architectural direction established.
* conceptual RFC cluster completed for dimensions, coordinates, `CoordSet`,
  metadata taxonomy, labels, reader normalization, and provenance/history;
* stable architecture references promoted under
  `maintainers/architecture/metadata-and-support-model.md` and
  `maintainers/architecture/reader-normalization-architecture.md`.

Current priority: Low for conceptual design; future work is now mostly
implementation-alignment and follow-up RFC work rather than baseline contract
definition.

Remaining follow-up work:

* result provenance;
* reader cleanup/alignment;
* metadata implementation alignment outside the now-implemented portable
  subset baseline;
* optional portable extensions such as `filename` or label-only coordinates
  only if maintainers decide they are worth standardizing.

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

Status: Draft RFC with characterization largely complete

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

Current remaining work is primarily contract consolidation, open-question
triage, and selective follow-up decisions rather than broad new
characterization.

---

## Analysis and Fit Result Architecture

Status: Audited; conceptual RFC plus implemented contract

Reference:
[`maintainers/rfcs/analysis-fit-result-architecture.md`](analysis-fit-result-architecture.md)

Implementation reference:
[`maintainers/architecture/result-object-contract-rfc.md`](../architecture/result-object-contract-rfc.md)

Campaign summary:
[`maintainers/architecture/result-object-migration-roadmap.md`](../architecture/result-object-migration-roadmap.md)

Purpose:

* documents the current result-surface model across decomposition, analysis,
  and fit workflows;
* explains the current object-owned but semantically fragmented result model;
* preserves the broader conceptual context around the now-implemented Result
  object contract.

Current recommendation:

Treat the Result Object campaign as complete for the core estimators. Future
work in this area should focus on deferred infrastructure questions or on
broader conceptual follow-up, not on reopening the completed migration
campaign by default.

---

# Active and Near-Term Candidates

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

Status: Active.

---

## Project Object Review

Motivation:

The core ownership invariants of `Project` have been characterized and implemented
via a dedicated RFC (see
[`project-invariants-rfc.md`](./project-invariants-rfc.md), status: Implemented).

The long-term role of `Project` remains less clearly defined than other core
objects, but the invariants (single-parent ownership, acyclic hierarchy,
explicit duplicate rejection, key/name identity) are now enforced.

Questions that remain open:

* persistence model (typed-members constraints);
* relationship with workflows;
* interaction with future ecosystem developments (e.g., Result object
  integration).

Deferred topics documented in the RFC: copy semantics, root-name semantics.

Status: Future candidate.

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

Status: Ongoing parallel priority, not a single architecture campaign.

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

Status: Audited; RFC drafted; partial implementation completed.

Motivation:

Several maintainer reviews revealed multiple result-construction paths
throughout the codebase.

Current state:

- RFC drafted (`maintainers/rfcs/analysis-fit-result-architecture.md`).
- Result Object contract implemented and documented in
  `maintainers/architecture/result-object-contract-rfc.md`.
- Campaign summary documented in
  `maintainers/architecture/result-object-migration-roadmap.md`.

Future work should clarify:
- deferred infrastructure work around serialization, Project integration,
  provenance, display, and caching
- whether broader result-surface conventions beyond the current contract need
  further consolidation
- whether multi-output objects ever need a richer collection abstraction

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
