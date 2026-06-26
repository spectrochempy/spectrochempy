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

## Current priorities (2026-07)

### 1. Metadata Contract — gap analysis → acceptance

Analyse runtime behaviour against the PROPOSED Metadata Contract v1 matrix,
identify divergences, and advance the RFC towards ACCEPTED.

Expected next step:
    Gap audit across operation families using existing PR1-PR9 baselines.

### 2. Processing assembly — Group A/B policy decision

Resolve whether the Filter (Group A) and Baseline (Group B) metadata & history
divergence is intentional or accidental, then align.

Expected next step:
    Policy decision based on existing PR6 characterisation.

### 3. Writers — baseline characterisation

No writer normalisation exists.  Characterise current writer behaviour before
designing a normalisation contract.

Expected next step:
    Behavioural audit of all writer modules (analogous to reader PR1-PR9).

Release-driven exception — Plugin Ecosystem 0.11 preparation may temporarily
preempt the architecture priorities above when release coordination requires it.

---

# Decision Log

## 2026-06

### [✔] Global Result / Project / Persistence Audit — Closed 2026-06-26

- Result Object campaign closed after 9 core migration PRs, 2 plugin
  alignments, and post-alignment stabilisation.
- Core runtime Result contract completed: PCA, SVD, NMF, MCRALS, SIMPLISMA,
  EFA, FastICA, PLSRegression, Optimize all expose `.result`.
- Canonical API position: `.result` is the canonical grouped-output API for
  estimator-style objects that expose multiple scientific outputs and/or
  diagnostics.
- Explicit non-candidates: `Baseline`, `LSTSQ`, and `NNLS` do not currently
  justify `.result`; this reflects limited grouped-result value, not unfinished
  migration work.
- Closure assessment: no subclass of `ResultBase` was required. The concrete
  class (`AnalysisResult`, `FitResult`) is an implementation detail — `.result`
  IS the public API. Maintaining a separate public import path is not justified.
- Plugin alignment: completed for IRIS and TENSOR/CP. 0.11 compatibility is
  deferred to the release cycle.
- Project boundary: keep `Project` limited to `NDDataset` and nested `Project`.
  Dataset export is the supported bridge for saved Result outputs.
- Persistence boundary: established
  `NDDataset <-> xarray.Dataset <-> NetCDF` remains the dataset persistence
  path. Structured Result persistence is deferred per-RFC design choice.
- Lifecycle decision: the implementation is currently a live view. Retaining
  it, caching it, or adopting a fit-time snapshot remain open alternatives; no
  direction is currently preferred. This deferred decision does not block
  campaign closure.
- Non-blocking cosmetic follow-up: gallery examples that show estimator output
  could add `result = pca.result` (one extra line per example). This is
  documented in the closure audit and can be adopted gradually.
- Reference: `audit/~result-project-persistence-global-audit.md` and
  `audit/~result-object-final-audit.md`.

### Analysis and Fit Result Architecture Audit

- Decision: audit completed; conceptual RFC drafted; Result Object campaign now completed.
- Main conclusion: current result representation is object-owned but
  semantically fragmented.
- The Result Object contract has since been implemented for the core campaign.
- Remaining work is Result alignment, plugin alignment, documentation
  alignment, and compatibility review. Structured Result persistence and typed
  Project membership remain deferred optional directions.
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

Status: Completed (2026-06-26)

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

Current priority: Closed as an alignment campaign. Remaining follow-up is
limited to public Result exports, compatibility review, and documentation
alignment. Structured Result persistence and typed Project membership remain
deferred.

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

## Portable Metadata Subset / Portable xarray / NetCDF Persistence

Status: COMPLETED

Major outcomes:

* canonical RFCs drafted for xarray mapping and NetCDF persistence;
* portable same-dimension coordinates support merged;
* portable string-label subset support merged;
* portable `description`, `author`, `origin` aligned in PR1;
* portable `created`, `modified`, `acquisition_date` aligned in PR2;
* portable `history` aligned in PR3;
* full provenance/identity surface now portable through `scpy_*` carrier attrs;
* a tracked maintainer reference now exists under
  `maintainers/architecture/portable-persistence-model.md`.

Current priority: Closed as a baseline implementation campaign; future work is
limited to optional portable extensions such as `filename`, label-only
coordinates, and broader xarray/NetCDF RFC synchronization.

---

## Reader Alignment / Provenance Normalization

Status: COMPLETED

Refer to:
* [`maintainers/architecture/reader-normalization-architecture.md`](../architecture/reader-normalization-architecture.md)
  for the maintained normalization contract and campaign outcomes.
* [`maintainers/architecture/metadata-and-support-model.md`](../architecture/metadata-and-support-model.md)
  for the taxonomy and ownership model.

Major outcomes:

* semantic characterization baselines established for OMNIC, OPUS, JCAMP,
  LabSpec, CSV, TopSpin, WiRE, Quadera, SOC, MATLAB/DSO, and SPC;
* `acquisition_date` normalized across both campaign waves while retaining
  pointwise time coordinates where they already existed;
* `origin` consistency established and canonical origin vocabulary documented
  for the covered readers;
* history policy established: import events required, vendor processing history
  preserved where available, reorder events explicitly recorded;
* TopSpin gained import-history event; JCAMP sort dimension corrected;
* WiRE, Quadera, SOC, MATLAB/DSO, and SPC provenance normalization completed
  in the second-wave follow-up work;
* dual-time rule (`acquisition_date` + time coordinates) documented in both
  the developer guide and architecture notes;
* portable-persistence alignment for all typed provenance fields completed
  across PR1–PR3 of the parallel portable-persistence campaign.

Current priority: Closed as an alignment campaign. Remaining related items are
future enhancements rather than unfinished alignment work, including
Carroucell temperature semantics, optional vendor-log classification/refinement
policies, and cosmetic history-message harmonization.

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

Status: COMPLETED for conceptual contract phase and reader-alignment
implementation; future work is follow-up topics only

Major outcomes:

* metadata propagation semantics documented;
* preservation and recomputation rules clarified;
* architectural direction established;
* conceptual RFC cluster completed for dimensions, coordinates, `CoordSet`,
  metadata taxonomy, labels, reader normalization, and provenance/history;
* stable architecture references promoted under
  `maintainers/architecture/metadata-and-support-model.md` and
  `maintainers/architecture/reader-normalization-architecture.md`;
* portable persistence alignment completed for all typed provenance fields;
* reader-normalization alignment completed for OMNIC, OPUS, JCAMP, LabSpec,
  CSV, TopSpin, WiRE, Quadera, SOC, MATLAB/DSO, and SPC.

Current priority: Low for conceptual design; future work is limited to
follow-up topics listed below.

Remaining follow-up work:

* result provenance;
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

Treat the Result Object campaign as complete for the core estimators. Remaining
work is Result alignment, plugin alignment, documentation alignment, and
compatibility review, not a reopening of the completed architecture campaign.

---

# Active and Near-Term Candidates

The following topics appear to offer the highest architectural value relative to their expected complexity.

---

## Metadata Contract — Implementation Sequence

Status: Active campaign.  Conceptual phase complete; RFC still PROPOSED.

Implementation sequence:

PR1 — Gap analysis
:    Map current runtime behaviour against the Metadata Contract v1 matrix
     (preserve, recompute, override, drop per operation family).  Leverage
     existing PR1-PR9 baselines.

PR2 — Behaviour alignment
:    Fix divergences that are clearly accidental (low-effort, high-confidence
     changes).

PR3 — RFC acceptance
:    Update Metadata Contract RFC status from PROPOSED to ACCEPTED after
     gap analysis and alignment.

PR4 — Processing alignment
:    Resolve Group A/B processing assembly split to conform to contract.

---

## Processing Wrapper Assembly

Status: Active debt.  Group A vs Group B divergence characterised in PR6.

The Filter subclass family (Group A: smooth, savgol, whittaker, denoise) rewrites
history and appends method suffixes.  The Baseline subclass family (Group B:
basc, detrend, asls) preserves name and appends history.  This divergence is
visible to users and blocks Metadata Contract alignment.

Implementation sequence:

PR1 — Policy decision
:    Determine whether the split is intentional (derived vs same-object
     transformation) or accidental (inheritance from Filter vs Baseline).

PR2 — Alignment
:    Unify towards the chosen model, or document as deliberate policy.

---

## Writers Architecture

Status: Active debt.  No characterisation exists.

Writers were not included in any reader normalisation campaign.  Current
behaviour is undocumented.

Implementation sequence:

PR1 — Behavioural characterisation
:    Audit all writer modules (CSV, Excel, JCAMP, MATLAB) for provenance,
     metadata, coordinate, and label handling — analogous to reader PR1-PR9.

PR2 — Normalisation contract
:    Semantic destination map and writer-side normalisation policy.

PR3 — Alignment
:    Fix divergences and align writer behaviour with the normalisation contract.

---

## Analysis Output Semantics Formalisation

Status: Active.  Characterisation complete (PR9); formalisation pending.

Three output families were identified: latent derived, diagnostic/model-summary,
and reconstructed source-space.  SVD remains an outlier.

Implementation sequence:

PR1 — Family formalisation
:    Document the three families and the `_set_output()` / `_wrap_ndarray_output_to_nddataset()` assembly pattern.

PR2 — SVD resolution
:    Align SVD or document its exception from the standard contract.

---

## Plugin Ecosystem — 0.11 Preparation

Status: Active.  7 official plugins; version bounds and compatibility policy
need attention before coordinated release.

Implementation sequence:

PR1 — Version bounds and classifier
:    Decide `<0.11` vs `<0.12` upper bound.  Resolve Cantera missing-classifier
     status (official or experimental).

PR2 — Compatibility policy
:    Document plugin version compatibility policy (aligned or independent
     cadence).

PR3 — Coordinated release plan
:    Determine which plugins release with 0.11 and in what order.

---

## Result Post-Campaign

Status: Optional cleanup.  Core Result Object campaign is complete; remaining
packaging and documentation can be done opportunistically.

Implementation sequence:

PR1 — Public imports
:    Make `ResultBase`, `AnalysisResult`, `FitResult` publicly importable.

PR2 — Gallery updates
:    Add `.result = pca.result` to estimator examples.

PR3 — Display integration
:    Semantic HTML display for Result objects (optional, may defer).

---

## Dynamic Plugin Discovery

Status: Future candidate.  Depends on no unresolved blocker but needs design
review before implementation.

Motivation:

Replace static `KNOWN_PLUGIN_READERS` / `KNOWN_PLUGIN_NAMESPACES` with
entry-point-driven discovery, removing the need for core edits when adding
official plugins.

Implementation sequence:

PR1 — Design
:    Entry-point replacement for static plugin lists; review import-time
     performance and graceful degradation for missing plugins.

PR2 — Implementation
:    Add dynamic path alongside static lists, with feature flag.

PR3 — Cleanup
:    Remove static lists after one release cycle.

---

## Mathematical Semantics Clarification

Status: Maintained contract; future targeted reviews only.

The current math layer is usable; operation semantics are now largely
characterised (PR1-PR9).  No active campaign planned.  Targeted reviews
may occur when the Metadata Contract or class hierarchy work reaches this
area.

---

## I/O Architecture Modernization (`#1105`)

Status: WAITING — depends on Dynamic Plugin Discovery and Metadata Contract.

The reader normalisation campaign stabilised semantic destinations but did not
modernise the structural I/O architecture.  This topic cannot proceed until
dynamic plugin discovery and the Metadata Contract provide the foundation for
shared reader assembly helpers and format-parsing separation.

This remains a roadmap-level future candidate, not an active implementation
campaign.

---

## Core / Plugin Boundary Review (`#1172`)

Status: WAITING — depends on Dynamic Plugin Discovery and Plugin Ecosystem
stabilisation.

The broader question of what belongs in core versus official plugins is
deferred until the dynamic discovery mechanism and plugin compatibility policy
are established.

This remains a roadmap-level future candidate, not an active implementation
campaign.

---

## Project Object Review

Status: Future candidate.

The core ownership invariants are implemented.  Open questions around typed
persistence, workflow integration, and Result object interaction remain but
require no immediate action.

---

## User-Facing Improvements

Architectural work should not prevent progress on user-facing capabilities.

Examples include:

* OPUS improvements;
* SPG support;
* PerkinElmer support;
* CSV interoperability follow-up such as simple 2D export;
* plugin ecosystem improvements;
* documentation improvements.

Examples of possible future interoperability work:

* CSDM integration.

Explicit non-candidate:

* Excel reader/writer support in core is not planned; CSV remains the intended
  lightweight tabular interchange path and richer structured interchange
  should continue to prefer NetCDF/xarray or other scientific formats.

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

Status: Largely subsumed by Result Object contract.

Motivation:

Several maintainer reviews revealed multiple result-construction paths
throughout the codebase.

Current state:

- RFC drafted (`maintainers/rfcs/analysis-fit-result-architecture.md`).
- Result Object contract implemented and documented in
  `maintainers/architecture/result-object-contract-rfc.md`.
- Campaign summary documented in
  `maintainers/architecture/result-object-migration-roadmap.md`.
- This topic is now largely subsumed by the implemented Result Object
  contract.

Future work should clarify:
- Result alignment and plugin alignment around the implemented runtime Result
  contract
- optional infrastructure work around structured Result persistence, typed
  Project membership, provenance, display, and caching
- whether broader result-surface conventions beyond the current contract need
  further consolidation
- whether multi-output objects ever need a richer collection abstraction

---

## Result Alignment and Optional Infrastructure

Status: Campaign complete; optional infrastructure deferred.

Current assessment:

The core Result Object campaign is complete. The maintained estimator-style
scope is aligned in core and in official estimator-style plugins. Near-term
work is limited to compatibility review, public Result exports, and
documentation alignment.

The post-alignment audit concluded that not every fitted or stateful object
should receive `.result`. In particular, `Baseline`, `LSTSQ`, and `NNLS` do
not currently justify Result alignment because they offer limited grouped
scientific value relative to their existing APIs.

Deferred optional topics include:

* structured Result persistence;
* typed Project membership;
* provenance enrichment;
* display integration;
* caching.

Dataset persistence is already established. These optional topics should not be
treated as required follow-up or as a reopening of the completed core Result
Object migration campaign.

---

## Display / Plotting Ecosystem

Status: Future candidate.

Current assessment:

The core display architecture is implemented, but broader plotting ecosystem
questions remain visible as future design work rather than active campaign
items.

Examples include:

* backend abstraction boundaries;
* Matplotlib coupling;
* plugin-owned plotting extensions and formatting hooks.

This topic is distinct from the completed core display-architecture campaign.

---

## Carroucell Support Semantics

Status: Future candidate.

Current assessment:

Carroucell was explicitly left outside the completed Reader Alignment campaign.
Its remaining questions are support/label semantics questions rather than
unfinished core reader-alignment work.

Examples include:

* temperature representation;
* label-vs-auxiliary-coordinate placement;
* plugin-specific support semantics cleanup.

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
