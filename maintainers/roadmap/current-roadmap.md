# SpectroChemPy — Current Architecture Roadmap

**Status:** Living document

## Purpose

This document records the current architectural priorities and active
candidates for SpectroChemPy.

It is not a formal RFC and not a commitment.  It is intended to help
maintainers prioritise work and avoid repeating completed investigations.

For completed campaigns, see [`completed-campaigns.md`](completed-campaigns.md).
For historical decision records, see the
[SpectroChemPy Maintainer Repository](https://github.com/spectrochempy/spectrochempy-maintainer).

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

## Active Documentation and Contract Topics

### Metadata Contract

Status: COMPLETED for conceptual contract phase and reader-alignment
implementation; future work is follow-up topics only.

Major outcomes:
- metadata propagation semantics documented;
- preservation and recomputation rules clarified;
- architectural direction established;
- conceptual RFC cluster completed for dimensions, coordinates, `CoordSet`,
  metadata taxonomy, labels, reader normalization, and provenance/history;
- stable architecture references promoted under
  `maintainers/architecture/metadata-and-support-model.md` and
  `maintainers/architecture/reader-normalization-architecture.md`;
- portable persistence alignment completed for all typed provenance fields;
- reader-normalization alignment completed for OMNIC, OPUS, JCAMP, LabSpec,
  CSV, TopSpin, WiRE, Quadera, SOC, MATLAB/DSO, and SPC.

Remaining follow-up work:
- result provenance;
- metadata implementation alignment outside the now-implemented portable
  subset baseline;
- optional portable extensions such as `filename` or label-only coordinates
  only if maintainers decide they are worth standardizing.

---

## Active Draft RFCs

### Array Class Responsibility

Status: Draft RFC

Reference:
[`maintainers/architecture/array-class-responsibility.md`](../architecture/array-class-responsibility.md)

Purpose: documents where responsibilities currently live across `NDArray`,
`NDComplexArray`, `Coord`, `NDDataset`, `NDMath`, and `NDIO`; treats
`NDLabelled` as a candidate future direction, not a decision.

Current recommendation: clarify mathematical semantics before any hierarchy
redesign.

---

### Mathematical Semantics and Metadata Propagation

Status: Draft RFC with characterization largely complete

Reference:
[`maintainers/architecture/mathematical-semantics-and-metadata-propagation.md`](../architecture/mathematical-semantics-and-metadata-propagation.md)

Purpose: documents how operations should behave across arithmetic, ufuncs,
reductions, shape operations, combination operations, labels, masks, units,
and metadata.

Current recommendation: establish and characterize the math semantics contract
before introducing a responsibility split such as `NDLabelled`.

---

### Analysis and Fit Result Architecture

Status: Audited; conceptual RFC plus implemented contract

Reference:
[`maintainers/rfcs/analysis-fit-result-architecture.md`](analysis-fit-result-architecture.md)

Implementation reference:
[`maintainers/architecture/result-object-contract-rfc.md`](../architecture/result-object-contract-rfc.md)

Campaign summary:
[`maintainers/architecture/result-object-migration-roadmap.md`](../architecture/result-object-migration-roadmap.md)

Purpose: documents the current result-surface model across decomposition,
analysis, and fit workflows.

Current recommendation: treat the Result Object campaign as complete for the
core estimators.  Remaining work is alignment, documentation, and compatibility
review, not a reopening of the completed architecture campaign.

---

## Active and Near-Term Candidates

The following topics offer the highest architectural value relative to their
expected complexity.

---

### Metadata Contract — Implementation Sequence

Status: Active campaign.  Conceptual phase complete; RFC still PROPOSED.

Implementation sequence:

PR1 — Gap analysis
:    Map current runtime behaviour against the Metadata Contract v1 matrix.

PR2 — Behaviour alignment
:    Fix divergences that are clearly accidental.

PR3 — RFC acceptance
:    Update Metadata Contract RFC status from PROPOSED to ACCEPTED.

PR4 — Processing alignment
:    Resolve Group A/B processing assembly split to conform to contract.

---

### Processing Wrapper Assembly

Status: Active debt.  Group A vs Group B divergence characterised in PR6.

The Filter subclass family (Group A: smooth, savgol, whittaker, denoise) rewrites
history and appends method suffixes.  The Baseline subclass family (Group B:
basc, detrend, asls) preserves name and appends history.  This divergence is
visible to users and blocks Metadata Contract alignment.

Implementation sequence:

PR1 — Policy decision
:    Determine whether the split is intentional or accidental.

PR2 — Alignment
:    Unify towards the chosen model, or document as deliberate policy.

---

### Writers Architecture

Status: Active debt.  No characterisation exists.

Implementation sequence:

PR1 — Behavioural characterisation
:    Audit all writer modules (CSV, Excel, JCAMP, MATLAB) for provenance,
     metadata, coordinate, and label handling.

PR2 — Normalisation contract
:    Semantic destination map and writer-side normalisation policy.

PR3 — Alignment
:    Fix divergences and align writer behaviour with the normalisation contract.

---

### Analysis Output Semantics Formalisation

Status: Active.  Characterisation complete (PR9); formalisation pending.

Three output families were identified: latent derived, diagnostic/model-summary,
and reconstructed source-space.  SVD remains an outlier.

Implementation sequence:

PR1 — Family formalisation
:    Document the three families and the `_set_output()` assembly pattern.

PR2 — SVD resolution
:    Align SVD or document its exception from the standard contract.

---

### Plugin Ecosystem — 0.11 Preparation

Status: Active.  7 official plugins; version bounds and compatibility policy
need attention before coordinated release.

Implementation sequence:

PR1 — Version bounds and classifier
:    Decide `<0.11` vs `<0.12` upper bound.  Resolve Cantera missing-classifier
     status.

PR2 — Compatibility policy
:    Document plugin version compatibility policy.

PR3 — Coordinated release plan
:    Determine which plugins release with 0.11 and in what order.

---

### Result Post-Campaign

Status: Optional cleanup.  Core Result Object campaign is complete.

Implementation sequence:

PR1 — Public imports
:    Make `ResultBase`, `AnalysisResult`, `FitResult` publicly importable.

PR2 — Gallery updates
:    Add `.result = pca.result` to estimator examples.

PR3 — Display integration
:    Semantic HTML display for Result objects (optional, may defer).

---

### Dynamic Plugin Discovery

Status: Future candidate.  Needs design review before implementation.

Motivation: replace static `KNOWN_PLUGIN_READERS` / `KNOWN_PLUGIN_NAMESPACES`
with entry-point-driven discovery.

Implementation sequence:

PR1 — Design
:    Entry-point replacement for static plugin lists.

PR2 — Implementation
:    Add dynamic path alongside static lists, with feature flag.

PR3 — Cleanup
:    Remove static lists after one release cycle.

---

### Mathematical Semantics Clarification

Status: Maintained contract; future targeted reviews only.

The current math layer is usable; operation semantics are now largely
characterised (PR1-PR9).  No active campaign planned.

---

### I/O Architecture Modernization (`#1105`)

Status: WAITING — depends on Dynamic Plugin Discovery and Metadata Contract.

The reader normalisation campaign stabilised semantic destinations but did not
modernise the structural I/O architecture.  Cannot proceed until dynamic plugin
discovery and the Metadata Contract provide the foundation.

---

### Core / Plugin Boundary Review (`#1172`)

Status: WAITING — depends on Dynamic Plugin Discovery and Plugin Ecosystem
stabilisation.

The broader question of what belongs in core versus official plugins is
deferred until the dynamic discovery mechanism and plugin compatibility policy
are established.

---

### Project Object Review

Status: Future candidate.

The core ownership invariants are implemented.  Open questions around typed
persistence, workflow integration, and Result object interaction remain but
require no immediate action.

---

### User-Facing Improvements

Architectural work should not prevent progress on user-facing capabilities.

Examples include:
- OPUS improvements;
- SPG support;
- PerkinElmer support;
- CSV interoperability follow-up such as simple 2D export;
- plugin ecosystem improvements;
- documentation improvements.

Explicit non-candidate:
- Excel reader/writer support in core is not planned; CSV remains the intended
  lightweight tabular interchange path.

Status: Ongoing parallel priority, not a single architecture campaign.

---

## General Principles

Recent architecture reviews repeatedly converged toward the same conclusion:

The largest risks in SpectroChemPy are not mathematical correctness problems.

The largest risks are:
- implicit semantics;
- architectural fragmentation;
- duplicated logic;
- undocumented design decisions.

Future architectural work should therefore favor:
- clarification;
- simplification;
- consistency;
- explicit contracts;

over large-scale redesigns.

---

## Revisit Policy

Topics marked as completed or deferred should not automatically trigger new
audits.

A new investigation should generally require at least one of:
- a concrete maintenance problem;
- a user-facing limitation;
- repeated bug reports;
- significant new requirements;
- evidence that previous assumptions were incorrect.

This document exists to preserve current priorities and avoid rediscovering the
same conclusions repeatedly.
