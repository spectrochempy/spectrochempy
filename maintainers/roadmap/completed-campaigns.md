[Maintainers](../README.md) · [Roadmap](INDEX.md)

# SpectroChemPy — Completed Architecture Campaigns

**Status:** Public record of completed campaigns

This document lists architecture campaigns that have been completed or closed.
For current priorities and active candidates, see
[`current-roadmap.md`](current-roadmap.md).

---

## CoordSet Storage Redesign

Status: Completed

Major outcomes:
- runtime storage unification;
- lifecycle clarification;
- removal of legacy storage assumptions;
- stabilization of same-dim behaviour;
- improved architectural consistency.

---

## Integration Semantics

Status: Characterized

Major outcomes:
- single-axis integration behaviour documented for `trapezoid()` and `simpson()`;
- unit transformation behaviour documented as central operation semantics;
- integration classified as reduction-like in geometry but derived in
  scientific identity;
- mask handling flagged for separate scientific review.

Current priority: Low, except for possible mask-handling follow-up.

---

## Coordinate Arithmetic Review

Status: Completed

Major outcomes:
- current behaviour documented;
- design space explored;
- future options identified.

Current priority: Low.

---

## Dataset-vs-Coord Semantics

Status: Completed

Major outcomes:
- object-model ambiguity identified;
- semantic bug corrected;
- object responsibilities clarified.

Current priority: Closed.

---

## Display Architecture (#843)

Status: Completed

Refer to [`maintainers/architecture/display-architecture.md`](../architecture/display-architecture.md)
for the final architecture.

Key outcomes: Coord, CoordSet, NDDataset, and Project now use the semantic HTML
path (`_repr_sections()` → `_render_sections()`).

---

## Result Object Architecture

Status: Completed (2026-06-26)

Refer to:
- [`maintainers/architecture/result-object-contract-rfc.md`](../architecture/result-object-contract-rfc.md)
- [`maintainers/architecture/result-object-migration-roadmap.md`](../architecture/result-object-migration-roadmap.md)

Major outcomes:
- stable Result object contract implemented;
- core estimator migration completed;
- ownership, provenance boundary, serialization boundary, and display scope
  clarified.

Current priority: Closed as an alignment campaign.  Remaining follow-up is
limited to public Result exports, compatibility review, and documentation
alignment.

---

## Project Invariants

Status: Completed

Major outcomes:
- single-parent ownership clarified and enforced;
- cycle protection clarified;
- duplicate rejection clarified;
- key/name identity clarified.

---

## Project Copy Semantics

Status: Completed

Major outcomes:
- `Project.copy()` contract documented and implemented;
- detached recursive copy model clarified;
- stale-parent behaviour removed from the maintained contract.

---

## Portable Metadata Subset / Portable xarray / NetCDF Persistence

Status: COMPLETED

Major outcomes:
- canonical RFCs drafted for xarray mapping and NetCDF persistence;
- portable same-dimension coordinates support merged;
- portable string-label subset support merged;
- portable `description`, `author`, `origin` aligned in PR1;
- portable `created`, `modified`, `acquisition_date` aligned in PR2;
- portable `history` aligned in PR3;
- full provenance/identity surface now portable through `scpy_*` carrier attrs;
- tracked maintainer reference:
  `maintainers/architecture/portable-persistence-model.md`.

Current priority: Closed as a baseline implementation campaign.

---

## Reader Alignment / Provenance Normalization

Status: COMPLETED

Refer to:
- [`maintainers/architecture/reader-normalization-architecture.md`](../architecture/reader-normalization-architecture.md)
- [`maintainers/architecture/metadata-and-support-model.md`](../architecture/metadata-and-support-model.md)

Major outcomes:
- semantic characterization baselines established for OMNIC, OPUS, JCAMP,
  LabSpec, CSV, TopSpin, WiRE, Quadera, SOC, MATLAB/DSO, and SPC;
- `acquisition_date` normalized across both campaign waves;
- `origin` consistency established and canonical origin vocabulary documented;
- history policy established: import events required, vendor processing history
  preserved where available, reorder events explicitly recorded;
- dual-time rule (`acquisition_date` + time coordinates) documented in both
  the developer guide and architecture notes;
- portable-persistence alignment for all typed provenance fields completed
  across PR1–PR3.

Current priority: Closed as an alignment campaign.

---

## Security / Trusted Persistence

Status: Completed security hardening; active contract clarification

Major outcomes:
- native persistence security hardening completed;
- safe-writer and trust-boundary work merged;
- trusted-versus-portable persistence framing documented as the current
  architectural direction.

Current priority: Documentation and boundary clarification rather than urgent
implementation work.
