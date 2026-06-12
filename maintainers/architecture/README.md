# Maintainer Architecture Documents

This directory contains curated architecture references for maintainers.

These documents are not user documentation. They preserve durable decisions,
risks, invariants, and review context that should remain available after local
audit notes are discarded.

## Current Reference Documents

### CoordSet Storage Architecture

File: `coordset-storage-architecture.md`

Status: Completed architecture note

Summary:
Documents the final post-redesign `CoordSet` storage model. `_storage` is now
the runtime coordinate container, the group model is the semantic reference for
lookup and lifecycle reasoning, and the historical `_coords` validator model
must not return as runtime storage.

Why it exists:
The storage redesign was a multi-PR migration. This note captures the final
invariants without preserving every intermediate PR note.

### Metadata Contract

File: `../rfcs/metadata-contract.md`

Status: Draft Maintainer RFC

Summary:
Defines how `NDDataset` metadata should be preserved, recomputed, overridden,
or dropped across operation categories.

Why it exists:
Metadata behavior should depend on operation semantics, not on the internal
construction path used to build a result.

### Coordinate Arithmetic Semantics

File: `../rfcs/coordinate-arithmetic-semantics.md`

Status: Draft Maintainer RFC

Summary:
Records the current maintainer position on coordinate arithmetic: NumPy-style
broadcasting, unit-aware arithmetic, and last-dimension coordinate validation
for dataset-vs-dataset arithmetic.

Why it exists:
The arithmetic model is intentionally spectroscopy-oriented and should be
discussed from an explicit baseline rather than rediscovered from `NDMath`.

### Coordinate Arithmetic Audit

File: `coordinate-arithmetic-audit.md`

Status: Completed architecture audit

Summary:
Describes the current implementation contract for coordinate compatibility in
arithmetic, including what is checked, what is ignored, and where the source of
truth lives.

Why it exists:
The RFC records the maintainer position. This audit preserves the technical
map that led to it.

### Coordinate Arithmetic Decision Audit

File: `coordinate-arithmetic-decision-audit.md`

Status: Historical decision-space audit

Summary:
Compares the plausible future models for coordinate arithmetic, from preserving
the current spectroscopic last-dimension model to adopting richer coordinate
or dimension semantics.

Why it exists:
Future arithmetic proposals should account for the tradeoffs already explored
before reopening the design space.

### Dataset-vs-Coord Arithmetic Audit

File: `dataset-vs-coord-arithmetic-audit.md`

Status: Completed semantic audit

Summary:
Documents why `Coord` is best treated as an axis/support object rather than a
signal-bearing operand for `NDDataset` arithmetic.

Why it exists:
The object-model boundary matters for future arithmetic changes and for
deciding when a 1D operand should be a `Coord` or an `NDDataset`.


### Units Audit

File: `units-audit.md`

Status: Completed architecture audit

Summary:
Comprehensive audit of the unit system and quantity propagation across SpectroChemPy.
Maps current implementation (Pint integration, unit storage in NDArray, NDMath
arithmetic propagation, Coord unit handling), identifies tested vs. untested
behaviors, catalogs inconsistent or underspecified semantics (e.g., coordinate units
ignored in arithmetic, silent unit drops in comparisons, power operation edge cases),
and proposes a conservative v1 unit contract alongside prioritized recommendations.

Why it exists:
Unit handling is central to SpectroChemPy's value proposition in spectroscopy.
This audit preserves the technical baseline for future unit-system evolution,
including immediate fixes (coordinate unit validation, comparison fixes, power
validation), medium-term improvements (coordinate unit semantics redesign,
Quantity/NDDataset unification), and long-term redesign (full Pint Quantity
integration, dimensional coordinate system).


### NDMath Maintainability Audit

File: `ndmath-maintainability-audit.md`

Status: Deferred maintenance reference

Summary:
Maps the responsibilities concentrated in `NDMath`, including NumPy wrapping,
operator dispatch, unit handling, masks, coordinate checks, result construction,
history, and plugin hooks.

Why it exists:
No immediate refactor is planned, but future changes to shared arithmetic paths
need this risk map.

### Display Architecture Audit

File: `display-architecture-audit.md`

Status: Deferred architecture audit

Summary:
Documents fragmentation across compact text, detailed text, HTML display, and
`Project` representation paths.

Why it exists:
It provides a starting point for a future display contract if representation
consistency becomes an active maintenance priority.

### Tensor Plugin Migration

File: `tensor-plugin-migration.md`

Status: Completed architecture note

Summary:
Records the boundary created when CP/PARAFAC decomposition moved from core into
the official tensor plugin.

Why it exists:
Plugin boundaries, compatibility shims, and dependency ownership are durable
maintainer decisions.

## Planning Documents

### Architecture Roadmap

File: `../rfcs/architecture-roadmap.md`

Status: Maintainer discussion document

Summary:
Summarizes completed and deferred architecture topics so maintainers can avoid
repeating recent investigations.

Why it exists:
PR-by-PR audit details should remain local, while their durable outcomes should
be visible to future maintainers.
