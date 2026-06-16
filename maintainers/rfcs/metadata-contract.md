# Metadata Contract v1

## Status

Draft Maintainer RFC with June 2026 decision updates.

This document represents the current maintainer consensus and serves as the
reference for future metadata-related discussions.

It may evolve based on practical experience and implementation feedback.

This document is normative. It defines the metadata semantics that future
SpectroChemPy work should implement. It does not define how those semantics
must be implemented.

The key words MUST, SHOULD, and MAY are to be interpreted as normative
requirements for maintainers and future contributors.

## 1. Purpose

The Metadata Contract exists to make metadata behavior predictable across
SpectroChemPy operations.

It solves one problem: metadata semantics must depend on the kind of scientific
operation being performed, not on the internal construction path used to build
the result.

Inside scope:

- metadata semantics for `NDDataset` result objects;
- operation-category rules for metadata preservation, recomputation,
  override, merge, and drop;
- multi-source rules;
- history rules;
- governance for gradual adoption.

Outside scope:

- result assembly mechanisms;
- implementation details;
- APIs;
- coordinate arithmetic;
- alignment semantics;
- internal builder or helper design.

## 2. Guiding Principle

Metadata Contract v1 adopts the following principle as normative:

```text
Preserve scientific context.
Recompute geometry-dependent metadata.
Never silently drop user metadata.
```

This principle governs all field-level rules in this document.

Interpretation:

- Scientific context SHOULD survive ordinary single-source transformations.
- Metadata tied to shape, dimensions, coordinates, masking, or physical domain
  MUST be recomputed, replaced, or dropped when it would otherwise become
  stale.
- User-supplied metadata MUST NOT disappear without an explicit semantic rule
  that justifies that loss.

## 3. Metadata Inventory

The contract covers the following `NDDataset` fields.

`data` is not part of this contract. It is the scientific result itself, not
metadata about the result.

### Scientific context

- `title`
- `units`
- `meta`
- `description`

These fields describe what the data means, how it should be interpreted, or
additional user-supplied scientific context attached to the dataset.

### Geometry-dependent metadata

- `dims`
- `coordset`
- `mask`
- `transposed`

These fields depend on array structure, coordinate structure, masking, or
representation. They become stale when geometry or domain changes.

### Historical removed fields

- `roi`
- `modeldata`

These fields were previously carried on `NDDataset` as geometry-dependent or
derived state, but both have now been removed from the public runtime data
model.

- `roi` was removed after audit as orphaned historical interactive-selection
  state with no stable semantic contract.
- `modeldata` was removed as orphaned historical fit/model infrastructure —
  see the `modeldata RFC <modeldata-semantic-contract.md>`_ and issue `#1168`_.

Legacy serialized `roi` and `modeldata` entries remain load-compatible and
should be ignored on read rather than restored onto live objects.

### Provenance metadata

- `author`
- `history`
- `created`
- `modified`
- `origin`
- `filename`

These fields describe lineage, source, authorship, timestamps, and provenance.

### Identity metadata

- `name`

This field identifies the dataset as an object within user workflows.

### Classification note

The classification is by dominant semantics, not by exclusive meaning.

For example:

- `title` is scientific context even though some operations may override it;
- `filename` is provenance even though users often treat it as scientific
  context;

## 4. Operation Categories

Only the following top-level categories are valid in Metadata Contract v1.

### A — Preserve Geometry

Operations that change values while preserving dataset geometry.

Examples: elementwise arithmetic, unary math, filtering, baseline correction,
scaling, normalization.

### B — Modify Geometry

Operations that change dimensionality, coordinate values, shape, or coordinate
structure while remaining within the same scientific object family.

Examples: reduction, integration, interpolation, reshape, concatenation,
stacking, matrix contraction.

### C — Derived Dataset

Operations that produce a new scientific object with distinct semantics from
the source dataset.

Examples: scores, loadings, components, extracted feature datasets.

### D — Domain Transformation

Operations that transform the physical or analytical domain of the data.

Examples: frequency-domain transforms, analytic-signal transforms, spectral
power transforms.

## 5. Metadata Matrix

The following matrix is normative for single-source operations unless the
multi-source rules in Section 6 apply.

| Field | A — Preserve Geometry | B — Modify Geometry | C — Derived Dataset | D — Domain Transformation |
|---|---|---|---|---|
| `dims` | preserve | recompute | recompute | recompute |
| `coordset` | preserve | recompute | recompute | override |
| `name` | preserve | preserve (default) | recompute | recompute |
| `title` | preserve | preserve | override | override |
| `mask` | recompute | recompute | recompute | recompute |
| `units` | preserve | recompute | recompute | recompute |
| `meta` | preserve | preserve | preserve | preserve |
| `author` | preserve | preserve | preserve | preserve |
| `description` | preserve | preserve | override | override |
| `history` | override | override | override | override |
| `created` | preserve | preserve | preserve | preserve |
| `modified` | recompute | recompute | recompute | recompute |
| `origin` | preserve | preserve | preserve | preserve |
| `transposed` | preserve | recompute | recompute | recompute |
| `filename` | preserve | preserve | preserve | preserve |

### Matrix interpretation

- `preserve` means the field SHOULD be carried forward unchanged.
- `recompute` means the field MUST be recomputed from result semantics.
- `override` means the field MUST be replaced by an operation-defined value.
- `drop` means the field MUST be cleared because inherited values would be
  misleading or invalid.

### Field notes

- `title` in Category B is preserved by default. An operation MAY override it
  when the result's scientific meaning clearly changes.
- `name` in Category B is preserved by default. Operations MAY override the
  default behavior when preserving the original name would be misleading.
- `description` in Category B is preserved by default. An operation MAY
  override it when preserving the source description would misdescribe the
  result.
- `history` is listed as `override` in all categories because v1 governs it
  through explicit history rules rather than through plain inheritance. See
  Section 7.
- `created` is preserved in v1 as the creation date of the source dataset
  lineage. This remains an open question for future revisions.
- `filename` is preserved for single-source operations. Multi-source
  operations follow the stricter rules in Section 6.

### Historical field decisions

- `roi` — **removed from the runtime array model.**  The ROI audit showed inconsistent stale
  propagation, no visible global semantic contract, and no active documented
  user-facing feature depending on it.  Future interactive selection work, if
  needed, should use explicit view/model state rather than hidden dataset
  structure.
- `modeldata` — **removed from the runtime array model.**  The modeldata RFC audit showed
  zero production writes, accidental stale propagation, and a single
  production reader (`plot1d`).  Fit/model outputs should be explicit datasets
  or future fit-result objects.
- Legacy serialized `roi` / `modeldata` fields SHOULD be ignored on load for
  backwards compatibility rather than reattached to runtime dataset objects.

## 6. Multi-Source Rules

Multi-source rules take precedence over the single-source matrix whenever more
than one dataset contributes scientifically meaningful content to the result.

### General rule

- The primary source is the left-most `NDDataset` unless an operation defines
  a different scientific primary source.
- `meta` MUST NOT be deep-merged in v1.
- `filename` MUST be dropped whenever more than one dataset is a true source of
  the result.
- Secondary-source `history` MUST NOT be blindly inherited.

### Binary arithmetic

For binary arithmetic:

- If only one source is an `NDDataset` and the other input is scalar-like, the
  operation is treated as single-source.
- If both sources are datasets, the left operand is the primary source.
- `meta`, `author`, `description`, `origin`, and `title` SHOULD come from the
  primary source unless the operation explicitly overrides a field.
- `filename` MUST be dropped for dataset-vs-dataset arithmetic.
- `history` MUST append one generated entry naming the operation and the
  secondary source. Secondary-source history is not merged.

### Concatenation

For concatenation:

- The first dataset is the primary structural source.
- `coordset`, `units`, and other structural defaults SHOULD follow the first
  dataset unless operation-specific rules require recomputation.
- `author` and `origin` SHOULD be merged as deduplicated textual provenance.
- `description` SHOULD be generated to describe the assembled result.
- `meta` SHOULD come from the first dataset unchanged in v1.
- `filename` MUST be dropped.
- `history` MUST be replaced by a generated concatenation record.

### Stacking

Stacking follows the same metadata rules as concatenation.

The introduction of a new axis does not change the provenance rules:

- first dataset remains primary;
- merged provenance remains allowed for `author` and `origin`;
- `filename` is dropped;
- `history` is replaced with a generated stacking record.

### Dot

For `dot` and similar asymmetric matrix contractions:

- the left operand is the primary source unless a future operation explicitly
  documents a different primary source;
- `meta`, `author`, `description`, and `origin` SHOULD come from the primary
  source;
- `title` MAY be composed from both sources;
- `units` MUST be recomputed from both sources;
- `coordset` MUST be recomputed from the surviving dimensions;
- `filename` MUST be dropped;
- `history` MUST be replaced by a generated record naming both sources.

### Other multi-source operations

For other multi-source operations:

- maintainers MUST explicitly identify the primary source in code review or
  design notes;
- if there is no meaningful single primary source, provenance fields SHOULD be
  generated rather than inherited;
- `meta` SHOULD default to the primary source without deep merge;
- `filename` MUST be dropped;
- `history` MUST be generated and MUST NOT silently combine multiple full
  histories.

## 7. History Rules

History is governed by three actions: append, replace, and generated.

### Append

History MUST append a generated entry when:

- the operation is single-source; and
- the result remains part of the same dataset lineage.

This is the default for most Category A operations and most single-source
Category B and D operations.

### Replace

History MUST be replaced when:

- the result is assembled from multiple true dataset sources; or
- inheriting prior history would misrepresent the result as a continuation of a
  single lineage; or
- the result is a synthetic summary or assembled object whose identity is new.

This is the default for concatenation, stacking, dot, and most Category C
results.

### Generated

Every operation that writes history MUST generate an operation-specific entry.

A generated entry SHOULD identify:

- the operation performed;
- the relevant dimension or domain when applicable;
- the contributing source names for multi-source operations when useful.

The contract does not standardize message text in v1. It standardizes only the
semantic rule that history updates must be explicit and operation-derived.

## 8. Known Open Questions

The following topics remain open for future revision.

### Name semantics

Category-level defaults are defined in v1, but the exact distinction between
preserve, override, clear, and generated naming still needs maintainers to settle more
precisely, especially for Category B results.

### Filename semantics

Single-source preservation is adopted in v1, and multi-source drop is adopted
for clarity. A future revision may choose stricter semantics if maintainers
decide that filename should always represent only a direct external file source.

### Created / modified semantics

v1 preserves `created` and recomputes `modified`. Some maintainers may prefer
resetting both for new result objects. This question is intentionally left open.

### Removed historical fields

- `modeldata` has been removed from `NDDataset`.  See the `modeldata RFC
  <modeldata-semantic-contract.md>`_ for the full audit and rationale.
- `roi` has been removed from `NDDataset` as orphaned interactive-selection
  state.  See the `roi RFC <roi-semantic-contract.md>`_ for the audit summary
  and decision record.

### Title and description semantics

Category-level defaults are defined in v1, but the threshold for when an
operation MUST override rather than preserve these fields remains partly
judgment-based.

## 9. Non-Goals

Metadata Contract v1 does not define:

- result assembly;
- coordinate arithmetic;
- alignment semantics;
- CoordSet internal storage design;
- label ownership and NDLabelled discussions;
- implementation details;
- helper abstractions;
- APIs.

It is a semantic specification only.

## 10. Relationship to Other Contracts

Metadata Contract v1 is intended to be the semantic foundation for:

- a future Result Assembly Contract;
- a future Coordinate Arithmetic Contract.

Those documents may refine implementation and compatibility rules but should
not redefine the metadata semantics established here without explicit review.

## 11. Adoption Strategy

Adoption SHOULD proceed incrementally.

### Governance guidance

- New operations SHOULD declare which top-level category they belong to.
- PRs that change metadata behavior SHOULD reference this contract explicitly.
- Exceptions to this contract SHOULD be documented in tests and maintainer
  notes.

### Migration guidance

- Existing behavior does not need to be normalized everywhere at once.
- High-user-impact paths SHOULD adopt the contract first.
- Characterization tests SHOULD be preserved until contract-driven behavior
  replaces them deliberately.
- Contract adoption SHOULD happen field-by-field and family-by-family rather
  than through broad semantic rewrites.

### Review guidance

- Maintainers SHOULD review metadata changes against category semantics, not
  against local implementation habits.
- A metadata change is complete only when the intended semantic rule is clear,
  tested, and documented.
