# Maintainer Architecture Documents

This directory contains durable architecture notes for maintainers. These files
preserve decisions, invariants, risks, and technical context that should remain
available after local audit notes are discarded.

For active maintainer contracts and RFCs, also see [`../rfcs/`](../rfcs/).
For the curated architecture entry point, see [`INDEX.md`](INDEX.md).
For the RFC inventory, see [`../rfcs/INDEX.md`](../rfcs/INDEX.md).
For campaign ordering and priorities, see
[`../rfcs/architecture-roadmap.md`](../rfcs/architecture-roadmap.md).

## Placement Guide

| Location | Use for |
|---|---|
| `../rfcs/` | Normative or near-normative behavior contracts and roadmap documents |
| this directory | Durable audits, implementation maps, design baselines, and draft architecture analyses |
| `audit/~*.md` | Local working notes that should not be versioned by default |

## Reading Guidance

Within this tracked directory:

- **current** material is the first place to look for maintained technical
  behavior or durable architecture boundaries;
- **reference** material provides supporting risk maps, design-space analysis,
  or implementation context;
- **historical** material remains useful for migration context but is no longer
  the primary authority.

The curated grouping of those documents lives in [`INDEX.md`](INDEX.md).

## Active Draft Architecture Analyses

| Document | Status | Purpose |
|---|---|---|
| [`array-class-responsibility.md`](array-class-responsibility.md) | Draft RFC | Maps responsibilities across `NDArray`, `NDComplexArray`, `Coord`, `NDDataset`, `NDMath`, and `NDIO` before any hierarchy cleanup. |
| [`mathematical-semantics-and-metadata-propagation.md`](mathematical-semantics-and-metadata-propagation.md) | Draft RFC / characterization largely complete | Characterizes operation behavior, result assembly, scientific object identity, provenance, and metadata propagation before `NDMath` or class hierarchy changes. |
| [`result-object-contract-rfc.md`](result-object-contract-rfc.md) | Implemented RFC | Defines the Result object contract, ownership, provenance boundary, serialization boundary, and display scope now validated by the completed campaign. |

## Reference Architecture Notes

| Document | Status | Purpose |
|---|---|---|
| [`coordset-storage-architecture.md`](coordset-storage-architecture.md) | Completed note | Captures final `CoordSet` storage invariants after the storage redesign. |
| [`display-architecture.md`](display-architecture.md) | Completed note | Documents the final post-migration display architecture and semantic HTML path. |
| [`result-object-migration-roadmap.md`](result-object-migration-roadmap.md) | Completed campaign summary | Summarizes the final Result Object campaign outcome, stable contract, architectural findings, deferred infrastructure work, and audit trail. |
| [`tensor-plugin-migration.md`](tensor-plugin-migration.md) | Completed note | Records the core/plugin boundary after CP/PARAFAC decomposition moved to the tensor plugin. |

## Completed Audits

| Document | Status | Purpose |
|---|---|---|
| [`coordinate-arithmetic-audit.md`](coordinate-arithmetic-audit.md) | Completed audit | Technical map for coordinate compatibility in arithmetic. |
| [`coordinate-arithmetic-decision-audit.md`](coordinate-arithmetic-decision-audit.md) | Decision-space audit | Tradeoff analysis for possible future coordinate arithmetic models. |
| [`dataset-vs-coord-arithmetic-audit.md`](dataset-vs-coord-arithmetic-audit.md) | Completed audit | Explains why `Coord` is an axis/support object rather than a signal operand. |
| [`display-architecture-audit.md`](display-architecture-audit.md) | Completed audit | Historical context for the display architecture migration. |
| [`ndmath-maintainability-audit.md`](ndmath-maintainability-audit.md) | Deferred reference | Maps risks concentrated in `NDMath`. |
| [`plotting-audit.md`](plotting-audit.md) | Completed audit | Baseline for plotting backend extensibility and Matplotlib coupling. |
| [`units-audit.md`](units-audit.md) | Completed audit | Baseline for unit handling, quantity propagation, and unit semantics. |

## Related RFCs

| Document | Status | Purpose |
|---|---|---|
| [`../rfcs/architecture-roadmap.md`](../rfcs/architecture-roadmap.md) | Roadmap | Summarizes completed, active, and deferred architecture topics. |
| [`../rfcs/analysis-fit-result-architecture.md`](../rfcs/analysis-fit-result-architecture.md) | Draft conceptual RFC | Preserves the broader conceptual analysis of result surfaces and ownership around the implemented Result contract. |
| [`../rfcs/coordinate-arithmetic-semantics.md`](../rfcs/coordinate-arithmetic-semantics.md) | Draft RFC | Maintainer position on coordinate arithmetic semantics. |
| [`../rfcs/metadata-contract.md`](../rfcs/metadata-contract.md) | Draft RFC | Normative direction for `NDDataset` metadata preservation, recomputation, override, merge, and drop behavior. |
| [`../rfcs/modeldata-semantic-contract.md`](../rfcs/modeldata-semantic-contract.md) | Accepted decision record | Audit and removal decision for orphaned `NDDataset.modeldata`. |
| [`../rfcs/roi-semantic-contract.md`](../rfcs/roi-semantic-contract.md) | Accepted decision record | Audit and removal decision for orphaned `NDDataset.roi`. |
