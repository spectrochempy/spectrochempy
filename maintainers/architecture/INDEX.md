# Architecture Index

This index is the curated entry point for tracked maintainer architecture
documents.

Use this file when you want to understand which documents define current
behavior, which ones provide durable technical reference, and which ones remain
valuable mainly as historical context.

For normative maintainer contracts and position statements, also see the
[`../rfcs/INDEX.md`](../rfcs/INDEX.md) and the
[`../rfcs/architecture-roadmap.md`](../rfcs/architecture-roadmap.md).

These tracked architecture notes are authoritative once a design or campaign
has stabilized. Local audits, implementation notes, campaign logs, and
characterization reports remain useful context, but they are not the first
place to look for the maintained project contract.

## Core Architecture

These documents are the primary tracked references for current maintained
behavior, durable architecture boundaries, or active maintainer-facing
contracts.

| File | Description |
|---|---|
| [`array-class-responsibility.md`](array-class-responsibility.md) | Responsibility map across the core array classes before any future hierarchy cleanup. |
| [`coordinate-arithmetic-audit.md`](coordinate-arithmetic-audit.md) | Technical map of the current coordinate-compatibility model used by arithmetic. |
| [`coordset-storage-architecture.md`](coordset-storage-architecture.md) | Final tracked architecture for the completed `CoordSet` storage redesign. |
| [`dataset-vs-coord-arithmetic-audit.md`](dataset-vs-coord-arithmetic-audit.md) | Semantic boundary showing why `Coord` is an axis/support object rather than a signal operand. |
| [`display-architecture.md`](display-architecture.md) | Final post-migration display architecture and semantic HTML model. |
| [`mathematical-semantics-and-metadata-propagation.md`](mathematical-semantics-and-metadata-propagation.md) | Active maintainer-facing contract draft for operation behavior, result assembly, provenance, and metadata propagation. |
| [`metadata-and-support-model.md`](metadata-and-support-model.md) | Primary architecture reference for dimensions, coordinates, `CoordSet`, labels, metadata, provenance, ownership, and persistence boundaries. |
| [`reader-normalization-architecture.md`](reader-normalization-architecture.md) | Primary architecture reference for normalizing imported reader information into typed fields, coordinates, labels, provenance, `Meta`, and parser-only state. |
| [`result-object-contract-rfc.md`](result-object-contract-rfc.md) | Implemented tracked contract for result objects, ownership, serialization boundary, and display scope. |
| [`result-object-migration-roadmap.md`](result-object-migration-roadmap.md) | Final campaign summary for the completed Result Object migration. |

## Reference Architecture

These documents are useful supporting references, decision-space analyses, or
durable risk maps that maintainers may still rely on.

| File | Description |
|---|---|
| [`coordinate-arithmetic-decision-audit.md`](coordinate-arithmetic-decision-audit.md) | Decision-space analysis for possible future coordinate arithmetic models. |
| [`ndmath-maintainability-audit.md`](ndmath-maintainability-audit.md) | Risk map for the responsibility concentration inside `NDMath`. |
| [`plotting-audit.md`](plotting-audit.md) | Baseline architecture review for plotting backend separation and extensibility. |
| [`tensor-plugin-migration.md`](tensor-plugin-migration.md) | Durable note on the core/plugin boundary after the tensor migration. |
| [`units-audit.md`](units-audit.md) | Baseline reference for unit handling, quantity propagation, and unit semantics. |

## Historical Context

These documents remain useful for historical understanding, but they are not
the first place to look for the current maintained contract.

| File | Description |
|---|---|
| [`display-architecture-audit.md`](display-architecture-audit.md) | Pre-migration display analysis preserved for history after the final display architecture note. |

## Reading Order

For a quick orientation path, start with:

1. [`../README.md`](../README.md)
2. [`../rfcs/architecture-roadmap.md`](../rfcs/architecture-roadmap.md)
3. [`../rfcs/INDEX.md`](../rfcs/INDEX.md)
4. [`coordset-storage-architecture.md`](coordset-storage-architecture.md)
5. [`metadata-and-support-model.md`](metadata-and-support-model.md)
6. [`reader-normalization-architecture.md`](reader-normalization-architecture.md)
7. [`display-architecture.md`](display-architecture.md)
8. [`result-object-contract-rfc.md`](result-object-contract-rfc.md)
9. [`mathematical-semantics-and-metadata-propagation.md`](mathematical-semantics-and-metadata-propagation.md)
10. [`array-class-responsibility.md`](array-class-responsibility.md)
