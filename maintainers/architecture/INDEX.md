# Architecture Index

This index is the curated entry point for tracked maintainer architecture
documents.

Use this file when you want to understand which documents define current
behavior, which ones provide durable technical reference, and which ones remain
valuable mainly as historical context.

For normative maintainer contracts and position statements, also see the
[`../rfcs/INDEX.md`](../rfcs/INDEX.md) and the
[`../roadmap/architecture-roadmap.md`](../roadmap/architecture-roadmap.md).

For promoted historical audits, see [`../audits/INDEX.md`](../audits/INDEX.md).

These tracked architecture notes are authoritative once a design or campaign
has stabilized. Local audits, implementation notes, campaign logs, and
characterization reports remain useful context, but they are not the first
place to look for the maintained project contract.

## Start Here by Topic

Use this shorter routing table before scanning the full inventory:

| Topic | First document |
|---|---|
| Runtime model for dimensions, coordinates, labels, metadata, provenance | [`metadata-and-support-model.md`](metadata-and-support-model.md) |
| Reader import normalization | [`reader-normalization-architecture.md`](reader-normalization-architecture.md) |
| Portable xarray / NetCDF persistence | [`portable-persistence-model.md`](portable-persistence-model.md) |
| Result objects and result-surface ownership | [`result-object-contract-rfc.md`](result-object-contract-rfc.md) |
| CoordSet storage and lifecycle | [`coordset-storage-architecture.md`](coordset-storage-architecture.md) |
| Display architecture | [`display-architecture.md`](display-architecture.md) |
| Future operation semantics work | [`mathematical-semantics-and-metadata-propagation.md`](mathematical-semantics-and-metadata-propagation.md) |

## Core Architecture

These documents are the primary tracked references for current maintained
behavior, durable architecture boundaries, or active maintainer-facing
contracts.

| File | Description |
|---|---|
| [`array-class-responsibility.md`](array-class-responsibility.md) | Responsibility map across the core array classes before any future hierarchy cleanup. |
| [`coordset-storage-architecture.md`](coordset-storage-architecture.md) | Final tracked architecture for the completed `CoordSet` storage redesign. |
| [`display-architecture.md`](display-architecture.md) | Final post-migration display architecture and semantic HTML model. |
| [`mathematical-semantics-and-metadata-propagation.md`](mathematical-semantics-and-metadata-propagation.md) | Active maintainer-facing contract draft for operation behavior, result assembly, provenance, and metadata propagation. |
| [`metadata-and-support-model.md`](metadata-and-support-model.md) | Primary architecture reference for dimensions, coordinates, `CoordSet`, labels, metadata, provenance, ownership, and persistence boundaries. |
| [`portable-persistence-model.md`](portable-persistence-model.md) | Primary architecture reference for the currently implemented portable xarray / NetCDF persistence surface, including portable identity, structure, labels, provenance, and `Meta` payloads. |
| [`reader-normalization-architecture.md`](reader-normalization-architecture.md) | Primary architecture reference for normalizing imported reader information into typed fields, coordinates, labels, provenance, `Meta`, and parser-only state. |
| [`result-object-contract-rfc.md`](result-object-contract-rfc.md) | Implemented tracked contract for result objects, ownership, serialization boundary, and display scope. |
| [`result-object-migration-roadmap.md`](result-object-migration-roadmap.md) | Final campaign summary for the completed Result Object migration. |

## Reference Architecture

These documents are useful supporting references, decision-space analyses, or
durable risk maps that maintainers may still rely on.

| File | Description |
|---|---|
| [`tensor-plugin-migration.md`](tensor-plugin-migration.md) | Durable note on the core/plugin boundary after the tensor migration. |

## Reading Order

For a quick orientation path, start with:

1. [`../README.md`](../README.md)
2. [`../roadmap/architecture-roadmap.md`](../roadmap/architecture-roadmap.md)
3. [`metadata-and-support-model.md`](metadata-and-support-model.md)
4. [`reader-normalization-architecture.md`](reader-normalization-architecture.md)
5. [`portable-persistence-model.md`](portable-persistence-model.md)
6. [`../rfcs/INDEX.md`](../rfcs/INDEX.md)
7. [`coordset-storage-architecture.md`](coordset-storage-architecture.md)
8. [`display-architecture.md`](display-architecture.md)
9. [`result-object-contract-rfc.md`](result-object-contract-rfc.md)
10. [`mathematical-semantics-and-metadata-propagation.md`](mathematical-semantics-and-metadata-propagation.md)
11. [`array-class-responsibility.md`](array-class-responsibility.md)
