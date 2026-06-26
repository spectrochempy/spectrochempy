# Architecture Index

Curated entry point for tracked maintainer architecture documents.

For normative contracts (RFCs), see [`../rfcs/INDEX.md`](../rfcs/INDEX.md).
For promoted historical audits, see [`../audits/INDEX.md`](../audits/INDEX.md).
For campaign ordering, see
[`../roadmap/current-roadmap.md`](../roadmap/current-roadmap.md).

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

## Core Maintained Architecture

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
| [`result-object-contract-rfc.md`](result-object-contract-rfc.md) | Implemented tracked contract for result objects, ownership, serialization boundary, and display scope. |

## Reader Architecture

| File | Description |
|---|---|
| [`reader-normalization-architecture.md`](reader-normalization-architecture.md) | Primary architecture reference for normalizing imported reader information into typed fields, coordinates, labels, provenance, `Meta`, and parser-only state. |

## Historical Migrations

These documents preserve campaign summaries and durable risk maps from
completed migrations. They are not current normative references.

| File | Description |
|---|---|
| [`result-object-migration-roadmap.md`](result-object-migration-roadmap.md) | Final campaign summary for the completed Result Object migration. |
| [`tensor-plugin-migration.md`](tensor-plugin-migration.md) | Durable note on the core/plugin boundary after the tensor migration. |

## Reading Order

For a quick orientation path, start with:

1. [`../README.md`](../README.md)
2. [`../roadmap/current-roadmap.md`](../roadmap/current-roadmap.md)
3. [`metadata-and-support-model.md`](metadata-and-support-model.md)
4. [`reader-normalization-architecture.md`](reader-normalization-architecture.md)
5. [`portable-persistence-model.md`](portable-persistence-model.md)
6. [`../rfcs/INDEX.md`](../rfcs/INDEX.md)
7. [`coordset-storage-architecture.md`](coordset-storage-architecture.md)
8. [`display-architecture.md`](display-architecture.md)
9. [`result-object-contract-rfc.md`](result-object-contract-rfc.md)
10. [`mathematical-semantics-and-metadata-propagation.md`](mathematical-semantics-and-metadata-propagation.md)
11. [`array-class-responsibility.md`](array-class-responsibility.md)
