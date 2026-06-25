# Maintainer Architecture Documents

This directory contains durable architecture notes for maintainers. These files
preserve decisions, invariants, risks, and technical context that should remain
available after local audit notes are discarded.

For active maintainer contracts and RFCs, also see [`../rfcs/`](../rfcs/).
For the curated architecture entry point, see [`INDEX.md`](INDEX.md).
For the RFC inventory, see [`../rfcs/INDEX.md`](../rfcs/INDEX.md).
For promoted historical audits, see [`../audits/`](../audits/).
For campaign ordering and priorities, see
[`../roadmap/architecture-roadmap.md`](../roadmap/architecture-roadmap.md).

## Current High-Value References

For most current maintainer questions, start with:

- [`metadata-and-support-model.md`](metadata-and-support-model.md) for the
  central runtime model around dimensions, coordinates, `CoordSet`, labels,
  metadata, and provenance;
- [`reader-normalization-architecture.md`](reader-normalization-architecture.md)
  for reader semantic destinations and the completed Reader Alignment outcome;
- [`portable-persistence-model.md`](portable-persistence-model.md) for the
  implemented portable xarray / NetCDF persistence surface;
- [`result-object-contract-rfc.md`](result-object-contract-rfc.md) for the
  implemented Result Object contract.

These should usually answer current-state questions before any local audit
notes are needed.

## Placement Guide

| Location | Use for |
|---|---|
| `../rfcs/` | Normative or near-normative behavior contracts |
| `../roadmap/` | Active and completed migration roadmaps |
| `../audits/` | Promoted historical audits and decision-space analyses |
| this directory | Durable architecture notes, implementation maps, and design baselines |
| local notes under `audit/` | Local working notes that should not be versioned by default |

Tracked files in this directory may still contain `audit` in the filename.
That naming reflects document history, not authority level. If a file is
indexed here, it is part of the maintained documentation set and is distinct
from local campaign notes under the repository-level `audit/` directory.

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
| [`metadata-and-support-model.md`](metadata-and-support-model.md) | Authoritative architecture note | Primary maintained reference for dimensions, coordinates, `CoordSet`, metadata, labels, provenance, ownership, and persistence boundaries. |
| [`portable-persistence-model.md`](portable-persistence-model.md) | Authoritative architecture note | Primary maintained reference for the currently implemented portable xarray / NetCDF persistence surface. |
| [`reader-normalization-architecture.md`](reader-normalization-architecture.md) | Authoritative architecture note | Primary maintained reference for how readers should normalize imported information into the runtime model. |
| [`result-object-contract-rfc.md`](result-object-contract-rfc.md) | Implemented RFC | Defines the Result object contract, ownership, provenance boundary, serialization boundary, and display scope now validated by the completed campaign. |

## Reference Architecture Notes

| Document | Status | Purpose |
|---|---|---|
| [`coordset-storage-architecture.md`](coordset-storage-architecture.md) | Completed note | Captures final `CoordSet` storage invariants after the storage redesign. |
| [`display-architecture.md`](display-architecture.md) | Completed note | Documents the final post-migration display architecture and semantic HTML path. |
| [`result-object-migration-roadmap.md`](result-object-migration-roadmap.md) | Completed campaign summary | Summarizes the final Result Object campaign outcome, stable contract, architectural findings, deferred infrastructure work, and audit trail. |
| [`tensor-plugin-migration.md`](tensor-plugin-migration.md) | Completed note | Records the core/plugin boundary after CP/PARAFAC decomposition moved to the tensor plugin. |

## Related RFCs

| Document | Status | Purpose |
|---|---|---|
| [`../roadmap/architecture-roadmap.md`](../roadmap/architecture-roadmap.md) | Roadmap | Summarizes completed, active, and deferred architecture topics. |
| [`../rfcs/analysis-fit-result-architecture.md`](../rfcs/analysis-fit-result-architecture.md) | Draft conceptual RFC | Preserves the broader conceptual analysis of result surfaces and ownership around the implemented Result contract. |
| [`../rfcs/coordinate-arithmetic-semantics.md`](../rfcs/coordinate-arithmetic-semantics.md) | Accepted RFC | Maintainer position on coordinate arithmetic semantics. |
| [`../rfcs/metadata-contract.md`](../rfcs/metadata-contract.md) | Draft RFC | Normative direction for `NDDataset` metadata preservation, recomputation, override, merge, and drop behavior. |
| [`../rfcs/modeldata-semantic-contract.md`](../rfcs/modeldata-semantic-contract.md) | Accepted decision record | Audit and removal decision for orphaned `NDDataset.modeldata`. |
| [`../rfcs/roi-semantic-contract.md`](../rfcs/roi-semantic-contract.md) | Accepted decision record | Audit and removal decision for orphaned `NDDataset.roi`. |
