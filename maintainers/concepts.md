# Concept Index

This index maps SpectroChemPy concepts to the documents that define them.
Use it when you need to find all documentation about a specific topic.

Documents are organised by role:

- **What** (RFC) — behaviour contracts and position statements.
- **How** (Architecture) — implementation organisation and durable reference.
- **Why** (Audit) — historical context and decision background.

## Metadata

| Role | Document |
|---|---|
| What | [`rfcs/metadata-contract.md`](rfcs/metadata-contract.md) (PROPOSED) |
| What | [`rfcs/metadata-taxonomy-contract.md`](rfcs/metadata-taxonomy-contract.md) (ACCEPTED) |
| How | [`architecture/metadata-and-support-model.md`](architecture/metadata-and-support-model.md) |
| What (modeldata) | [`rfcs/modeldata-semantic-contract.md`](rfcs/modeldata-semantic-contract.md) (IMPLEMENTED) |
| What (ROI) | [`rfcs/roi-semantic-contract.md`](rfcs/roi-semantic-contract.md) (IMPLEMENTED) |

## Coordinates / CoordSet

| Role | Document |
|---|---|
| What | [`rfcs/coordinate-and-coordset-semantics.md`](rfcs/coordinate-and-coordset-semantics.md) (ACCEPTED) |
| What | [`rfcs/coordinate-arithmetic-semantics.md`](rfcs/coordinate-arithmetic-semantics.md) (ACCEPTED) |
| What | [`rfcs/coord-labels-portable-semantics.md`](rfcs/coord-labels-portable-semantics.md) (PROPOSED) |
| How | [`architecture/coordset-storage-architecture.md`](architecture/coordset-storage-architecture.md) |

## Dimensions

| Role | Document |
|---|---|
| What | [`rfcs/dimensional-semantics-contract.md`](rfcs/dimensional-semantics-contract.md) (ACCEPTED) |
| How | [`architecture/metadata-and-support-model.md`](architecture/metadata-and-support-model.md) |

## Labels

| Role | Document |
|---|---|
| What | [`rfcs/label-semantics-contract.md`](rfcs/label-semantics-contract.md) (ACCEPTED) |
| What | [`rfcs/coord-labels-portable-semantics.md`](rfcs/coord-labels-portable-semantics.md) (PROPOSED) |

## Provenance & History

| Role | Document |
|---|---|
| What | [`rfcs/provenance-and-history-contract.md`](rfcs/provenance-and-history-contract.md) (ACCEPTED) |
| How | [`architecture/metadata-and-support-model.md`](architecture/metadata-and-support-model.md) |

## Persistence

| Role | Document |
|---|---|
| What (portable subset) | [`rfcs/portable-metadata-subset-contract.md`](rfcs/portable-metadata-subset-contract.md) (IMPLEMENTED) |
| What (trusted vs portable) | [`rfcs/trusted-and-portable-persistence.md`](rfcs/trusted-and-portable-persistence.md) (PROPOSED) |
| What (xarray / NetCDF) | [`rfcs/xarray-backed-netcdf-persistence.md`](rfcs/xarray-backed-netcdf-persistence.md) (PROPOSED) |
| What (NDDataset ↔ xarray) | [`rfcs/nddataset-xarray-mapping-specification.md`](rfcs/nddataset-xarray-mapping-specification.md) (PROPOSED) |
| How | [`architecture/portable-persistence-model.md`](architecture/portable-persistence-model.md) |

## Display

| Role | Document |
|---|---|
| What | [`rfcs/display-representation-model-rfc.md`](rfcs/display-representation-model-rfc.md) (SUPERSEDED) |
| How | [`architecture/display-architecture.md`](architecture/display-architecture.md) |

## Result Objects

| Role | Document |
|---|---|
| What | [`rfcs/analysis-fit-result-architecture.md`](rfcs/analysis-fit-result-architecture.md) (SUPERSEDED) |
| How | [`architecture/result-object-contract-rfc.md`](architecture/result-object-contract-rfc.md) |
| How | [`architecture/result-object-migration-roadmap.md`](architecture/result-object-migration-roadmap.md) |

## Readers

| Role | Document |
|---|---|
| What (normalisation) | [`rfcs/reader-metadata-normalization-contract.md`](rfcs/reader-metadata-normalization-contract.md) (ACCEPTED) |
| How | [`architecture/reader-normalization-architecture.md`](architecture/reader-normalization-architecture.md) |

## Project Model

| Role | Document |
|---|---|
| What (invariants) | [`rfcs/project-invariants-rfc.md`](rfcs/project-invariants-rfc.md) (IMPLEMENTED) |
| What (copy semantics) | [`rfcs/project-copy-semantics-rfc.md`](rfcs/project-copy-semantics-rfc.md) (IMPLEMENTED) |
| What (object model) | [`rfcs/scientific-object-model-and-persistence-boundaries.md`](rfcs/scientific-object-model-and-persistence-boundaries.md) (PROPOSED) |

## I/O Namespace API

| Role | Document |
|---|---|
| What | [`rfcs/namespace-api-convention.md`](rfcs/namespace-api-convention.md) (IMPLEMENTED) |
| How | [`api-conventions.md`](api-conventions.md) |

## CSDM

| Role | Document |
|---|---|
| What | [`rfcs/csdm-position-statement.md`](rfcs/csdm-position-statement.md) (ACCEPTED) |

## Tensor / Array Layer

| Role | Document |
|---|---|
| How | [`architecture/array-class-responsibility.md`](architecture/array-class-responsibility.md) |
| How | [`architecture/tensor-plugin-migration.md`](architecture/tensor-plugin-migration.md) |

## Operations / Mathematics

| Role | Document |
|---|---|
| How | [`architecture/mathematical-semantics-and-metadata-propagation.md`](architecture/mathematical-semantics-and-metadata-propagation.md) |
