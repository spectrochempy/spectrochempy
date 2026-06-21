# RFC Index

This index provides a maintainer-facing entry point to the RFC documents stored
in this directory.

## Purpose

In SpectroChemPy, RFCs record maintainer-level behavior contracts, position
statements, and proposed architectural boundaries.

They are different from the other documentation layers:

- RFCs define or propose maintainer-facing contracts and decisions.
- Architecture notes preserve durable implementation maps, risk analyses, and
  reference context.
- Audits are working documents and implementation-history notes that may later
  become outdated, superseded, or purely historical.

Authoritative maintainer references usually live in:

- the roadmap;
- accepted RFCs;
- implemented RFCs;
- tracked architecture notes once a design has stabilized.

Audits, implementation notes, campaign logs, and characterization reports are
not authoritative by default.

The usual RFC lifecycle is:

```text
PROPOSED -> ACCEPTED -> IMPLEMENTED
                 \-> SUPERSEDED
```

Not every document passes through every stage. Some RFCs are accepted position
statements or decision records rather than implementation plans.

## Status Definitions

### `PROPOSED`

The document records a proposed maintainer contract or architecture direction.
It is useful for guidance, but the behavior is not yet fully adopted.

### `ACCEPTED`

The document records an accepted maintainer position or decision. It is
authoritative even if it is not primarily an implementation roadmap.

### `IMPLEMENTED`

The documented contract has been adopted in merged behavior and should be
treated as the current maintained reference.

### `SUPERSEDED`

The RFC remains useful for history, but another document is now the primary
authoritative reference.

## Related Roadmap

For campaign ordering and architecture priorities, see
[`architecture-roadmap.md`](architecture-roadmap.md).

## RFC Inventory

| File | Title | Status | Short description | Related issue(s) |
|---|---|---|---|---|
| [`analysis-fit-result-architecture.md`](analysis-fit-result-architecture.md) | Analysis and Fit Result Architecture | `PROPOSED` | Conceptual background for result ownership, result surfaces, and the role of `NDDataset`; implementation authority moved to the tracked Result contract. | — |
| [`coord-labels-portable-semantics.md`](coord-labels-portable-semantics.md) | Coord Labels Portable Semantics | `PROPOSED` | Defines which `Coord.labels` usages belong to the portable persistence contract. | — |
| [`coordinate-and-coordset-semantics.md`](coordinate-and-coordset-semantics.md) | Coordinate and CoordSet Semantics Contract | `PROPOSED` | Defines `Coord` and `CoordSet` as the runtime support and coordinate-ownership layer around structural dimensions, including defaults, siblings, lifecycle, and portability boundaries. | — |
| [`coordinate-arithmetic-semantics.md`](coordinate-arithmetic-semantics.md) | Coordinate Arithmetic Semantics RFC | `ACCEPTED` | Records the current maintainer position on coordinate arithmetic semantics and its future evolution space. | — |
| [`csdm-position-statement.md`](csdm-position-statement.md) | Current Position on CSDM | `ACCEPTED` | States the current role of CSDM as an optional exchange format rather than native or primary portable persistence. | `#1153` |
| [`dimensional-semantics-contract.md`](dimensional-semantics-contract.md) | Dimensional Semantics Contract | `PROPOSED` | Defines dimensions as structural axis identifiers, coordinate anchors, and portable schema elements without adopting full alignment semantics. | — |
| [`display-representation-model-rfc.md`](display-representation-model-rfc.md) | Display / Representation Model RFC | `SUPERSEDED` | Historical representation-model RFC retained for context; final authority now lives in the tracked display architecture note. | `#843` |
| [`label-semantics-contract.md`](label-semantics-contract.md) | Label Semantics Contract | `PROPOSED` | Defines labels as a distinct support-local semantic surface spanning annotation, identification, categorization, display, and portability boundaries without treating labels as structural dimensions. | — |
| [`metadata-contract.md`](metadata-contract.md) | Metadata Contract v1 | `PROPOSED` | Normative direction for `NDDataset` metadata preservation, recomputation, override, merge, and drop behavior. | — |
| [`metadata-taxonomy-contract.md`](metadata-taxonomy-contract.md) | Metadata Taxonomy Contract | `PROPOSED` | Defines the maintainer taxonomy for scientific identity, structural, provenance, presentation, and extension/private metadata, plus ownership and persistence boundaries. | — |
| [`modeldata-semantic-contract.md`](modeldata-semantic-contract.md) | modeldata — Semantic Contract RFC | `ACCEPTED` | Decision record for removal of `NDDataset.modeldata` while preserving load compatibility for legacy serialized state. | `#1168` |
| [`nddataset-xarray-mapping-specification.md`](nddataset-xarray-mapping-specification.md) | NDDataset ↔ xarray Dataset Mapping Specification | `PROPOSED` | Canonical mapping contract between `NDDataset` and `xarray.Dataset` for portable persistence and interchange; a substantial subset is already reflected in current code. | — |
| [`project-copy-semantics-rfc.md`](project-copy-semantics-rfc.md) | Project Copy Semantics | `IMPLEMENTED` | Defines the maintained `Project.copy()` contract and the deep-copy model used by the completed campaign. | `#1164` |
| [`project-invariants-rfc.md`](project-invariants-rfc.md) | Project Invariants and Ownership Semantics | `IMPLEMENTED` | Defines the maintained `Project` ownership, parent, cycle, and key/name identity invariants. | `#1164` |
| [`roi-semantic-contract.md`](roi-semantic-contract.md) | roi — Semantic Contract RFC | `ACCEPTED` | Decision record for removal of orphaned `NDDataset.roi` runtime state while keeping legacy files readable. | — |
| [`scientific-object-model-and-persistence-boundaries.md`](scientific-object-model-and-persistence-boundaries.md) | Scientific Object Model and Persistence Boundaries | `PROPOSED` | Defines the proposed object-role and persistence-boundary vocabulary for datasets, Projects, results, and extensions. | — |
| [`trusted-and-portable-persistence.md`](trusted-and-portable-persistence.md) | Trusted and Portable Persistence | `PROPOSED` | Defines the conceptual distinction between trusted native persistence and portable scientific persistence; parts of this framing already guide the current security posture. | — |
| [`xarray-backed-netcdf-persistence.md`](xarray-backed-netcdf-persistence.md) | xarray-Backed NetCDF Persistence | `PROPOSED` | Defines the proposed NetCDF contract for the xarray-backed portable persistence layer; parts of the portable round-trip path are already implemented. | — |

## Notes

- Status assignments above reflect the current maintainer documentation review
  and roadmap synchronization work.
- Several `PROPOSED` RFCs already describe behavior that is partially
  reflected in current code. They remain `PROPOSED` here because the relevant
  contract is not yet fully adopted or reclassified.
- `coordinate-arithmetic-semantics.md` is listed as `ACCEPTED` here because it
  already records the current maintainer position future discussions are
  expected to start from.
