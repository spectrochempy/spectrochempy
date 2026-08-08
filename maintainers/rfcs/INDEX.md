[Maintainer Docs](../README.md)

# RFC Index

This index lists the maintainer RFCs that are stable enough to remain in the
main `spectrochempy` repository.

The usual lifecycle remains:

```text
Proposed -> Accepted -> Implemented
                \-> Superseded
```

For this public subset, the main emphasis is on `Accepted`,
`Implemented`, and `Superseded` documents that still provide durable
architectural context.

## Naming Conventions

- files in `rfcs/` should normally describe the topic only; the directory
  already carries the RFC classification, so filename suffixes such as
  `-rfc` are usually unnecessary;
- titles should normally name the topic rather than repeating `RFC` in the
  heading, with RFC status carried by the document status section and this
  index;
- code identifiers such as `modeldata` or `roi` should be typeset as code in
  titles when the document is about the exact historical field.

## Status Conventions

Use Title Case for status labels. Prefer:

- `Proposed maintainer RFC`
- `Accepted maintainer RFC`
- `Implemented maintainer RFC`
- `Superseded RFC`

When a status section contains explanatory text, prefer the same pattern:

- first line: the short status label only;
- next sentence: one plain-language sentence explaining the current role of
  the document.

## Status Definitions

| Status | Meaning |
|---|---|
| `Proposed` | Draft position or contract still under maintainer review. |
| `Accepted` | Accepted position or decision; authoritative even without full implementation. |
| `Implemented` | Contract adopted in merged behavior; current maintained reference. |
| `Superseded` | Historical; another document is now the primary reference. |

## RFC Inventory

| File | Title | Status | Short description |
|---|---|---|---|
| [`analysis-fit-result-architecture.md`](analysis-fit-result-architecture.md) | Analysis and Fit Result Architecture | `Superseded` | Historical conceptual background retained for context; final authority lives in the tracked Result object architecture notes. |
| [`coordinate-and-coordset-semantics.md`](coordinate-and-coordset-semantics.md) | Coordinate and CoordSet Semantics Contract | `Accepted` | Defines `Coord` and `CoordSet` as the runtime support and coordinate-ownership layer around structural dimensions. |
| [`coordinate-arithmetic-semantics.md`](coordinate-arithmetic-semantics.md) | Coordinate Arithmetic Semantics | `Accepted` | Records the current maintainer position on coordinate arithmetic semantics and its future evolution space. |
| [`csdm-position-statement.md`](csdm-position-statement.md) | Current Position on CSDM | `Accepted` | States the current role of CSDM as an optional exchange format rather than native or primary portable persistence. |
| [`dimensional-semantics-contract.md`](dimensional-semantics-contract.md) | Dimensional Semantics Contract | `Accepted` | Defines dimensions as structural axis identifiers, coordinate anchors, and portable schema elements. |
| [`display-representation-model.md`](display-representation-model.md) | Display / Representation Model | `Superseded` | Historical representation-model RFC retained for context; final authority now lives in the display architecture note. |
| [`label-semantics-contract.md`](label-semantics-contract.md) | Label Semantics Contract | `Accepted` | Defines labels as a distinct support-local semantic surface spanning annotation, identification, categorization, display, and portability boundaries. |
| [`metadata-taxonomy-contract.md`](metadata-taxonomy-contract.md) | Metadata Taxonomy Contract | `Accepted` | Defines the maintainer taxonomy for scientific identity, structural, provenance, presentation, and extension/private metadata. |
| [`modeldata-semantic-contract.md`](modeldata-semantic-contract.md) | `modeldata` Semantic Contract | `Implemented` | Decision record for removal of `NDDataset.modeldata` while preserving load compatibility for legacy serialized state. |
| [`namespace-api-convention.md`](namespace-api-convention.md) | Namespace API Convention | `Implemented` | Defines the uniform namespace-based public API for I/O and domain-specific operations across core and plugins. |
| [`portable-metadata-subset-contract.md`](portable-metadata-subset-contract.md) | Portable Metadata Subset Contract | `Implemented` | Defines the maintained portable subset for identity, structure, labels, provenance, and `Meta`. |
| [`project-copy-semantics.md`](project-copy-semantics.md) | Project Copy Semantics | `Implemented` | Defines the maintained `Project.copy()` contract and deep-copy model. |
| [`project-invariants.md`](project-invariants.md) | Project Invariants and Ownership Semantics | `Implemented` | Defines the maintained `Project` ownership, parent, cycle, and key/name identity invariants. |
| [`provenance-and-history-contract.md`](provenance-and-history-contract.md) | Provenance and History Contract | `Accepted` | Defines provenance as source and lineage context, and history as the explicit event trail. |
| [`reader-metadata-normalization-contract.md`](reader-metadata-normalization-contract.md) | Reader Metadata Normalization Contract | `Accepted` | Defines how imported reader metadata should be normalized across typed fields, coordinates, labels, provenance, `Meta`, and runtime-only parser state. |
| [`roi-semantic-contract.md`](roi-semantic-contract.md) | `roi` Semantic Contract | `Implemented` | Decision record for removal of orphaned `NDDataset.roi` runtime state while keeping legacy files readable. |

## Scope Note

Proposed RFCs that are still under active private iteration, immature
architectural exploration, assistant work in progress, and private roadmap
material remain outside this public subset for now.
