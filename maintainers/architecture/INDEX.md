[Maintainer Docs](../README.md)

# Architecture Index

Curated entry point for the stable architecture documents that are shared in
the main `spectrochempy` repository.

For normative contracts, see [`../rfcs/INDEX.md`](../rfcs/INDEX.md).

## Conventions

Architecture notes in this directory should normally expose the same basic
entry points near the top of the file:

- a breadcrumb back to `Maintainer Docs` and this index;
- a clear title;
- a short `## Status` section (`Implemented`, `Completed architecture note`,
  or `Architecture reference`);
- optional short companion sections such as `## Date`, `## Scope`, or
  `## Related RFC` when they materially help orientation;
- related RFC or scope links when they materially help orientation.

Use Title Case for status labels. Avoid all-caps forms such as `IMPLEMENTED`
or `ACCEPTED`. Preferred active labels in this directory are:

- `Draft architecture note`
- `Accepted architecture contract`
- `Implemented architecture note`
- `Completed architecture note`
- `Architecture reference`

When a status section contains a follow-up sentence, prefer the same pattern:

- first line: the short status label only;
- next sentence: one plain-language sentence explaining the current role of
  the document.

## Status Definitions

| Status | Meaning |
|---|---|
| `Draft architecture note` | Active architecture note still under maintainer iteration. It may guide ongoing work, but it is not yet the stable shared reference. |
| `Accepted architecture contract` | Architectural decision accepted by maintainers. It is authoritative for direction and constraints, even if some implementation work may still remain. |
| `Implemented architecture note` | Architecture note that describes the current merged and maintained architecture. This is the normal label for a stable current-reference document. |
| `Completed architecture note` | Architecture note for a completed campaign or migration whose main value is to preserve the final architectural outcome. It remains useful, but it is often narrower or more historical than the main current-reference notes. |
| `Architecture reference` | Higher-level, durable reference document that organizes principles or conceptual framing across multiple architecture notes rather than recording a single campaign outcome. |

## Start Here by Topic

| Topic | First document |
|---|---|
| Runtime model for dimensions, coordinates, labels, metadata, provenance | [`metadata-and-support-model.md`](metadata-and-support-model.md) |
| Reader import normalization | [`reader-normalization-architecture.md`](reader-normalization-architecture.md) |
| Portable xarray / NetCDF persistence | [`portable-persistence-model.md`](portable-persistence-model.md) |
| Result objects and result-surface ownership | [`result-object-architecture.md`](result-object-architecture.md) |
| Fit result scientific surface | [`fitresult-scientific-contract.md`](fitresult-scientific-contract.md) |
| Optimize / FitResult boundary | [`optimize-fitresult-architecture.md`](optimize-fitresult-architecture.md) |
| Fitting parser and canonical fitting model boundary | [`adr-fitparameters-role.md`](adr-fitparameters-role.md) |
| Canonical DSL role and guarantees | [`canonical-dsl-contract.md`](canonical-dsl-contract.md) |
| Display architecture | [`display-architecture.md`](display-architecture.md) |

## Core Maintained Architecture

These documents are the primary tracked references for current maintained
behavior and durable architecture boundaries.

| File | Description |
|---|---|
| [`adr-fitparameters-role.md`](adr-fitparameters-role.md) | Architecture contract defining `FitParameters` as parser AST and compatibility layer, not the permanent canonical fitting model representation. |
| [`canonical-dsl-contract.md`](canonical-dsl-contract.md) | Architecture note defining the fitting DSL as a semantic interchange format and clarifying its guarantees and non-guarantees. |
| [`coordset-storage-architecture.md`](coordset-storage-architecture.md) | Final tracked architecture for the completed `CoordSet` storage redesign. |
| [`display-architecture.md`](display-architecture.md) | Final post-migration display architecture and semantic HTML model. |
| [`fitresult-scientific-contract.md`](fitresult-scientific-contract.md) | Primary architecture reference for the maintained scientific surface of `FitResult`. |
| [`framework-principles.md`](framework-principles.md) | Higher-level architectural and philosophical reference for how SpectroChemPy reasons about its framework model. |
| [`metadata-and-support-model.md`](metadata-and-support-model.md) | Primary architecture reference for dimensions, coordinates, labels, metadata, provenance, ownership, and persistence boundaries. |
| [`optimize-fitresult-architecture.md`](optimize-fitresult-architecture.md) | Primary architecture reference for the current boundary between `Optimize` solver state and `FitResult` scientific interpretation. |
| [`portable-persistence-model.md`](portable-persistence-model.md) | Primary architecture reference for the currently implemented portable xarray / NetCDF persistence surface. |
| [`reader-normalization-architecture.md`](reader-normalization-architecture.md) | Primary architecture reference for normalizing imported reader information into the SpectroChemPy runtime model. |
| [`result-object-architecture.md`](result-object-architecture.md) | Implemented tracked architecture note for result objects, ownership, serialization boundary, and display scope. |
| [`tensor-plugin-migration.md`](tensor-plugin-migration.md) | Durable note on the maintained core/plugin boundary after the tensor migration. |

## Historical But Still Useful

These documents remain useful to understand why current architecture looks the
way it does, even when they are not the primary active design surface anymore.

| File | Description |
|---|---|
| [`analysis-fit-result-architecture.md`](../rfcs/analysis-fit-result-architecture.md) | Historical conceptual background for the Result architecture family. |

## Scope Note

This public subset intentionally excludes:

- private or immature assistant architecture work;
- private NMR campaign material and roadmap planning;
- active draft notes that are still moving too quickly;
- detailed audit evidence and implementation logs.
