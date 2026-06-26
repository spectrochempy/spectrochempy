[Maintainers](../../README.md) · [RFCs](../INDEX.md)

# Metadata Taxonomy Contract

## Status

Accepted Maintainer RFC.

This document is conceptual and normative in intent.

It defines the maintainer-level taxonomy for metadata in SpectroChemPy:

- what kinds of metadata exist;
- who owns them;
- how they should propagate in principle;
- what belongs in runtime-only, native, and portable persistence layers.

It builds on:

- [`dimensional-semantics-contract.md`](dimensional-semantics-contract.md)
- [`coordinate-and-coordset-semantics.md`](coordinate-and-coordset-semantics.md)

It does not revisit or reopen those decisions.

It does not define implementation details, migration sequencing, or API work.

The key words MUST, SHOULD, and MAY are to be interpreted as normative
requirements for maintainers and future contributors.

## 1. Purpose

This RFC answers four linked questions:

```text
What kinds of metadata exist in SpectroChemPy?
Who owns them?
How should they propagate?
What belongs in portable persistence?
```

This contract is defined from the SpectroChemPy runtime object model first.

Its main architectural reference points are:

- SpectroChemPy's own scientific object model;
- the dimensional and coordinate contracts already established;
- the existing metadata propagation contract draft;
- the trusted-versus-portable persistence boundary.

Inside scope:

- the current metadata landscape;
- a proposed taxonomy of metadata categories;
- ownership boundaries across core object types;
- category-level propagation philosophy;
- persistence boundaries for those categories;
- taxonomy boundaries around labels.

Outside scope:

- implementation changes;
- migration plans;
- arithmetic redesign;
- dimensional semantics;
- coordinate semantics;
- label semantics in full;
- reader-specific normalization rules in detail;
- a full provenance graph design.

## 2. Part 1 — Current Metadata Landscape

This section describes current behavior, not intended behavior.

Metadata in SpectroChemPy already exists in several forms.

## 2.1 Typed metadata fields

The runtime object model already uses explicit typed fields for a substantial
part of its metadata surface.

Observed examples include:

- `name`
- `title`
- `units`
- `dims`
- `mask`
- `transposed`
- `description`
- `author`
- `origin`
- `filename`
- `created`
- `modified`
- `history`

Some of these are clearly scientific or structural.
Some are clearly provenance-oriented.
Some blend presentation with science.

## 2.2 `Meta` payloads

`Meta` exists as the main extensible key/value metadata container.

Current behavior shows that `Meta` is used for:

- user-supplied scientific context;
- reader-imported payloads that do not map to typed fields;
- extension or plugin payloads;
- miscellaneous retained metadata that is meaningful but not yet normalized.

`Meta` is therefore not the only metadata system, but it is the main
extensible metadata reservoir.

## 2.3 Provenance fields

The runtime model already has dedicated provenance-oriented fields, especially
on `NDDataset`.

These include:

- `author`
- `origin`
- `filename`
- `created`
- `modified`
- `history`

This is important because provenance is not just an arbitrary `Meta` payload in
current architecture. It already has first-class typed representation.

## 2.4 Coordinate-local metadata

Metadata also lives locally on support objects.

`Coord` currently carries:

- `name`
- `title`
- `units`
- `labels`
- `meta`
- linearization and rounding state
- display-oriented axis behavior

`CoordSet` does not have its own `Meta` container, but it owns structured
coordinate-group metadata such as:

- default selection;
- same-dimension grouping;
- references;
- grouping state tied to coordinate lifecycle.

This means some metadata is attached to support descriptions rather than to the
dataset as a whole.

## 2.5 Presentation-oriented metadata

Some metadata already functions primarily as presentation or rendering context.

Examples include:

- display-oriented `title` use in some contexts;
- coordinate rounding and linearization state;
- coordinate reversal/display conventions;
- formatting-related hints embedded in coordinate state;
- attrs or markers used mainly to aid a specific textual or graphical view.

These fields may overlap with scientific interpretation, but current behavior
already shows a meaningful presentation-oriented stratum.

## 2.6 Persistence metadata

Persistence introduces another metadata layer.

Current examples include:

- native `_attributes_()` lists and JSON encoding helpers;
- `scpy_*` attrs used in xarray/NetCDF portable reconstruction;
- dataset-level attrs that describe primary variables, mask variables,
  coordinate roles, and reconstruction markers.

This metadata is often not scientific content in the same sense as user-facing
scientific descriptors. It is schema or reconstruction metadata.

## 3. Part 2 — Metadata Categories

The project already behaves as if it recognizes several kinds of metadata.

This RFC makes those categories explicit.

## 3.1 Scientific Identity Metadata

Scientific identity metadata describes what a scientific object is, how it
should be interpreted, or what scientific meaning the user intends to preserve.

Typical examples include:

- `title`
- `description`
- data units
- scientific descriptors stored in `meta`
- acquisition meaning or domain interpretation
- coordinate-local scientific titles or axis meanings

Scientific identity metadata SHOULD answer questions such as:

- what is this dataset or coordinate scientifically;
- what does this signal represent;
- what does this axis mean;
- which user-supplied scientific context should survive normal work.

## 3.2 Structural Metadata

Structural metadata describes how a scientific object is organized.

Typical examples include:

- `dims`
- coordinate attachment
- `coordset`
- default coordinate selection
- same-dimension grouping semantics
- mask structure
- transposed/representation state where it affects object structure

Structural metadata is not primarily about scientific narrative meaning. It is
about object topology, support attachment, and internal geometry.

The dimensional and coordinate contracts already define core parts of this
category. This RFC only classifies them.

## 3.3 Provenance Metadata

Provenance metadata describes source, authorship, lineage, and transformation
history.

Typical examples include:

- `author`
- `origin`
- `filename`
- `created`
- `modified`
- `history`

This category is already partly typed in current runtime behavior.

It is conceptually distinct from scientific identity metadata, even when users
sometimes read provenance fields as scientific context.

## 3.4 Presentation Metadata

Presentation metadata describes how an object should be displayed, formatted,
or rendered for human interpretation.

Typical examples include:

- display hints;
- formatting controls;
- rendering preferences;
- coordinate rounding and linearization hints when used for display;
- coordinate reversal/display policy;
- other output-oriented representational hints.

Presentation metadata may influence the user experience strongly, but it is
not usually the primary scientific contract.

## 3.5 Extension / Private Metadata

Extension or private metadata covers payloads that are valid to retain but are
not part of the core cross-project normalized contract.

Typical examples include:

- reader-specific payloads;
- plugin-specific payloads;
- temporary metadata;
- internal experimental payloads;
- imported metadata that has not yet been normalized into a typed field or a
  stable taxonomy category.

This category exists to avoid forcing every useful payload into premature core
normalization.

## 4. Part 3 — Ownership

Metadata ownership is distributed across multiple objects, but not equally.

## 4.1 `NDArray`

`NDArray` is the base owner of several broad metadata surfaces:

- `name`
- `title`
- `units`
- `dims`
- `labels`
- `mask`
- `meta`
- `transposed`

This makes `NDArray` the lowest shared ownership layer for identity,
structure, units, labels, and general metadata.

`NDArray` is therefore the source of truth for some metadata fields, but not
for the full scientific or provenance model.

## 4.2 `Coord`

`Coord` owns coordinate-local metadata for one support description:

- axis-specific `name`
- axis-specific `title`
- axis-specific `units`
- coordinate-local `labels`
- coordinate-local `meta`
- coordinate-local presentation state

`Coord` is the source of truth for metadata attached to one support
description, not for dataset-wide scientific identity or provenance.

## 4.3 `CoordSet`

`CoordSet` owns structural support metadata, not generic `Meta` payloads.

It is the source of truth for:

- default coordinate selection;
- same-dimension grouping semantics;
- coordinate sibling relationships;
- coordinate references;
- grouping/lifecycle state around coordinates.

`CoordSet` does not own arbitrary scientific descriptors or free-form user
payloads.

## 4.4 `NDDataset`

`NDDataset` is the main owner of dataset-level metadata.

It is the source of truth for:

- dataset-level scientific identity metadata;
- dataset-level provenance metadata;
- dataset-level structure through owned `dims` and `coordset`;
- dataset-level user/extensible `meta` payloads where not delegated to a
  support object.

This makes `NDDataset` the main maintainer-level metadata anchor for scientific
objects in current runtime architecture.

## 4.5 `Project`

`Project` owns organizational and container-level metadata.

It may also own `meta`, but it does not become the source of truth for the
internal metadata of its member datasets.

Its metadata role is therefore:

- organizational identity;
- container-level context;
- recursive organization of eligible scientific objects.

## 4.6 Result objects

Result objects currently have their own public state, but do not fully
participate in the same metadata taxonomy as `NDDataset`.

They currently own fields such as:

- estimator identity;
- parameters;
- outputs;
- diagnostics.

These are real semantic fields, but not yet a normalized metadata taxonomy.

For now, result objects SHOULD be understood as owning result-surface metadata
that remains partly distinct from the dataset-centered metadata contract.

## 5. Part 4 — Propagation Semantics

This section does not define implementation details.

It defines the expected propagation philosophy by metadata category.

The four main propagation strategies are:

- preserve
- recompute
- merge
- drop

## 5.1 Scientific identity metadata

Scientific identity metadata SHOULD preserve by default for ordinary
single-source transformations where the scientific object remains the same.

It MAY be recomputed or overridden when:

- the scientific object becomes a genuinely derived object;
- the operation changes domain or interpretation substantially;
- preserving the old identity would become misleading.

Scientific identity metadata SHOULD NOT be silently dropped merely because the
implementation path changed.

## 5.2 Structural metadata

Structural metadata SHOULD follow the geometry and support contracts.

Its dominant strategy is:

- recompute when structure changes;
- preserve when structure truly survives;
- drop when the relevant structure no longer exists.

Structural metadata is the clearest category for recompute-oriented behavior.

## 5.3 Provenance metadata

Provenance metadata usually SHOULD preserve lineage while recording change.

In practice this means provenance categories often need a mixed strategy:

- preserve source-lineage fields where lineage continuity matters;
- recompute fields like modification timestamps;
- generate or append history records;
- merge in bounded ways for multi-source operations when several contributors
  matter scientifically.

Provenance is therefore not a pure preserve or pure recompute category.

## 5.4 Presentation metadata

Presentation metadata SHOULD preserve when the same display intent still makes
sense.

It SHOULD be recomputed or dropped when:

- inherited formatting would become stale or misleading;
- the object's structure no longer supports the old view;
- a new derived object deserves its own display defaults.

This category is generally weaker than scientific identity metadata and easier
to recompute or drop when necessary.

## 5.5 Extension / private metadata

Extension or private metadata SHOULD preserve conservatively by default inside
the runtime model when doing so is safe and not misleading.

However:

- it MUST NOT block clear semantic recomputation of structural metadata;
- it MAY be dropped at portability boundaries;
- it SHOULD NOT be deep-merged blindly across true multi-source operations;
- it MAY remain intentionally unnormalized until a later RFC sharpens it.

## 6. Part 5 — Persistence Boundaries

This taxonomy must work across three layers:

```text
Runtime richness
Native persistence
Portable persistence
```

## 6.1 Runtime richness

Runtime richness is the broadest metadata layer.

It MAY include:

- all typed metadata fields;
- full `Meta` payloads;
- coordinate-local metadata;
- `CoordSet` grouping metadata;
- presentation-oriented state;
- extension or private metadata;
- temporary but meaningful runtime context.

Runtime richness is the primary source of truth for the live object model.

## 6.2 Native persistence

Native persistence SHOULD preserve the full SpectroChemPy-owned metadata model
as far as the native format contract allows.

Metadata that should always persist natively includes:

- core scientific identity metadata;
- structural metadata needed for faithful reconstruction;
- provenance metadata;
- coordinate-local metadata needed for faithful reconstruction.

Metadata that usually persists natively includes:

- user `Meta` payloads;
- presentation-oriented state where it is part of current runtime meaning;
- extension/private payloads that the native format can safely round-trip.

Runtime-only metadata MAY remain non-persistent natively if it depends on:

- live process state;
- transient caches;
- ephemeral rendering internals;
- unstable experimental infrastructure.

## 6.3 Portable persistence

Portable persistence is narrower by design.

Metadata that portable formats SHOULD always persist includes:

- structural metadata required to reconstruct dimensions and coordinate
  attachment;
- core scientific identity metadata needed for scientific interpretation;
- textual units for data and exported coordinates;
- bounded provenance metadata when it belongs to the portable scientific
  record;
- required schema/reconstruction metadata such as `scpy_*` attrs.

Metadata that portable formats SHOULD usually persist includes:

- coordinate titles;
- description fields;
- selected JSON-compatible `Meta` payloads;
- default-versus-auxiliary coordinate role markers;
- textual history where it belongs to a portable record.

Metadata that MAY remain runtime-only or native-only includes:

- full `CoordSet` grouping richness;
- alias behavior;
- internal reference topology;
- arbitrary plugin payloads with no portable schema;
- private or temporary reader payloads;
- presentation details that do not belong to portable scientific meaning.

Portable persistence should preserve the SpectroChemPy metadata contract as far
as the carrier schema allows. The carrier does not define the taxonomy.

## 7. Part 6 — Relationship to Labels

This RFC does not fully define label semantics.

It only establishes taxonomy boundaries.

Labels do not fit cleanly into one single metadata category in current
behavior.

The most accurate current taxonomy boundary is:

- labels are not pure structural metadata, even though they are attached to
  support structure;
- labels are not pure scientific identity metadata, even though they can
  contribute to interpretation;
- labels are not merely presentation metadata, even though they affect display.

For taxonomy purposes, labels SHOULD currently be treated as a distinct
support-local semantic surface that intersects:

- coordinate-local metadata;
- scientific interpretation;
- presentation and annotation use cases.

The future Label Semantics Contract should sharpen this boundary rather than
burying labels inside a generic metadata bucket.

## 8. Part 7 — Comparison

This section is comparative only.

The taxonomy above is defined from SpectroChemPy runtime semantics, not derived
from external libraries or formats.

## 8.1 xarray attrs

xarray largely concentrates extensible metadata in attrs attached to datasets,
variables, and coordinates.

SpectroChemPy intentionally differs because:

- it already has many typed metadata fields;
- it distinguishes structural dimensions from support coordinates explicitly;
- it uses `Meta` as only one part of a broader metadata landscape;
- it carries more runtime-specific grouping semantics than plain attrs alone
  can explain.

The nearest xarray analogue to SpectroChemPy extension/private metadata is
therefore `attrs`, but SpectroChemPy's taxonomy is broader than an attrs-only
model.

## 8.2 pandas metadata conventions

pandas generally keeps metadata conventions lighter and more object-local.

SpectroChemPy differs because its scientific objects require:

- explicit units;
- explicit support coordinates;
- typed provenance fields;
- richer structural metadata.

This makes SpectroChemPy's taxonomy more formal than the light metadata
conventions typical of many pandas workflows.

## 8.3 Scientific exchange formats

Scientific exchange formats usually prefer:

- explicit structural schema;
- portable scientific descriptors;
- bounded provenance;
- strong restrictions on private runtime state.

SpectroChemPy agrees with that boundary for portable persistence.

It intentionally differs by preserving a richer runtime and native metadata
layer that does not need to be fully portable.

## 9. Part 8 — Explicit Non-Goals

This RFC does not:

- redefine dimensional semantics;
- redefine coordinate semantics;
- redefine coordinate arithmetic semantics;
- define algorithm-specific result semantics;
- define a full provenance graph model;
- define reader normalization rules in detail;
- define portable schema mechanics;
- define implementation or migration sequencing.

It also does not require every current `Meta` payload to be immediately
normalized into a typed field.

## 10. Part 9 — Candidate Follow-Up RFCs

### Label Semantics Contract

Dependency:

- should build on this taxonomy rather than reinvent metadata categories.

Reason:

- labels currently straddle support-local, scientific, and presentation roles.

### Reader Metadata Normalization Contract

Dependency:

- should build on this taxonomy to decide which imported payload belongs in
  typed fields, `Meta`, coordinate-local metadata, or runtime-only extension
  space.

Reason:

- reader-local metadata is one of the clearest current fragmentation points.

### Provenance & History Contract

Dependency:

- should build on this taxonomy's provenance category and ownership model.

Reason:

- provenance already has typed fields, but history, timestamps, and lineage
  policy still need sharper semantics.

## 11. Part 10 — Promotion Candidates

### Suitable for architecture notes

Implemented as:

- [`maintainers/architecture/metadata-and-support-model.md`](../architecture/metadata-and-support-model.md)
  documents the metadata taxonomy adopted from this RFC family.

Remaining future work:

- `maintainers/architecture/provenance-and-history-semantics.md`
  Durable reference note once provenance/history boundaries are better
  characterized.

### Suitable for future RFCs

- `maintainers/rfcs/label-semantics.md`
- `maintainers/rfcs/reader-metadata-normalization-contract.md`
- `maintainers/rfcs/provenance-and-history-contract.md`

### Suitable to remain historical only

- campaign-local inventories of stale reader payloads once normalization rules
  are defined;
- transient implementation details about internal helper placement;
- historical notes about removed runtime fields such as `roi` and
  `modeldata` once their semantic removal decisions are fully absorbed into
  maintained docs.
