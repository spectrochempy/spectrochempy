# Metadata and Support Model

## Status

IMPLEMENTED — The model described here reflects the current maintained
architecture after the CoordSet storage redesign, metadata taxonomy adoption,
portable metadata subset implementation, and reader normalization alignment.

## Overview

This note is the primary maintainer reference for the current SpectroChemPy
model around:

- dimensions;
- coordinates;
- `CoordSet`;
- labels;
- metadata;
- provenance and history.

The maintained architecture is runtime-first.

`NDDataset` is the primary scientific object.
Dimensions provide structural axis identity.
Coordinates provide support descriptions attached to those dimensions.
`CoordSet` organizes coordinate ownership and lifecycle.
Metadata provides scientific identity, provenance, presentation, and extension
context.
Labels remain a distinct support-local semantic surface.

This note describes the current maintained model. It does not reproduce RFC
discussion history or rejected alternatives.

For the currently implemented portable xarray / NetCDF persistence surface,
read [`portable-persistence-model.md`](portable-persistence-model.md) first.

## Dimensions

In SpectroChemPy, a dimension is:

```text
an ordered structural axis identifier
```

Dimensions are fundamental for:

- axis count;
- axis order;
- axis identity across reshape-like or reorder-like operations;
- coordinate attachment;
- portable schema identity.

Dimensions are not:

- coordinate value containers;
- full alignment keys in the xarray sense;
- categorical indexes in the pandas sense;
- scientific meaning by themselves.

The source of truth for base dimension count, names, and order lives in the
host array object. In practice, `NDDataset` is the main scientific carrier of
full dimensional semantics because it combines dims with coordinates,
operation-level behavior, and persistence participation.

## Coordinates

`Coord` is the runtime support object attached to one dimension.

Coordinates may carry:

- support values;
- units;
- axis-local titles and names;
- labels;
- coordinate-local metadata;
- selected display and normalization state.

Coordinates are primarily support objects, not signal-bearing scientific data
objects.

Coordinates describe where dataset values live along a dimension. They do not
own the dimension itself. They attach to a structural dimension that already
exists on the host dataset.

Coordinate semantics remain narrower than a full alignment model. SpectroChemPy
uses coordinates to enrich support meaning, not to replace the structural role
of dimensions.

## CoordSet

`CoordSet` is the owned coordinate container of `NDDataset`.

Its responsibilities are:

- coordinate ownership relative to dataset dimensions;
- default coordinate selection;
- same-dimension coordinate grouping;
- reference relationships where applicable;
- coordinate lifecycle across slicing, reduction, interpolation, reshape,
  concatenation, transpose, and mutation paths.

`CoordSet` does not create dimensions independently.

It organizes coordinates relative to dimensions that the dataset already owns.

The current maintained architecture is the post-storage-redesign model:

- `_storage` is the runtime coordinate container;
- group projection is the semantic reference model;
- public behavior remains centered on coordinates, names, titles, labels,
  units, sizes, defaults, and same-dimension semantics.

For maintainers, `CoordSet` is the support-organization layer that prevents
coordinates from collapsing into a simplistic “one axis, one coordinate array”
model.

## Labels

Labels are a distinct semantic surface.

They are not reducible to:

- dimensions;
- coordinates alone;
- generic metadata;
- pure display hints.

Labels are support-local content that may serve as:

- identifiers;
- annotations;
- categories;
- display-facing names or substitutions.

Labels usually live on coordinates rather than on the dataset signal itself.
`NDDataset` disables direct data labels and relies on coordinate-attached
labels for user-facing labelled support.

Labels may coexist with numeric support values or appear in label-only
coordinates. They do not define dimensional identity, and they should not be
treated as the primary structural key of an axis.

## Metadata Taxonomy

The maintained metadata taxonomy distinguishes:

- scientific identity metadata;
- structural metadata;
- provenance metadata;
- presentation metadata;
- extension / private metadata.

### Scientific identity metadata

Scientific identity metadata explains what an object is scientifically.

Typical examples:

- `title`
- `description`
- scientific descriptors in `meta`
- units when they express scientific meaning

### Structural metadata

Structural metadata explains how an object is organized.

Typical examples:

- `dims`
- coordinate attachment
- `coordset`
- same-dimension grouping semantics
- mask/topology-related state

### Provenance metadata

Provenance metadata explains source, lineage, authorship, and source context.

Typical examples:

- `author`
- `origin`
- `filename`
- `created`
- `modified`
- `history`

### Presentation metadata

Presentation metadata explains how an object should be displayed or rendered.

Typical examples:

- display-oriented titles in some contexts;
- coordinate display state;
- formatting-oriented hints.

### Extension / private metadata

Extension or private metadata captures useful payloads that should be retained
without being promoted to the normalized core contract.

Typical examples:

- reader-specific payloads;
- plugin-specific payloads;
- imported technical metadata without a stable typed home.

## Provenance and History

Provenance and history are related but distinct.

Provenance describes:

- where an object came from;
- who created or supplied it;
- what source lineage remains attached;
- when source or import events occurred.

History is the explicit event trail attached to the object during its
lifecycle.

The maintained distinction is:

```text
provenance = source and lineage context
history    = explicit event trail
```

`history` contributes to provenance, but provenance is broader than `history`.

Temporal information needs special care:

- time is provenance when it describes acquisition session, creation,
  modification, or import lineage;
- time is support geometry when it locates points along a dataset axis;
- one source may legitimately supply both kinds of time.

## Ownership Model

### `NDArray`

`NDArray` owns the broad base array surface:

- data storage;
- dimensions;
- units;
- generic metadata;
- direct labels for compatible base-array cases.

It is not the primary owner of the full scientific support/provenance model.

### `Coord`

`Coord` owns one support description attached to one dimension:

- support values;
- axis-local units;
- axis-local titles/names;
- coordinate-local labels;
- coordinate-local metadata.

It may carry provenance-adjacent content, but it is not the primary owner of
dataset-level provenance.

### `CoordSet`

`CoordSet` owns coordinate organization:

- default selection;
- same-dimension sibling structure;
- coordinate references;
- support lifecycle relative to dimensions.

It does not own arbitrary free-form metadata or the dataset provenance surface.

### `NDDataset`

`NDDataset` is the primary owner of the scientific object model.

It owns:

- signal data;
- dimensions as carried by the host array object;
- `CoordSet`;
- dataset-level scientific identity metadata;
- dataset-level provenance and history;
- dataset-level `meta` payloads.

### `Project`

`Project` is a typed ownership hierarchy for datasets and nested subprojects.

It owns:

- container structure;
- parent/child ownership invariants;
- project-level context through its own metadata surface.

It aggregates child provenance but does not replace the provenance owned by
member datasets.

### Result objects

Result objects own:

- estimator identity;
- parameters;
- outputs;
- diagnostics.

They are not yet the primary normalized provenance/metadata owner in the same
way as `NDDataset`. Provenance usually remains visible through output datasets
and result context rather than through a fully separate result metadata
taxonomy.

## Runtime vs Persistence

### Runtime model

The runtime model is the richest form of this architecture.

It supports:

- structural dimensions;
- rich coordinates and `CoordSet` grouping;
- coordinate labels;
- typed metadata fields;
- extensible `Meta`;
- provenance/history;
- support-local and dataset-level context.

### Native persistence

Trusted native persistence preserves the SpectroChemPy runtime model as far as
the native contract allows.

That includes:

- dataset-level metadata and provenance/history;
- coordinate structure and labels;
- project hierarchy;
- richer runtime payloads that do not need to fit an interchange schema.

### Portable persistence

Portable persistence preserves a narrower maintained subset.

It should preserve:

- dimension names and order;
- default coordinate structure;
- accepted auxiliary coordinate structure where supported;
- portable typed metadata fields;
- portable labels only within the accepted subset;
- JSON-compatible metadata and reconstruction attrs needed by the portable
  mapping.

Portable persistence does not define the runtime model. It projects a subset of
it into an interchange-oriented schema.

## Maintainer Guidance

- Treat dimensions as structural axis identifiers first.
- Treat coordinates as support objects attached to those dimensions.
- Treat `CoordSet` as the ownership and lifecycle layer for support.
- Treat labels as support-local semantics, not as structural identity.
- Use the metadata taxonomy to decide whether information belongs to scientific
  identity, structure, provenance, presentation, or extension/private payloads.
- Keep provenance and history distinct: provenance is lineage context; history
  is the explicit event trail.
- Apply the dual-time rule: `acquisition_date` records session/lineage
  provenance; timestamp coordinates record observation-level support geometry.
  One source may legitimately supply both, and they should coexist in the
  normalized runtime model.
- Read this note first for support/metadata architecture questions, then use
  the underlying RFCs for decision history and finer contract boundaries.

## Reader Alignment Status

Reader normalization alignment is COMPLETED for the maintained campaign scope.

### Wave 1

| Reader | acquisition_date | origin | history | Status |
| ------ | --------------- | ------ | ------- | ------ |
| OMNIC SPG/SPA/SRS | Set from parsed timestamps | `omnic` for all variants | Import event + vendor history when available | ALIGNED |
| OPUS | Set from parameter timestamp | `opus-<type>` (e.g. `opus-AB`) | Import event | ALIGNED |
| JCAMP-DX | Set from earliest LONGDATE/TIME | Preserved `##ORIGIN`; deterministic multi-origin join | Import event + sort event | ALIGNED |
| LabSpec TXT | Set from parsed acquisition start | `labspec` | Import event | ALIGNED |
| CSV (generic + OMNIC) | OMNIC CSV parses date from filename | Caller-specified or OMNIC-flavored | Import event | ALIGNED |
| TopSpin (plugin) | Set from vendor DATE | `topspin` | Import event | ALIGNED |

### Wave 2

| Reader | acquisition_date | origin | history | Status |
| ------ | --------------- | ------ | ------- | ------ |
| WiRE | Set from first session timestamp | application/version per reader convention | Import event | ALIGNED |
| Quadera | Set from first acquisition timestamp | `quadera` | Import event | ALIGNED |
| SOC | Inherited OMNIC-style acquisition provenance | `soc` | SOC-specific import event appended while preserving inherited provenance | ALIGNED |
| MATLAB/DSO | DSO acquisition provenance promoted when available | `matlab` / `dso` | Import event; DSO vendor history preserved | ALIGNED |
| SPC | Set from parsed header/session timestamp | `thermo galactic` | Import event | ALIGNED |

Semantic characterization baselines and provenance assertions now exist for the
targeted wave-1 and wave-2 readers listed above.

Topics intentionally left outside this completed campaign include:

- Carroucell temperature semantics;
- vendor-log classification refinements such as SPC `LOGSTC`;
- wording harmonization of history messages;
- additional vendor-specific metadata promotion work.

For the complete reader normalization contract and per-reader destination rules,
see [`reader-normalization-architecture.md`](reader-normalization-architecture.md).
