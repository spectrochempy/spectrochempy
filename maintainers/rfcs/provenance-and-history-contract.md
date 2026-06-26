[Maintainers](../../README.md) · [RFCs](../INDEX.md)

# Provenance and History Contract

## Status

Accepted Maintainer RFC.

This document is conceptual and normative in intent.

It defines the maintainer-level contract for:

- what provenance is;
- what history is;
- who owns them;
- how they should propagate;
- what belongs in persistence.

It builds on:

- [`metadata-taxonomy-contract.md`](metadata-taxonomy-contract.md)
- [`reader-metadata-normalization-contract.md`](reader-metadata-normalization-contract.md)
- [`label-semantics-contract.md`](label-semantics-contract.md)

It must not reopen those decisions.

It does not define implementation details, migration sequencing, or API work.

The key words MUST, SHOULD, and MAY are to be interpreted as normative
requirements for maintainers and future contributors.

## 1. Purpose

SpectroChemPy already exposes provenance-like concepts such as:

- `author`
- `origin`
- `filename`
- `created`
- `modified`
- `history`

But their boundaries are only partly explicit.

The project now needs a stable answer to five questions:

```text
What is provenance?
What is history?
Who owns them?
How should they propagate?
What belongs in persistence?
```

Inside scope:

- current runtime provenance/history behavior;
- the conceptual distinction between provenance and history;
- ownership boundaries across core objects;
- the role of acquisition time and other temporal information;
- propagation philosophy across major operation families;
- interaction with readers and persistence layers.

Outside scope:

- implementation changes;
- migration plans;
- structured workflow provenance graphs;
- parser details;
- metadata taxonomy redesign;
- label, coordinate, or dimensional redesign.

## 2. Part 1 — Current Landscape

This section describes current behavior, not intended behavior.

### 2.1 `NDArray`

`NDArray` is not the primary provenance owner.

Its current base responsibilities are broader array metadata such as:

- `name`
- `title`
- `units`
- `meta`
- `labels`
- `dims`

It does not appear to own the main provenance surface in current runtime
architecture.

### 2.2 `Coord`

`Coord` may carry support-local metadata and labels, but it does not currently
own an explicit first-class provenance surface parallel to `NDDataset`.

Current readers do, however, encode provenance-adjacent material inside
coordinates:

- acquisition timestamps as numeric coordinate values;
- acquisition dates, filenames, or sample names as coordinate labels;
- axis-local imported support identity.

This means provenance-related meaning sometimes appears in coordinates even when
the coordinate object is not the formal provenance owner.

### 2.3 `NDDataset`

`NDDataset` is currently the main runtime owner of typed provenance-like
fields.

Observed current fields include:

- `author`
- `origin`
- `filename`
- `created`
- `modified`
- `history`
- `acquisition_date` as a separate but not fully integrated temporal field

These fields are part of the dataset save/load attribute surface and are
displayed in dataset summaries.

`history` is stored internally as dated entries and presented as a formatted
list of strings.

### 2.4 `Project`

`Project` does not currently mirror the full dataset provenance surface as
typed first-class fields.

Observed current behavior:

- project-level author-like information may appear in `Project.meta`;
- project persistence is native and recursive;
- copy semantics are defined for ownership and structural isolation, not as a
  provenance model.

This means `Project` participates in provenance indirectly, but not through the
same explicit typed field set as `NDDataset`.

### 2.5 Result objects

Result objects currently describe:

- estimator identity;
- parameters;
- outputs;
- diagnostics.

Their own docstring mentions “identity” and “provenance”, but current runtime
state does not expose a typed provenance surface analogous to `NDDataset`.

In practice, result provenance currently lives mostly through:

- the datasets stored in outputs;
- estimator identity;
- diagnostics/parameters;
- history carried by produced datasets where applicable.

### 2.6 Reader imports

Readers already import provenance information, but not always consistently.

Observed current patterns include:

- `filename` from the opened path;
- `origin` from format or reader identity;
- `history` from import messages or vendor processing logs;
- `author` and date in the MATLAB reader;
- acquisition timestamps stored as support coordinates;
- filenames, dates, and acquisition names stored in coordinate labels in OMNIC,
  OPUS, JCAMP, and related readers.

### 2.7 Native persistence

Trusted native persistence already preserves much of the dataset provenance
surface:

- typed provenance fields on `NDDataset`;
- history entries;
- project-owned datasets and subprojects recursively;
- coordinate content carrying provenance-adjacent support information.

### 2.8 Portable persistence

Portable persistence is narrower.

Current and proposed portable work already assumes:

- some dataset-level metadata such as `author`, `origin`, and `history` may be
  portable when JSON-compatible;
- coordinate and label portability is narrower than full runtime richness;
- portable persistence is not yet a full provenance-graph model.

### 2.9 Current inconsistencies

Current inconsistencies include:

- `history` is often treated as the most explicit provenance carrier, but
  provenance also lives in `origin`, `author`, `filename`, and timestamps;
- acquisition time may appear as provenance, support geometry, or both;
- readers map equivalent temporal/source semantics differently;
- operation families append, rewrite, or synthesize history inconsistently;
- project-level provenance is partly implicit in `meta` rather than expressed
  as a dedicated maintained contract;
- result objects do not yet have an explicit provenance contract of their own.

## 3. Part 2 — Provenance vs History

This section answers:

```text
What is provenance?
What is history?
```

### 3.1 Provenance

Provenance is the metadata that explains:

```text
where a scientific object came from,
who created or supplied it,
when it was created or imported,
and what source lineage remains attached to it
```

Provenance therefore includes fields such as:

- `author`
- `origin`
- `filename`
- `created`
- `modified`

and may also include imported acquisition/source timestamps when they describe
lineage rather than support geometry.

### 3.2 History

History is the ordered record of transformation or import events attached to an
object during its lifecycle.

Typical examples include:

- import logs;
- processing records;
- transform logs;
- operation trail entries.

In current runtime architecture, `history` is the explicit event log surface.

### 3.3 Normative distinction

Provenance and history are:

- not one concept;
- not fully separate unrelated concepts;
- related but distinct concepts.

The normative position of this RFC is:

```text
Provenance = source and lineage context
History    = explicit event trail
```

History is one important carrier of provenance, but provenance is broader than
history.

### 3.4 Field classification

The maintained classification is:

- `author` belongs to provenance.
- `origin` belongs to provenance.
- `filename` belongs to provenance.
- `created` belongs to provenance.
- `modified` belongs to provenance.
- `history` belongs to history first, and contributes to provenance second.

This distinction matters because propagation rules need not be identical.

## 4. Part 3 — Ownership

This section answers:

```text
Who owns provenance?
Who owns history?
```

### 4.1 `NDArray`

`NDArray` is not the primary source of truth for provenance or history.

It may carry metadata that survives into higher-level objects, but the main
maintained provenance contract does not originate here.

### 4.2 `Coord`

`Coord` is not the primary owner of provenance or history.

However, `Coord` may carry support-local information with provenance-adjacent
meaning, especially imported temporal or identifying material.

This does not turn `Coord` into the authoritative provenance owner.

### 4.3 `CoordSet`

`CoordSet` owns support organization, not provenance.

It may contain coordinates whose values or labels encode provenance-adjacent
content, but that content remains support-local unless separately promoted into
dataset-level provenance fields.

### 4.4 `NDDataset`

`NDDataset` is the primary source of truth for dataset-level provenance and
history.

It owns:

- typed provenance fields;
- the explicit history trail;
- the relationship between dataset identity and dataset lineage;
- the persisted runtime provenance surface for scientific datasets.

### 4.5 `Project`

`Project` owns container-level context and may aggregate provenance, but it is
not the source of truth for the internal provenance of member datasets.

Its role is:

- to preserve container context;
- to preserve recursive membership;
- to aggregate child objects that each retain their own provenance.

### 4.6 Result objects

Result objects own result-surface context such as estimator identity,
parameters, outputs, and diagnostics.

They do not yet own a first-class maintained provenance/history surface
equivalent to `NDDataset`.

For now, the maintained position is:

- result provenance is partly represented by output datasets and estimator
  context;
- result objects are not yet the primary owner of a normalized provenance
  contract.

## 5. Part 4 — Acquisition Time and Temporal Information

This is one of the central ambiguity points.

### 5.1 When is time provenance?

Time is provenance when it describes source lineage, such as:

- file creation/import time;
- acquisition session time attached to the dataset as a whole;
- processing or derivation time;
- object creation or modification timestamps.

In these cases, time answers:

```text
when did this object come into existence
or when was it acquired/processed as a source event
```

### 5.2 When is time support geometry?

Time is support geometry when it locates data points along an axis.

Examples:

- time series x/y support;
- elapsed acquisition time for repeated measurements;
- timestamp arrays used as coordinate values for one dimension.

In these cases, time answers:

```text
where along the support axis does this point lie
```

### 5.3 Can time be both?

Yes.

A single scientific workflow may involve both:

- a provenance-level acquisition time for the dataset or run;
- support-level time coordinates locating individual observations.

The maintained position is:

```text
time provenance and time support are conceptually distinct
even when derived from the same source material
```

### 5.4 What should readers do?

Readers SHOULD normalize imported temporal information according to semantic
role:

- use provenance fields when the time describes source/session lineage;
- use coordinates when the time describes support geometry;
- use labels only when imported temporal values are support-local identifiers
  or annotations and do not fit better as primary coordinate values.

Readers MAY preserve both a support coordinate and provenance information when
the source truly carries both meanings.

They SHOULD NOT collapse all imported times into one representation merely for
convenience.

## 6. Part 5 — Propagation Philosophy

This section defines conceptual guidance, not algorithms.

The main propagation strategies are:

- preserve
- recompute
- merge
- drop

### 6.1 Provenance metadata

#### Arithmetic and ordinary single-source transforms

Provenance SHOULD generally `preserve` and `extend`.

The source lineage remains relevant, and the operation SHOULD record a new
history event rather than replacing provenance wholesale.

#### Reductions

Provenance SHOULD generally `preserve`.

If the reduction produces a derived quantity, provenance still links the result
to its source even when identity changes.

#### Slicing

Provenance SHOULD generally `preserve` and `extend`.

A slice is still traceable to its source object.

#### Concatenation and stacking

Provenance SHOULD generally `merge` or `synthesize`.

Multi-source results should not pretend to inherit only one source lineage.

#### Interpolation and representation changes

Provenance SHOULD generally `preserve` and `extend`.

The representation changes, but the source lineage remains attached.

#### Copy

Provenance SHOULD generally `preserve`.

A copy is not a new source event by itself. The copied object remains derived
from the same lineage, even if new container ownership or new detached roots
exist.

#### Serialization round-trips

Round-trips SHOULD generally `preserve` provenance as far as the persistence
contract allows.

Native persistence should preserve richer provenance than portable persistence.

### 6.2 History metadata

#### Arithmetic and ordinary single-source transforms

History SHOULD generally `extend`.

#### Reductions

History SHOULD generally `extend`, though some derived-object cases may
rephrase or synthesize the newest entry.

#### Slicing

History SHOULD generally `extend`.

#### Concatenation and stacking

History SHOULD generally `synthesize` or `merge`.

The key requirement is that multi-source derivation remains visible.

#### Interpolation

History SHOULD generally `extend`.

#### Copy

History SHOULD generally `preserve`.

Copying alone should not fabricate a new scientific processing event unless a
future explicit API chooses to expose one.

#### Serialization round-trips

History SHOULD generally `preserve` as textual event content, subject to the
portability limits of the persistence layer.

### 6.3 Summary rule

The maintained rule is:

```text
Preserve lineage broadly.
Extend history for new single-source events.
Synthesize lineage/history for multi-source results.
Do not drop provenance merely because identity changes.
```

## 7. Part 6 — Project and Result Objects

### 7.1 Does a `Project` own provenance?

A `Project` owns container-level context, but not the full internal provenance
of each member.

Its provenance role is therefore limited and aggregative.

### 7.2 Does a `Project` aggregate provenance?

Yes, conceptually.

A `Project` aggregates child objects that retain their own provenance, and it
may also carry project-level contextual metadata such as author-like
information in `meta`.

But it should not erase or replace member provenance.

### 7.3 Do Results inherit provenance?

Results may inherit provenance conceptually from source datasets and estimator
context, but they do not yet expose a dedicated normalized provenance contract
equivalent to `NDDataset`.

The maintained position is:

- output datasets produced by analyses SHOULD preserve or synthesize provenance
  according to dataset rules;
- result wrapper objects SHOULD be understood as provenance-adjacent but not
  yet the primary standardized provenance surface.

### 7.4 Conceptual model

The maintained conceptual model is:

```text
NDDataset
    primary owner of dataset provenance/history

Project
    container-level context + aggregation of child provenance

Result objects
    method/result context with partial provenance through outputs and estimator
    identity, pending any later dedicated result-provenance contract
```

## 8. Part 7 — Reader Interaction

Using the Reader Metadata Normalization Contract:

```text
When should imported information
become provenance?
```

Imported information SHOULD become provenance when it records:

- source file identity;
- acquisition session lineage;
- author/supplier identity;
- import/process origin;
- source or processing timestamps as lineage metadata;
- vendor processing history.

Imported information SHOULD become history when it is best represented as a
chronological event trail rather than a static source descriptor.

Imported information SHOULD become support coordinates instead when it is
primarily geometry, and SHOULD become labels when it is support-local
annotation or identification.

## 9. Part 8 — Persistence

### 9.1 Native persistence

Trusted native persistence SHOULD preserve the dataset provenance/history
surface as part of the native runtime model.

This includes:

- typed provenance fields;
- history entries;
- project aggregation through recursive persistence;
- support-local coordinates/labels carrying provenance-adjacent context.

### 9.2 Portable persistence

Portable persistence SHOULD preserve only the provenance/history subset that
fits the portable contract:

- JSON-compatible dataset-level provenance fields when included by the mapping;
- textual history content where supported;
- support-local temporal/label content only within the accepted portable
  coordinate and label subsets.

Portable persistence is not required to preserve the full richness of native
or runtime provenance.

### 9.3 What does not belong in the portable contract by default

The following SHOULD generally remain outside default portable guarantees:

- arbitrary rich runtime provenance helpers;
- non-portable coordinate-label structures;
- project-wide recursive provenance semantics beyond dataset-level contracts;
- workflow-graph or reproducibility-state models not yet defined by RFC.

## 10. Part 9 — Explicit Non-Goals

This RFC does not define:

- a workflow reproducibility graph;
- a dedicated project provenance schema;
- a dedicated result provenance schema;
- parser or reader implementation details;
- operation-by-operation exact message wording for history;
- a redesign of current dataset field names.

## 11. Part 10 — Final Contract

Maintainers SHOULD apply the following contract:

1. Treat provenance and history as related but distinct concepts.
2. Treat `NDDataset` as the primary owner of dataset-level provenance and
   history.
3. Treat `author`, `origin`, `filename`, `created`, and `modified` as
   provenance fields.
4. Treat `history` as the explicit event-trail field.
5. Preserve provenance broadly across identity changes when scientific lineage
   remains traceable.
6. Extend history for ordinary single-source operations; synthesize or merge it
   for multi-source operations.
7. Distinguish temporal lineage from temporal support geometry; the same source
   may legitimately supply both.
8. Use reader normalization to place imported metadata into provenance,
   history, coordinates, labels, or `Meta` by meaning rather than by storage
   convenience.
9. Preserve richer provenance in native persistence than in portable
   persistence.

## 12. Promotion Candidates

### Future architecture notes

- A maintainer architecture note on provenance propagation across the major
  operation families once the current draft propagation work stabilizes.
- A note on temporal semantics connecting acquisition time, support time, and
  provenance time.

### Future RFCs

- `result-provenance-contract.md`
- `project-provenance-and-container-context.md`
- `portable-provenance-subset-contract.md`

### Historical-only material

- exact current per-operation history wording;
- implementation quirks of timestamp handling in individual readers;
- temporary inconsistencies preserved only for backward compatibility.
