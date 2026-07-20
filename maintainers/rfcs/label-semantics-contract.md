[Maintainer Docs](../README.md) · [RFC Index](INDEX.md)

# Label Semantics Contract

## Status

Accepted maintainer RFC.

This document is conceptual and normative in intent.

It defines the maintainer-level contract for labels in SpectroChemPy:

- what a label is;
- what a label is not;
- who owns labels;
- how labels should behave conceptually;
- what subset of label semantics belongs in portable persistence.

It builds on:

- [`dimensional-semantics-contract.md`](dimensional-semantics-contract.md)
- [`coordinate-and-coordset-semantics.md`](coordinate-and-coordset-semantics.md)
- [`metadata-taxonomy-contract.md`](metadata-taxonomy-contract.md)

It does not reopen those decisions.

It does not define implementation details, migration sequencing, or API work.

The key words MUST, SHOULD, and MAY are to be interpreted as normative
requirements for maintainers and future contributors.

## 1. Purpose

This RFC answers five linked questions:

```text
What is a label in SpectroChemPy?
What is a label not?
Who owns labels?
How should labels behave?
What should be portable?
```

Recent architecture work already clarified:

- dimensions are structural axis identifiers;
- coordinates are support objects attached to dimensions;
- `CoordSet` owns support organization relative to dimensions;
- metadata has an explicit maintainer taxonomy.

Labels remain the main unresolved concept at the boundary between support,
scientific identification, annotations, and display.

This RFC therefore defines labels as their own semantic surface rather than
trying to collapse them into dimensions, coordinates, or generic metadata.

Inside scope:

- current label behavior across core object types;
- conceptual roles served by labels;
- ownership boundaries;
- the relationship between labels, metadata, and coordinates;
- label propagation philosophy;
- runtime versus native versus portable persistence boundaries;
- explicit non-goals and follow-up RFC candidates.

Outside scope:

- implementation changes;
- migration plans;
- dimensional semantics;
- coordinate semantics;
- full metadata redesign;
- arithmetic redesign;
- result-object algorithm behavior.

## 2. Part 1 — Current Behavior

This section describes current behavior, not intended behavior.

### 2.1 Labels on `NDArray`

`NDArray` currently defines a public `labels` surface.

Observed current behavior includes:

- `labels` is a base-array trait/property, not a `Coord`-only feature;
- labels are accepted only for 1D arrays in the normal direct-data model;
- multi-row labels are supported through an extra axis in the label array;
- label-only objects are possible in some cases, with labels acting as the
  visible payload when numeric data are absent;
- label-aware sorting and selection paths exist in the base array stack.

This means labels currently live lower in the class hierarchy than their most
natural scientific role would suggest.

### 2.2 Labels on `Coord`

`Coord` currently carries labels directly and is the main runtime home of
scientifically meaningful labels.

Observed uses include:

- coordinate labels accompanying numeric coordinate values;
- label-only coordinates;
- multiple label rows attached to one coordinate;
- datetime-like labels from readers;
- sample names, filenames, acquisition identifiers, channel names, species
  names, and target names;
- coordinate display and repr output that shows labels prominently.

`Coord` labels are sliced with the coordinate and are used by plotting and
selection paths.

### 2.3 Labels on `NDDataset`

`NDDataset` explicitly disables direct dataset labels with
`_labels_allowed = False`.

Observed consequence:

- dataset-level user-facing labels usually live in the coordinates owned by the
  dataset rather than on the dataset payload itself.

This means current runtime behavior already distinguishes:

- labels attached to support/axis semantics;
- the dataset signal itself.

### 2.4 Labels on `CoordSet`

`CoordSet` does not own a separate label storage model.

Its current role is organizational:

- it exposes labels through child coordinates;
- it preserves same-dimension grouping and default-coordinate context around
  those labels;
- lifecycle operations such as slicing, concatenation, stacking, and
  interpolation often preserve or rebuild labels through coordinate handling.

### 2.5 Labels in persistence

Current persistence behavior is mixed.

Observed current behavior:

- trusted native persistence preserves labels;
- JSON/native utilities already special-case label serialization;
- portable xarray/NetCDF persistence now preserves a limited string-label
  subset for coordinates;
- richer or non-exportable label cases remain partial, deferred, or
  warning-driven rather than fully portable.

This means labels are not runtime-only, but their portable contract is narrower
than their runtime richness.

### 2.6 Labels in display

Labels currently play a strong display role.

Observed uses include:

- coordinate repr output;
- plot tick labels;
- point annotations;
- synthetic or convenience labels generated for visualization workflows.

Display use is real, but it does not exhaust label semantics.

### 2.7 Labels in analysis workflows

Labels also participate in scientific and analysis workflows.

Observed uses include:

- target labels in PLS;
- sample identifiers or acquisition context imported by readers;
- dataset-name-derived stack labels;
- labels used to interpret results or organize observations.

This confirms that labels are not merely cosmetic.

## 3. Part 2 — Label Roles

Labels currently serve several different roles.

The project should make those roles explicit rather than pretending that one
label interpretation covers every case.

### 3.1 Labels as annotations

Labels MAY serve as annotations attached to support points.

Examples:

- point annotations;
- peak annotations;
- acquisition notes;
- per-point remarks.

This role is support-local and descriptive rather than structural.

### 3.2 Labels as identifiers

Labels MAY serve as human-meaningful identifiers.

Examples:

- sample identifiers;
- acquisition identifiers;
- experiment identifiers;
- component identifiers;
- target names.

This role is scientifically meaningful and often worth preserving.

### 3.3 Labels as categories

Labels MAY serve as categorical descriptors.

Examples:

- class names;
- groups;
- assignments;
- targets;
- condition tags.

This role can influence analysis interpretation without becoming a structural
dimension definition.

### 3.4 Labels as display aids

Labels MAY serve as display aids.

Examples:

- tick labels;
- human-readable display substitutions;
- synthetic plotting labels.

This role is valid, but it is not the whole contract.

### 3.5 Labels as coordinates

It is tempting to treat labels as a special coordinate type.

That is only partly true.

Labels may be carried by coordinates, and label-only coordinates exist in
current behavior. However, the distinction:

```text
numeric coordinate
vs
label coordinate
```

is more misleading than helpful if it is treated as a fundamental architecture
split.

The maintained position of this RFC is:

- labels MAY be one payload carried by a coordinate;
- labels MUST NOT redefine what a dimension is;
- labels MUST NOT automatically become coordinate values in the structural
  sense;
- labels SHOULD be treated as support-local descriptive or identifying content,
  not as the primary definition of support geometry.

## 4. Part 3 — Ownership

This section answers:

```text
Who owns labels?
```

### 4.1 `NDArray`

`NDArray` currently owns the lowest-level public label storage surface.

That is a current implementation reality, not the best conceptual source of
truth for scientific label semantics.

### 4.2 `Coord`

`Coord` SHOULD be treated as the primary semantic owner of labels.

This is the most stable interpretation of current scientific behavior because
labels are usually attached to one support axis and travel with that support
object.

Coordinate-local label ownership includes:

- support labels;
- sample or acquisition identifiers tied to one axis;
- categorical point descriptors tied to one axis;
- label-only support descriptions.

### 4.3 `CoordSet`

`CoordSet` does not own label values directly, but it owns label organization
relative to the dataset geometry.

Its responsibilities include:

- which coordinate on a dimension is the default label-bearing support;
- how same-dimension sibling coordinates coexist;
- how labels participate in lifecycle operations through coordinate grouping.

### 4.4 `NDDataset`

`NDDataset` SHOULD be treated as the owner of label context, not the direct
owner of most label values.

In practice:

- the dataset owns the dimensions and the `CoordSet`;
- coordinates attached to that `CoordSet` own the concrete labels.

Dataset-level conceptual labels therefore usually mean:

```text
labels carried by the dataset's support model
```

rather than:

```text
labels owned by the signal array itself
```

### 4.5 Ownership summary

The maintained ownership model is:

```text
NDArray
    current low-level label storage surface

Coord
    primary semantic owner of concrete labels

CoordSet
    owner of label organization relative to dimensions

NDDataset
    owner of the overall labelled scientific object through its CoordSet
```

## 5. Part 4 — Relationship to Metadata

Using the Metadata Taxonomy Contract, labels SHOULD be treated as a distinct
semantic surface that interacts with multiple metadata categories.

### 5.1 Are labels metadata?

Labels are partly metadata, but not reducible to generic metadata.

They can express:

- scientific identity;
- presentation hints;
- support-local annotation;
- categorical descriptors.

But they are also attached to coordinate support and often participate in
selection, plotting, and scientific interpretation.

### 5.2 Maintained position

For maintainer reasoning:

- labels MUST NOT be collapsed into generic `Meta`;
- labels MUST NOT be treated as pure structural metadata;
- labels MUST NOT be treated as pure presentation metadata;
- labels SHOULD be treated as a dedicated semantic surface at the boundary
  between support-local metadata, scientific identification, and display.

### 5.3 Metadata categories that interact with labels

The strongest interactions are with:

- scientific identity metadata;
- presentation metadata;
- extension/private metadata when labels originate from reader-specific import
  payloads before normalization.

Labels may also carry provenance-adjacent content such as filenames or
acquisition timestamps, but that does not make the label system itself the
primary provenance contract.

## 6. Part 5 — Relationship to Coordinates

Using the Coordinate Contract:

```text
coordinate
vs
label
```

must remain explicit.

### 6.1 A coordinate can exist without labels

Yes.

Numeric coordinates without labels are normal and fundamental.

### 6.2 Labels can exist without numeric coordinate values

Yes, in current behavior.

Label-only coordinates already exist and are meaningful for some workflows.

### 6.3 Labels do not define support structure

Labels SHOULD NOT be the primary definition of support structure.

Support structure belongs to:

- the dimension as structural axis identity;
- the coordinate as the support object attached to that dimension;
- `CoordSet` as the organizer of support objects.

### 6.4 Labels do not own dimensional semantics

Labels MUST NOT participate as the primary source of dimensional identity.

They may describe, annotate, or identify support positions on a dimension, but
they do not define the dimension itself.

### 6.5 Maintained boundary

The maintained conceptual boundary is:

```text
dimension
    structural axis identity

coordinate
    support object attached to that axis

label
    support-local identifying / annotating / categorical / display content
    that may be carried by the coordinate
```

This boundary allows multiple coordinates to exist on one dimension without
requiring every coordinate payload to be a label carrier.

## 7. Part 6 — Propagation Philosophy

This section defines conceptual guidance, not algorithms.

### 7.1 Slicing

Labels SHOULD generally be `preserve` when slicing preserves the underlying
support points represented by the owning coordinate.

### 7.2 Reduction

Labels on a reduced-away dimension SHOULD generally be `drop`.

Labels on non-reduced dimensions SHOULD generally be `preserve`.

### 7.3 Concatenation

Labels along the concatenated support axis SHOULD generally be `merge` when the
operation is structurally a join of support points.

Labels on untouched axes SHOULD generally be `preserve`.

### 7.4 Stacking

Stacking MAY `recompute` or `synthesize` labels for the new stack dimension
when that dimension is newly created from source-object identity.

Existing axis labels on preserved dimensions SHOULD generally be `preserve`.

### 7.5 Interpolation

Interpolation SHOULD distinguish two cases:

- exact support carry-over: labels MAY be `preserve`;
- newly created support points: labels SHOULD generally be `drop` unless a
  clearly valid recomputation rule exists.

In general, interpolation is not a safe default case for arbitrary label
preservation.

### 7.6 Reshape

Reshape that changes the support interpretation of a labelled axis SHOULD
generally `drop` labels unless there is an explicit support-preserving mapping.

Purely representational reshapes that preserve one labelled axis identically
MAY `preserve` that axis labels.

### 7.7 Transpose

Transpose SHOULD generally `preserve` labels because it reorders axes without
changing the support content attached to each preserved dimension.

### 7.8 Summary rule

The guiding principle is:

```text
Preserve labels when support identity is preserved.
Drop or recompute labels when support identity is destroyed or newly created.
Merge labels when support points are structurally combined.
```

## 8. Part 7 — Portable Persistence

Labels participate differently across three layers:

```text
Runtime richness
Native persistence
Portable persistence
```

### 8.1 Runtime richness

Runtime labels MAY be rich, heterogeneous, multi-row, object-typed, and tied
to specific workflows.

Runtime richness is broader than any portable contract.

### 8.2 Native persistence

Trusted native persistence SHOULD preserve labels as part of the project's
runtime scientific model.

That path exists to preserve SpectroChemPy-native richness, not to force a
reduction to an interchange subset.

### 8.3 Portable persistence

Portable persistence MUST preserve only the label subset that can be expressed
without redefining the portable carrier model.

The current portable-label RFC already defines that narrower subset.

Therefore:

- portable persistence SHOULD preserve labels when they fit the accepted
  portable coordinate-label subset;
- portable persistence MUST NOT be required to preserve every runtime label
  form;
- non-portable labels SHOULD trigger explicit user-visible handling rather than
  silent conceptual disappearance.

### 8.4 What should be portable

The highest-priority portable label content is:

- 1D support-aligned string labels;
- scientifically meaningful identifiers such as sample names, target names, or
  acquisition labels when they fit the accepted portable subset.

Lower-priority or non-portable content includes:

- mixed-type label arrays;
- semantically heterogeneous multi-row annotations without a stable portable
  interpretation;
- synthetic display-only labels;
- label usages that are really misplaced coordinate values.

## 9. Part 8 — Comparison

### 9.1 NumPy

NumPy has positional axes and array values but no first-class label semantics
for support objects.

SpectroChemPy intentionally differs by allowing support-local labels to travel
with coordinates.

### 9.2 xarray

xarray emphasizes named dimensions and coordinate variables.

SpectroChemPy intentionally remains runtime-first:

- dimensions stay structural;
- coordinates remain support objects;
- labels are not automatically promoted to full coordinate-schema identity.

Portable xarray mapping adapts to this model rather than defining it.

### 9.3 pandas-style indexing

pandas often treats labels as primary index semantics.

SpectroChemPy intentionally differs:

- labels are not the primary structural key;
- labels do not define dimension identity;
- labels usually annotate support rather than replace it.

### 9.4 Scientific exchange formats

Scientific exchange formats often preserve only a narrower subset of runtime
annotation and support semantics.

SpectroChemPy therefore distinguishes:

- rich runtime label semantics;
- trusted native persistence of that richness;
- narrower portable preservation when schemas allow it.

## 10. Part 9 — Explicit Non-Goals

This RFC does not define:

- dimensional semantics;
- coordinate semantics in full;
- arithmetic semantics;
- result-object algorithms;
- a full display policy;
- a full provenance graph;
- a complete normalization policy for reader-imported labels;
- implementation details for every portable label encoding.

## 11. Part 10 — Candidate Follow-Up RFCs

The next likely follow-up RFCs are:

- **Reader Metadata Normalization Contract**
  to clarify when imported reader payloads should become typed provenance
  fields, labels, or generic metadata.
- **Provenance and History Contract**
  to clarify when filenames, acquisition times, and transformation history
  should live as labels versus dedicated provenance metadata.
- **Label Selection and Indexing Contract**
  to clarify user-facing selection semantics, especially where labels influence
  indexing or analysis views.

These follow-up topics depend on the present RFC because they require a stable
answer to:

```text
what labels are
and what they are not
```

## 12. Promotion Candidates

### Future architecture notes

- A maintainer architecture note on support semantics across dimensions,
  coordinates, labels, and `CoordSet` once the dimensional, coordinate, and
  label RFCs stabilize together.
- A focused architecture note on label propagation and analysis/display
  implications once operation-level policy is better characterized.

### Future RFCs

- `reader-metadata-normalization-contract.md`
- `provenance-and-history-contract.md`
- `label-selection-and-indexing-contract.md`

### Historical-only material

- local characterization detail about individual reader label payloads;
- implementation-specific storage quirks of the current `NDArray.labels`
  machinery;
- temporary audit comparisons that are useful for migration history but do not
  belong in the long-term contract.
