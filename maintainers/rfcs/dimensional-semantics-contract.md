[Maintainer Docs](../README.md) · [RFC Index](INDEX.md)

# Dimensional Semantics Contract

## Status

Accepted maintainer RFC.

This document is conceptual and normative in intent.

It defines the maintainer-level contract for what a dimension is in
SpectroChemPy.

It does not define implementation details, migration sequencing, or API work.

The key words MUST, SHOULD, and MAY are to be interpreted as normative
requirements for maintainers and future contributors.

## 1. Purpose

This RFC answers one architectural question:

```text
What is a dimension in SpectroChemPy?
```

The goal is to make dimensional semantics explicit before any future work on:

- coordinate and `CoordSet` contracts;
- metadata taxonomy;
- portable persistence refinement;
- operation-specific geometry rules.

This contract is defined from the SpectroChemPy runtime object model first.

Its main architectural reference points are:

- SpectroChemPy's own scientific object model;
- NumPy for axis and shape intuition;
- Pint for unit-bearing support semantics;
- selected xarray concepts used for portable persistence.

It is not derived from CSDM, and no external interchange format is the primary
source of dimensional semantics for this RFC.

Inside scope:

- the conceptual role of dimensions;
- ownership and source of truth;
- the relationship between dimensions and coordinates;
- lifecycle semantics across common operation families;
- the minimum portable persistence contract for dimensions;
- comparison with adjacent ecosystems.

Outside scope:

- implementation changes;
- migration plans;
- xarray-style alignment;
- coordinate arithmetic redesign;
- metadata propagation rules;
- label semantics;
- `CoordSet` storage internals.

## 2. Current Model

SpectroChemPy currently uses a hybrid dimensional model.

In current behavior, a dimension is simultaneously:

- an ordered array axis;
- a structural key carried by the host array object;
- the anchor for attached support coordinates;
- the schema element exported into portable xarray / NetCDF mappings.

It is not currently:

- a full alignment key in the xarray sense;
- a complete scientific ontology;
- a coordinate value container;
- a general-purpose index object like a pandas index.

The current architecture therefore sits between two simpler models:

- narrower than xarray named-dimension alignment;
- richer than plain NumPy positional axes.

That hybrid model is already coherent enough to preserve, but it needs a more
explicit contract.

## 3. Architectural Problems

The main problem is not that dimensions are broken.

The main problem is that their role is only partly explicit, which creates
ambiguity in three areas.

### 3.1 Structural ambiguity

Dimensions are sometimes treated as plain axis positions and sometimes as named
structural identities.

This is manageable in practice, but it weakens reasoning about reshape,
concatenate, reduction, and portability.

### 3.2 Boundary ambiguity with coordinates

Dimensions anchor coordinates, but coordinates carry most axis meaning:

- values;
- units;
- titles;
- labels;
- same-dimension alternatives.

Without an explicit contract, dimensions and coordinates can look like two
competing definitions of the same thing.

### 3.3 Portability ambiguity

Portable persistence already relies on dimensions as stable schema anchors.

If their contract is not explicit, xarray/NetCDF mapping risks becoming an
accidental implementation convention rather than an intentional maintainer
boundary.

## 4. Proposed Dimensional Contract

### 4.1 Core definition

A dimension in SpectroChemPy MUST be understood as:

```text
an ordered structural axis identifier of a host scientific array object
that anchors coordinate attachment
and provides the primary portable schema key for that axis
```

This definition has five parts.

SpectroChemPy needs this concept because its runtime scientific objects must
simultaneously answer several structural questions:

- how many axes does this object have;
- which axis is which after reordering, reduction, or reshaping;
- where do support coordinates attach;
- how can one axis carry richer support information than plain positions;
- what minimum structural schema must survive portable persistence.

Dimensions exist to answer those questions without forcing coordinates to serve
as both structural identity and support description.

### Positional axis

A dimension corresponds to one axis position in the host array.

This is fundamental, but not sufficient by itself.

Plain position explains array layout, but not coordinate attachment or portable
schema identity.

### Structural key

A dimension is a named structural key.

This is fundamental.

The dimension key identifies the axis inside the host object and across
operations that preserve that axis.

### Semantic identifier

A dimension is a limited semantic identifier.

This is fundamental, but intentionally narrow.

It identifies which structural axis is being discussed, but it does not by
itself encode:

- coordinate values;
- physical meaning;
- alignment policy;
- category/index semantics.

### Coordinate anchor

A dimension is the anchor to which coordinates attach.

This is fundamental.

Coordinates do not float independently of dimensions in the `NDDataset` model.
They are owned relative to a dimension.

### Persistence schema element

A dimension is the primary portable schema element for an axis.

This is fundamental.

Portable persistence must preserve dimension count, names, and order even when
it cannot preserve all richer runtime coordinate relationships.

### 4.2 What is fundamental vs secondary

Fundamental:

- ordered host-axis identity;
- structural key semantics;
- coordinate anchoring;
- portable schema anchoring.

Secondary consequences:

- user-facing readability of names;
- domain meaning inferred from attached coordinates;
- operation-specific validation policies;
- display conventions.

The core contract is therefore:

```text
dimension = structural axis identity first
coordinate attachment anchor second
portable schema key third
```

All three are part of the same maintained concept.

## 5. Ownership and Source of Truth

### 5.1 Ownership split

Dimensional semantics are shared across several classes, but not equally.

### `NDArray`

`NDArray` owns the base ordered dimension tuple for the host object.

It therefore owns:

- dimension count;
- dimension names;
- dimension order.

### `NDDataset`

`NDDataset` is the main scientific object that gives dimensions their full
runtime meaning by combining:

- array shape and dims;
- `CoordSet` ownership;
- operation-level geometry behavior;
- persistence participation.

`NDDataset` is therefore the main practical carrier of dimensional semantics in
the scientific object model.

### `Coord`

`Coord` does not own dimensions.

It is attached to a dimension and describes support information for that
dimension.

Its values, units, labels, or title may explain the dimension, but they do not
create the dimension.

### `CoordSet`

`CoordSet` owns coordinate attachment and grouping semantics relative to
dimensions.

It owns:

- which coordinates belong to which dimension;
- which coordinate is default for a dimension;
- which same-dimension sibling coordinates exist;
- reference/grouping semantics among coordinates.

It does not own:

- dimension existence;
- dimension order;
- dimension identity independent of the host object.

### 5.2 Source of truth

The source of truth for dimension existence, names, and order MUST be the host
array object's `dims`.

For `NDDataset`, the source of truth is therefore:

```text
NDDataset.dims for axis identity
+
NDDataset.coordset for support attachment on those axes
```

No coordinate or `CoordSet` entry may create an independent dimension that is
not present in the owning dataset's `dims`.

## 6. Relationship with Coordinates

The core distinction is:

```text
dimension != coordinate
```

A dimension and a coordinate are related, but they are not interchangeable.

### 6.1 What a dimension provides

A dimension provides:

- structural axis identity;
- axis order within the host object;
- the attachment point for support information;
- the portable schema key for that axis.

A dimension does not provide:

- numerical support values;
- units of the support axis;
- coordinate labels;
- same-dimension alternative supports;
- coordinate-local metadata.

### 6.2 What a coordinate provides

A coordinate provides support description for one dimension.

That support may include:

- coordinate values;
- units;
- title;
- labels;
- coordinate-local metadata;
- display/linearization hints.

A coordinate does not define axis existence by itself.

This distinction matters because SpectroChemPy often needs more than one
support description on the same axis.

For one structural dimension, the runtime model may need to preserve:

- one default numeric support coordinate;
- one or more same-dimension alternative coordinates;
- labels or categorical annotations attached to that same axis;
- coordinate-local metadata relevant to interpretation or display.

If the dimension itself were treated as the coordinate, SpectroChemPy would
lose the clean separation between:

- axis identity;
- support values;
- alternative support descriptions.

### 6.3 What belongs to `CoordSet`

`CoordSet` provides the coordination layer between structural dimensions and
support objects.

It belongs to `CoordSet` to define:

- default coordinate selection per dimension;
- same-dimension sibling coordinates;
- reference semantics between coordinates;
- grouped lifecycle handling when dimensions are preserved, dropped, or
  reconstructed.

It does not belong to `CoordSet` to redefine the host dataset's dimension
order or create dimension identity independent of the host array.

### 6.4 Intentional project position

SpectroChemPy intentionally differs from coordinate-first models.

The maintained position is:

- dimensions define where an axis exists structurally;
- coordinates define how that axis is supported or interpreted;
- `CoordSet` defines how multiple support descriptions are organized around
  that structural axis.

This is why `CoordSet` exists.

SpectroChemPy does not only need "one axis -> one coordinate". It also needs a
runtime layer to represent:

- default coordinate selection;
- multiple same-dimension support descriptions;
- stable grouping around one structural axis;
- richer internal support semantics than most portable formats can preserve.

`CoordSet` is therefore not an accidental container. It is the layer that lets
SpectroChemPy keep dimensions structurally simple while allowing coordinate
semantics to remain scientifically expressive.

## 7. Lifecycle Semantics

This section defines how dimensional identity behaves across common operation
families.

### 7.1 Slicing

#### Slice-like selection

Slice-like selection that preserves an axis as an axis MUST preserve that
dimension's identity.

Examples:

- slice selection;
- list selection;
- boolean filtering along an axis;
- label-based subset selection that still returns that axis.

In these cases:

- the dimension survives;
- its order relative to other surviving dimensions survives;
- attached coordinates are sliced or filtered relative to that same dimension.

#### Scalarizing selection

Selection that consumes an axis into a scalar position MUST destroy that
dimension.

Examples:

- integer indexing;
- equivalent single-position extraction that removes the axis.

In these cases:

- the dimension no longer exists on the result;
- its coordinates are dropped or locally resolved as part of the selection
  result, not preserved as surviving axis structure.

### 7.2 Reduction

Reduction over a dimension destroys that dimension unless the operation
explicitly retains it.

Default reduction rule:

- reduced dimensions are removed;
- non-reduced dimensions preserve their identity.

If a future or current operation uses explicit keepdims-style semantics, that
operation MAY preserve the dimension key as a retained structural axis.

When retained, it remains the same dimension structurally, but with rebuilt
coordinate semantics as required by the operation.

### 7.3 Stacking

Stacking creates a new dimension.

The new stacking dimension MUST be treated as a newly created structural axis,
not as the preservation of a previously existing dimension.

Other input dimensions preserve their identity if they are carried through
unchanged.

### 7.4 Concatenation

Concatenation along an existing dimension preserves that dimension's identity.

This means:

- the concatenated axis remains the same named structural axis;
- its extent changes;
- its support coordinates may be locally rebuilt or concatenated;
- non-concatenated dimensions preserve their identity unchanged.

Concatenation does not by itself create a new dimension.

If a workflow needs a new axis, that is stacking, not concatenation.

### 7.5 Reshape

Generic reshape is a structural reinterpretation operation.

For dimensions affected by reshape, the default rule MUST be:

- old dimensional identity is destroyed;
- new dimensional identity is created according to the result schema.

Only dimensions explicitly carried through unchanged by the reshape contract
may preserve their identity.

This is an intentional divergence from a purely positional model.

Reshape may preserve data ordering while still destroying axis identity for the
affected axes.

### 7.6 Transpose

Transpose preserves dimensional identity and reorders dimensions.

It does not create or destroy dimensions.

It changes:

- axis order;
- the positional location of each dimension.

It does not change:

- which structural dimensions exist.

### 7.7 swapdims

`swapdims` preserves dimensional identity and reassigns positional order.

Like transpose, it does not create or destroy dimensions.

Its special meaning is explicit pairwise or named reordering rather than a
general permutation.

### 7.8 Summary table

| Operation | Preserve identity | Create new dimensions | Destroy dimensions |
|---|---|---|---|
| Slice / filter that keeps axis | Yes | No | No |
| Integer-like axis selection | No for selected axis | No | Yes for selected axis |
| Reduction | Yes for unreduced axes | No | Yes for reduced axes, unless explicitly retained |
| Stacking | Yes for carried-through axes | Yes | No |
| Concatenation | Yes | No | No |
| Reshape | Only for explicitly unaffected axes | Yes for affected result axes | Yes for affected source axes |
| Transpose | Yes | No | No |
| `swapdims` | Yes | No | No |

## 8. Portable Persistence

### 8.1 xarray mapping

In the xarray mapping, dimensions MUST survive as:

- explicit string dimension names;
- exact dimension order on the primary variable;
- the schema anchors for default and auxiliary coordinate attachment.

Portable persistence MUST preserve:

- dimension count;
- dimension names;
- dimension order;
- the distinction between a dimension and its attached coordinate values.

Portable persistence MUST NOT require:

- preservation of full runtime `CoordSet` topology;
- preservation of internal reference-sharing identity;
- promotion of dimensions into xarray-style alignment keys beyond the current
  mapping contract.

### 8.2 NetCDF mapping

In NetCDF, dimensions MUST remain the stable structural schema elements that
support:

- the primary variable shape;
- default coordinate variables;
- auxiliary same-dimension coordinate variables when preserved.

Backend constraints MUST NOT redefine the conceptual role of dimensions.

NetCDF is a persistence backend for an already-defined dimensional contract,
not the source of that contract.

### 8.3 Future CSDM interoperability

Future CSDM interoperability SHOULD preserve the same minimum dimensional
semantics:

- explicit axis count;
- axis order where required by the carrier;
- stable attachment between structural axes and support descriptions.

This requirement is intentionally narrow.

CSDM is an optional interoperability target, not the architectural reference
model for SpectroChemPy dimensions.

The maintained requirement is:

```text
portable formats should preserve
the SpectroChemPy dimensional contract
as far as their schemas allow
```

Interoperability should adapt to the SpectroChemPy contract, not define it.

## 9. Comparison with Adjacent Models

This section is comparative only.

The dimensional contract above is defined from SpectroChemPy runtime semantics,
not derived from any external library or exchange format.

### 9.1 NumPy

NumPy dimensions are positional axes only.

SpectroChemPy intentionally differs by giving dimensions:

- explicit names;
- structural identity;
- coordinate anchoring;
- portable schema meaning.

SpectroChemPy should remain NumPy-like for shape reasoning where practical, but
it is intentionally richer than pure positional-array semantics.

### 9.2 xarray

xarray dimensions are named structural axes and participate in alignment.

SpectroChemPy intentionally adopts only part of that model.

Shared:

- named dimensions;
- coordinate attachment;
- portable schema use.

Not adopted:

- automatic alignment;
- index-join semantics;
- dimension names as general compatibility keys across operations.

This is intentional.

SpectroChemPy is a spectroscopy-first scientific object model, not a full
labeled-array alignment system.

### 9.3 SpectroChemPy runtime emphasis

The most important comparison point is SpectroChemPy's own runtime model.

SpectroChemPy needs dimensions because it must keep three layers distinct:

- structural axis identity;
- support coordinates;
- grouped same-dimension coordinate semantics.

This is the project-specific rationale that neither NumPy nor xarray fully
captures.

NumPy is too thin because positional axes alone do not explain coordinate
attachment, default-coordinate choice, or multiple coordinates on one axis.

xarray is too strong as a reference model because its dimension semantics are
more tightly coupled to alignment and indexing behavior than SpectroChemPy
currently wants.

The maintained SpectroChemPy position is therefore:

- dimensions stay structurally authoritative;
- coordinates stay scientifically descriptive;
- `CoordSet` stays responsible for organizing multiple coordinate views of one
  structural axis.

### 9.4 pandas indexes

pandas indexes are central axis identity and alignment objects.

SpectroChemPy intentionally differs because:

- dimensions are not index objects;
- coordinates may carry some index-like meaning;
- alignment semantics are not defined through dimension names alone.

The nearest pandas analogue for many SpectroChemPy coordinates is therefore an
index-like support layer, not the dimension itself.

### 9.5 CSDM dimensions and interoperability

CSDM is useful as an interoperability comparison point, but it is secondary in
this RFC.

CSDM may provide a portable scientific dimension descriptor model for exchange
workflows. SpectroChemPy does not need to adopt that model as its own runtime
architecture.

SpectroChemPy intentionally keeps its own split:

- the dimension is the structural axis identity;
- the coordinate carries most axis-support meaning;
- `CoordSet` can represent richer same-dimension support relationships than a
  thin structural dimension alone.

The practical consequence is:

- CSDM may be a useful adapter target;
- CSDM does not define SpectroChemPy dimensional semantics;
- any CSDM mapping should preserve the SpectroChemPy contract as far as the
  exchange schema allows, while reporting loss or narrowing explicitly when
  needed.

## 10. Explicit Non-Goals

This RFC does not:

- introduce full alignment semantics;
- define coordinate arithmetic compatibility rules;
- define metadata propagation rules;
- define label semantics;
- define `CoordSet` storage internals;
- define migration or implementation sequencing;
- require any immediate public API redesign.

It also does not require dimensions to become a separate runtime class.

## 11. Open Questions

The following questions remain intentionally open.

- Should keepdims-style operations preserve the original dimension key in all
  cases, or only in selected reduction families?
- Should some reshape operations preserve identity for singleton insertion or
  removal, or should generic reshape remain identity-breaking for affected
  axes?
- Should future arithmetic or validation policies make stronger use of
  dimension identity without moving to full xarray-style alignment?
- How much same-dimension coordinate richness should portable formats be
  expected to preserve?
- Should some domain-specific dimensions acquire stronger conventional meaning
  without becoming global alignment keys?

## 12. Candidate Follow-Up RFCs

- `coordinate-and-coordset-semantics.md`
  Clarify support ownership, default coordinates, sibling coordinates, and
  reference semantics around dimensions.
- `metadata-taxonomy-contract.md`
  Clarify which metadata categories are structural, scientific, provenance, or
  presentation-related now that dimensional semantics are explicit.
- `label-semantics.md`
  Clarify which labels are coordinate-attached annotations, categorical
  identifiers, or display conveniences.
- `reader-metadata-normalization-contract.md`
  Clarify how imported external metadata should map onto dimensions,
  coordinates, and general metadata fields.

## 13. Promotion Candidates

### Suitable for architecture notes

Implemented as:

- [`maintainers/architecture/metadata-and-support-model.md`](../architecture/metadata-and-support-model.md)
  documents the dimensional semantics adopted from this RFC family.

The RFC itself remains the primary reference for dimensional semantic
contracts.

### Suitable for Coordinate & CoordSet RFC work

- the explicit split between structural axis identity, coordinate support
  values, and `CoordSet` grouping semantics;
- lifecycle rules for default and same-dimension sibling coordinates when a
  dimension is preserved, destroyed, or created;
- ownership boundaries between `NDDataset.dims` and `NDDataset.coordset`.

### Suitable for Metadata Taxonomy RFC work

- classification of dimensions as structural metadata rather than scientific
  identity metadata;
- the boundary between dimension names, coordinate titles, and coordinate
  labels;
- the persistence distinction between structural schema elements and richer
  coordinate-local metadata.
