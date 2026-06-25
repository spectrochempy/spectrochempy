# Coordinate and CoordSet Semantics Contract

## Status

Accepted Maintainer RFC.

This document is conceptual and normative in intent.

It defines the maintainer-level contract for:

- `Coord`;
- `CoordSet`;
- default coordinates;
- same-dimension coordinates;
- coordinate ownership;
- coordinate lifecycle.

It follows the dimensional contract defined in
[`dimensional-semantics-contract.md`](dimensional-semantics-contract.md).

It does not define implementation details, migration sequencing, or API work.

The key words MUST, SHOULD, and MAY are to be interpreted as normative
requirements for maintainers and future contributors.

## 1. Purpose

This RFC answers three linked questions:

```text
What is a coordinate in SpectroChemPy?
What is CoordSet responsible for?
Why does CoordSet exist?
```

This contract is defined from the SpectroChemPy runtime object model first.

Its main architectural reference points are:

- SpectroChemPy's own scientific object model;
- the dimensional contract already established for structural axes;
- NumPy for array intuition;
- selected xarray concepts used for portable persistence.

It is not derived from CSDM, and it does not treat external coordinate models
as the primary source of SpectroChemPy coordinate semantics.

Inside scope:

- current runtime coordinate behavior;
- conceptual responsibilities of `Coord` and `CoordSet`;
- ownership boundaries between `NDDataset`, dimensions, `Coord`, and
  `CoordSet`;
- coordinate lifecycle rules across major operation families;
- runtime versus native versus portable persistence boundaries.

Outside scope:

- implementation changes;
- migration plans;
- xarray-style alignment;
- arithmetic redesign beyond existing accepted coordinate-arithmetic policy;
- metadata propagation rules outside coordinate-local concerns;
- result-object semantics.

## 2. Part 1 — Current Model

This section describes current behavior, not intended behavior.

### 2.1 Coord

`Coord` is currently a 1D support object implemented as:

```text
Coord -> NDMath -> NDArray
```

This means its present behavior is a mixture of:

- explicit support-object responsibilities;
- inherited array responsibilities from `NDArray`;
- inherited mathematical behavior from `NDMath`;
- coordinate-specific display and normalization behavior.

### Current responsibilities

Observed current responsibilities include:

- 1D support values when numeric data are present;
- coordinate labels, including label-only coordinate cases;
- axis units;
- axis title and name;
- coordinate-local `Meta`;
- coordinate-local display behavior;
- coordinate rounding and linearization;
- coordinate reversal conventions;
- copying, equality, hashing, and serialization participation inherited through
  the broader array stack;
- some arithmetic-like behavior inherited from `NDMath`.

### Fundamental current responsibilities

The responsibilities most strongly supported by the object model are:

- describing where data lives along one structural dimension;
- carrying support values and/or labels for that dimension;
- carrying axis-specific units;
- carrying coordinate-local identity and descriptive metadata;
- acting as the runtime support object attached to a dimension.

### Historical or inheritance-driven responsibilities

Some current `Coord` behavior appears to be more historical or inheritance-led
than conceptually central:

- broad shared math surface through `NDMath`;
- many generic `NDArray` conveniences that exist because `Coord` inherits the
  general array base;
- some array-like operations that are meaningful only in selected coordinate
  cases;
- persistence and attribute-list behavior shaped by shared array
  infrastructure rather than by a narrow coordinate-only contract.

### Current semantic posture

The current runtime model treats `Coord` primarily as an axis/support object,
not as a signal-bearing scientific dataset.

That distinction is reinforced by the separate dataset-vs-coordinate
arithmetic audit and by the accepted coordinate-arithmetic RFC.

### 2.2 CoordSet

`CoordSet` is currently the owned coordinate container of `NDDataset`.

It is not part of the array inheritance chain.

It is a separate coordination layer that stores one or more `Coord` objects
relative to dimensions.

### Current ownership model

Current source-of-truth relationships are:

```text
NDDataset.dims
    structural axis identity

NDDataset.coordset
    support ownership and grouping relative to those axes

Coord
    one support description attached to one dimension

CoordSet
    collection and coordination of those support descriptions
```

`CoordSet` does not create dimensions independently. It organizes coordinate
objects relative to dimensions already owned by the host dataset.

### Current grouping model

Current runtime behavior supports two main grouping shapes:

- top-level per-dimension coordinate ownership;
- nested same-dimension coordinate groups.

The internal storage redesign now treats group projection as the semantic
reference model and `_storage` as the runtime container.

### Current default-coordinate model

`CoordSet` currently exposes:

- `default`;
- `default_index`.

For a same-dimension group, one coordinate is selected as the default active
support coordinate.

For an empty `CoordSet`, `default` and `default_index` are `None`.

### Current sibling-coordinate model

`CoordSet` currently supports multiple coordinates attached to the same
dimension.

This is represented through same-dimension grouping and explicit
`is_same_dim` metadata.

The model allows:

- one default coordinate;
- one or more sibling coordinates on the same structural dimension;
- compatibility aliases such as `_1`, `_2`, `_3`;
- lookup and lifecycle behavior relative to those grouped coordinates.

### Current references

`CoordSet` currently has explicit `references` metadata.

This allows some dimensions to refer to another coordinate relationship
without requiring that portable or runtime identity be reduced to one simple
"one axis, one array" model.

### Current lifecycle role

Current architecture notes and runtime behavior show `CoordSet` acting as the
main lifecycle manager for coordinates during:

- slicing;
- reduction;
- stacking;
- concatenation;
- interpolation;
- reshape;
- transpose / swap;
- mutation, assignment, deletion, and same-dimension updates.

This lifecycle role is a major reason `CoordSet` exists as its own object.

## 3. Part 2 — Architectural Tensions

### 3.1 Coord as support object vs array object

The clearest tension is that `Coord` is conceptually a support object but
inherits a broad array and math surface.

Some array behavior is clearly essential:

- 1D storage;
- slicing;
- units;
- labels;
- copy behavior;
- display;
- serialization participation.

Some broader math behavior is much less clearly essential:

- generic arithmetic inherited from `NDMath`;
- operator installation that exists because of shared hierarchy;
- array-like affordances that are convenient but not central to support
  semantics.

This creates a recurring ambiguity:

```text
How much of Coord's current array behavior is a true coordinate contract,
and how much is legacy inheritance?
```

### 3.2 Coord vs NDDataset

The current boundary is conceptually meaningful but still partly implicit.

`Coord` belongs to:

- support location;
- axis interpretation;
- axis-local units;
- axis-local labels and metadata.

`NDDataset` belongs to:

- scientific signal values;
- dataset-level identity and provenance;
- the owned structural dimensions;
- dataset-level lifecycle and persistence semantics.

What should never belong to `Coord` as a primary role:

- signal-bearing dataset semantics;
- broad multi-dimensional scientific result identity;
- independent dimension creation;
- becoming the generic home for any 1D numeric vector aligned to an axis.

The dataset-vs-coordinate arithmetic audit makes this boundary explicit:

```text
Coord answers: where is the data and how should the axis be interpreted?
NDDataset answers: what are the measured or derived values?
```

### 3.3 CoordSet vs xarray coordinates

`CoordSet` exceeds the default xarray coordinate model in several ways.

Runtime capabilities that are richer than xarray's primary model include:

- explicit default-coordinate selection;
- multiple same-dimension sibling coordinates;
- compatibility aliases;
- explicit reference metadata;
- richer grouped lifecycle handling.

Some of these capabilities are intentionally runtime-heavy:

- alias stability;
- internal grouping identity;
- reference-sharing topology;
- some nested same-dimension semantics.

Portable persistence can preserve part of this richness, but not all of it.

### 3.4 Same-dimension coordinates

Same-dimension coordinates solve a real problem:

- one structural axis may need more than one support description;
- one description may be the primary scientific support;
- another may be an alternative numeric support;
- another may be a label-bearing or annotation-oriented support view.

The current guarantees are weaker than a full formal contract, but current
behavior does provide:

- explicit same-dimension grouping;
- one selected default coordinate;
- sibling preservation through many lifecycle paths;
- explicit `is_same_dim` intent that must not be inferred only from structure.

## 4. Part 3 — Proposed Contract

### 4.1 What a Coord is

A `Coord` in SpectroChemPy MUST be understood as:

```text
a one-dimensional support object
attached to one structural dimension
that describes how data on that dimension is located, interpreted,
or annotated
```

Core responsibilities of a `Coord` are:

- carrying support values and/or labels for one dimension;
- carrying axis units where applicable;
- carrying coordinate-local title, name, and metadata;
- participating in the interpretation of one structural dimension;
- remaining subordinate to the dimension it supports.

### 4.2 What a Coord is not

A `Coord` is not:

- an independently dimension-owning object;
- a signal-bearing `NDDataset` substitute;
- a general-purpose alignment key;
- the primary source of structural axis identity;
- the generic home for arbitrary 1D scientific values merely because they
  align with an axis.

It MAY be array-like for practical reasons.

That does not make array-likeness its defining semantic role.

### 4.3 What CoordSet is

A `CoordSet` in SpectroChemPy MUST be understood as:

```text
the coordinate-ownership and coordination layer
that organizes one or more Coord objects
relative to the structural dimensions of a host scientific object
```

`CoordSet` exists because SpectroChemPy needs more than "dimension name ->
single coordinate array".

It is the object that lets the runtime model distinguish between:

- structural dimension identity;
- active default support;
- alternative same-dimension support descriptions;
- runtime grouping and reference semantics.

### 4.4 What CoordSet owns

`CoordSet` owns:

- which coordinates belong to which dimensions;
- default-coordinate selection within a same-dimension group;
- sibling relationships among same-dimension coordinates;
- coordinate grouping state;
- explicit reference metadata between coordinate relationships;
- lifecycle handling for coordinate survival, rebuild, and drop decisions.

`CoordSet` does not own:

- dimension existence;
- dimension order;
- dimension identity independent of the host dataset;
- dataset-level signal values;
- dataset-level provenance and scientific identity.

### 4.5 Default coordinates

A default coordinate is the currently selected primary support description for
one structural dimension.

Its meaning is:

- this is the coordinate the runtime model treats as the active support axis
  for ordinary interpretation;
- this is the coordinate that portable persistence SHOULD preserve as the
  minimum per-dimension coordinate layer;
- this is not proof that no sibling coordinates exist.

Default selection is therefore a semantic choice inside `CoordSet`, not a
property of the dimension alone.

### 4.6 Same-dimension coordinates

Same-dimension coordinates are multiple coordinate descriptions attached to the
same structural dimension.

Their role is to allow SpectroChemPy to represent:

- alternative numeric supports;
- alternative scientific descriptions of the same axis;
- label-oriented views of the same axis;
- runtime richness beyond one-axis-one-coordinate models.

Same-dimension coordinates do not create new dimensions.

They are multiple support descriptions for one already-existing dimension.

### 4.7 Coordinate identity

Coordinate identity in SpectroChemPy is intentionally layered.

It includes:

- dimension attachment;
- coordinate name;
- coordinate title;
- sibling relationships inside a same-dimension group;
- default-versus-sibling role.

The owning dimension is the structural anchor.

Coordinate names and titles help distinguish support descriptions, but they are
not a replacement for dimension attachment.

Sibling relationships are part of coordinate meaning because two coordinates
with similar values but different group roles are not necessarily equivalent.

## 5. Part 4 — Lifecycle Semantics

This section follows
[`dimensional-semantics-contract.md`](dimensional-semantics-contract.md).

The dimension contract defines whether a structural axis survives, is created,
or is destroyed.

The coordinate contract defines what then happens to the support descriptions
attached to those axes.

### 5.1 Slicing

When slicing preserves a dimension as a dimension:

- attached coordinates SHOULD survive;
- coordinate values or labels SHOULD be sliced consistently with that axis;
- coordinate identity SHOULD be preserved as the same support description for
  the same surviving dimension.

When selection destroys a dimension:

- its coordinates SHOULD be dropped as surviving axis objects;
- any local scalar resolution is part of value extraction, not of surviving
  coordinate identity.

### 5.2 Reduction

When reduction destroys a dimension:

- coordinates attached to the reduced dimension SHOULD be dropped;
- coordinates on unreduced dimensions SHOULD survive with preserved identity.

When a dimension is explicitly retained by keepdims-like semantics:

- the retained dimension remains structurally the same dimension;
- its coordinates MAY need local rebuild or normalization depending on the
  operation family;
- unreduced coordinates SHOULD preserve identity.

### 5.3 Stacking

Stacking creates a new structural dimension.

For that new dimension:

- a new coordinate SHOULD be created or synthesized when the operation defines
  one;
- that coordinate is a newly created support description, not a preserved
  identity from an existing dimension.

Coordinates on carried-through dimensions SHOULD preserve identity unless the
stacking operation explicitly rebuilds them.

### 5.4 Concatenation

Concatenation preserves the concatenated dimension structurally.

For that dimension:

- the default coordinate MAY need local rebuild or concatenation;
- sibling coordinates MAY also need coordinated rebuild or concatenation;
- coordinate identity is preserved at the dimension level, but individual
  support descriptions may be reconstructed to represent the combined axis.

Non-concatenated dimensions and their coordinates SHOULD preserve identity.

### 5.5 Interpolation

Interpolation preserves the scientific object family while rebuilding support
locally on the interpolated axis.

Expected coordinate behavior:

- non-interpolated dimensions keep their coordinate identity;
- the interpolated dimension keeps structural identity;
- the active support coordinate on that dimension is typically rebuilt;
- label-bearing support on the interpolated axis may only survive where the
  result remains semantically exact.

Interpolation is therefore a central example of:

```text
same dimension
new local support realization
```

### 5.6 Reshape

When reshape destroys and recreates affected dimensions structurally:

- coordinates attached to destroyed source dimensions SHOULD not be treated as
  preserved identities;
- new result dimensions SHOULD receive new or rebuilt coordinate descriptions
  only where the reshape semantics define them.

Reshape therefore does not guarantee coordinate identity preservation for
affected axes.

### 5.7 Transpose

Transpose preserves dimensions and reorders them.

Expected coordinate behavior:

- coordinates SHOULD preserve identity;
- dimension attachment SHOULD remain coherent after reordering;
- default and sibling relationships SHOULD remain the same support relations,
  only moved with their owning dimensions.

### 5.8 swapdims

`swapdims` preserves dimensions while reassigning order.

Coordinate expectations are the same as transpose in principle:

- coordinate identity SHOULD be preserved;
- attachment follows the swapped structural dimensions;
- default and sibling semantics remain attached to the same dimensions.

## 6. Part 5 — Portable Persistence

This contract distinguishes three layers:

```text
Runtime richness
Native persistence
Portable persistence
```

### 6.1 Runtime richness

Runtime richness includes:

- full `CoordSet` grouping semantics;
- default-coordinate selection;
- same-dimension sibling coordinates;
- aliases and compatibility lookup behavior;
- explicit reference metadata;
- coordinate-local display and normalization behavior;
- some internal grouping identity.

This is the richest layer and the primary source of coordinate semantics.

### 6.2 Native persistence

Native persistence SHOULD preserve the full SpectroChemPy-owned coordinate
model as far as the native format allows.

That includes preserving:

- `CoordSet` ownership;
- same-dimension grouping;
- default selection;
- references;
- compatibility-sensitive serialized state needed for faithful reconstruction.

Native persistence is therefore the correct place for exact runtime
reconstruction expectations.

### 6.3 Portable persistence

Portable persistence has a narrower contract.

Portable formats MUST preserve:

- dimension attachment of exported coordinates;
- one default coordinate per dimension;
- coordinate values and units for exported coordinates;
- the distinction between structural dimensions and their support
  descriptions.

Portable formats SHOULD preserve:

- auxiliary same-dimension coordinates where the carrier model supports them;
- coordinate titles;
- explicit role markers such as default versus auxiliary;
- enough metadata to reconstruct the portable coordinate layer predictably.

Portable formats MAY preserve:

- partial reference-related meaning;
- additional auxiliary coordinate metadata;
- bounded reconstruction hints for richer coordinate views.

Portable formats are not required to preserve:

- full runtime alias behavior;
- exact internal grouping identity;
- reference-sharing Python identity;
- every runtime-only `CoordSet` convenience.

### 6.4 xarray and NetCDF

The current portable reference model remains:

```text
NDDataset <-> xarray.Dataset <-> NetCDF
```

Within this model:

- one default coordinate per dimension is the portable minimum;
- auxiliary same-dimension coordinates are a supported best-effort extension;
- richer `CoordSet` topology remains only partially portable.

Portable persistence should preserve the SpectroChemPy coordinate contract as
far as the carrier schema allows. The carrier does not define the contract.

### 6.5 CSDM interoperability

CSDM is an optional interoperability example, not the architectural reference
model for `Coord` or `CoordSet`.

If CSDM interoperability is later implemented, it SHOULD:

- preserve the SpectroChemPy coordinate contract as far as the exchange schema
  allows;
- report lossy narrowing explicitly where `CoordSet` runtime richness exceeds
  the exchange model;
- remain bounded as an optional exchange mapping rather than a replacement for
  native or primary portable persistence.

## 7. Part 6 — Comparison

This section is comparative only.

The contract above is defined from SpectroChemPy runtime semantics, not
derived from external libraries.

### 7.1 NumPy

NumPy has positional axes, but no first-class coordinate ownership layer like
`CoordSet`.

Relative to NumPy, SpectroChemPy adds:

- explicit support objects;
- explicit ownership of support relative to dimensions;
- same-dimension coordinate multiplicity;
- default-coordinate semantics.

NumPy remains the nearest reference point for array shape intuition, but not
for support semantics.

### 7.2 xarray

xarray offers named dimensions and coordinates, but SpectroChemPy still
intentionally differs.

Key differences:

- `CoordSet` is an explicit runtime ownership layer;
- same-dimension multiplicity is more central in SpectroChemPy;
- default-coordinate selection is explicit;
- coordinate grouping and references are richer than the portable minimum;
- SpectroChemPy does not adopt full alignment semantics as the governing model.

xarray is therefore a portable carrier and comparison point, not the runtime
source of truth.

### 7.3 pandas indexes

pandas indexes are central axis identity and alignment objects.

SpectroChemPy intentionally keeps a different split:

- dimensions hold structural axis identity;
- coordinates describe or annotate those axes;
- `CoordSet` organizes multiple coordinate views of the same axis.

Some coordinates may feel index-like in practice, especially label-bearing
ones, but a `Coord` is not simply a pandas `Index`, and `CoordSet` is not just
an index manager.

## 8. Part 7 — Explicit Non-Goals

This RFC does not:

- introduce full coordinate-aware alignment;
- redefine the accepted coordinate-arithmetic contract;
- define metadata propagation for all operation families;
- define label semantics in full;
- redesign `Coord` inheritance;
- redesign `CoordSet` storage internals;
- define migration or implementation sequencing;
- require any immediate public API changes.

It also does not require portable formats to reproduce full runtime `CoordSet`
richness.

## 9. Open Questions

The following questions remain intentionally open:

- how much of `Coord`'s inherited math surface should remain part of the
  long-term maintained contract;
- whether some same-dimension sibling roles should later be classified more
  sharply;
- how much portable persistence should preserve beyond default and auxiliary
  coordinate layers;
- whether coordinate names need a stronger formal identity contract beyond
  current runtime practice;
- whether some coordinate-local metadata should later move under a sharper
  metadata taxonomy.

## 10. Candidate Follow-Up RFCs

- `label-semantics.md`
  Clarify labels as annotations, categorical identifiers, or support-local
  display conveniences.
- `metadata-taxonomy-contract.md`
  Clarify which coordinate-local fields are structural, scientific,
  provenance-related, or presentation-oriented.
- `reader-metadata-normalization-contract.md`
  Clarify how imported external metadata should map into coordinates,
  `CoordSet`, and dataset-level metadata.

## 11. Promotion Candidates

### Suitable for architecture notes

- `maintainers/architecture/coordinate-and-coordset-semantics-reference.md`
  Stable reference note once lifecycle and persistence rules have been adopted
  and no longer primarily belong in RFC space.

### Suitable for dimensional-semantics follow-up

- tighter cross-reference language between structural dimension survival and
  coordinate survival/rebuild/drop rules;
- clarification of which operation families preserve coordinate identity versus
  only dimension identity.

### Suitable for Metadata Taxonomy RFC work

- classification of coordinate-local metadata versus dataset-level metadata;
- boundaries between coordinate names, titles, labels, and structural
  dimension keys;
- persistence distinction between portable coordinate metadata and runtime-only
  `CoordSet` richness.
