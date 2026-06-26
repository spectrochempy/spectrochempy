[Maintainers](../../README.md) · [Architecture](../INDEX.md)

# Array Class Responsibility

Status: Draft RFC

## Purpose

This document records the current maintainer understanding of responsibilities
across SpectroChemPy's core array classes:

- `NDArray`
- `NDComplexArray`
- `Coord`
- `NDDataset`
- `NDMath`
- `NDIO`

It exists to support the next architecture discussion around mathematical
semantics, labels, metadata propagation, and possible future responsibility
splits.

This document does not decide on a redesign.

In particular, an `NDLabelled`-style layer is treated here as a candidate future
direction, not as an accepted implementation plan.

## Current Class Hierarchy

The current runtime hierarchy is:

```text
NDArray
NDComplexArray -> NDArray
Coord          -> NDMath -> NDArray
NDDataset      -> NDMath -> NDIO -> NDComplexArray -> NDArray
```

Important consequences:

- `Coord` does not inherit `NDComplexArray`, so it does not directly inherit
  the main complex-data layer.
- `Coord` does inherit `NDMath`, so it shares much of the mathematical surface
  used by `NDDataset`.
- `NDDataset` receives math behavior from `NDMath`, persistence behavior from
  `NDIO`, complex behavior from `NDComplexArray`, and base array behavior from
  `NDArray`.
- `CoordSet` is not part of the inheritance chain. It is owned by `NDDataset`
  through `_coordset` and stores `Coord` objects.

The current model is therefore not a simple "array core plus orthogonal
capabilities" design. It is a historically layered design where subclasses
inherit broad behavior and then specialize, restrict, or delegate parts of it.

## Responsibility Inventory

### NDArray

`NDArray` is the broad base object. It currently owns more than raw nD array
storage.

Responsibilities include:

- data storage and the `data` setter;
- dtype handling for base arrays;
- mask storage and masked-data access;
- labels for 1D labelled arrays;
- units and unit conversion helpers;
- dimension names;
- title, name, and id;
- generic `Meta` metadata;
- copy/equality/hash behavior;
- base slicing and location-to-index behavior;
- basic display and HTML conversion.

This makes `NDArray` convenient as a shared base, but it also means labels,
metadata, units, masks, and display are all available before subclasses decide
whether those concepts are appropriate.

### NDComplexArray

`NDComplexArray` adds complex-data behavior on top of `NDArray`.

Responsibilities include:

- complex dtype normalization;
- `has_complex_dims`, `is_complex`, and `is_interleaved`;
- conversion of interleaved real arrays to complex arrays;
- `real`, `imag`, and complex-aware limits;
- complex-aware terminal and HTML display;
- generic plugin-facing display hooks for special complex-like backends.

Recent hypercomplex work has moved quaternion semantics back into the
hypercomplex plugin while keeping only generic display dispatch and fallback in
core. That direction is consistent with this responsibility boundary:

```text
core    -> generic complex/display infrastructure
plugin  -> hypercomplex/quaternion semantics
```

### Coord

`Coord` represents a dataset axis/support object. It inherits from `NDMath` and
`NDArray`, but narrows or rejects several inherited behaviors.

Responsibilities include:

- 1D coordinate values and/or coordinate labels;
- coordinate rounding and linearization;
- coordinate reversal policy, including plugin override via `coord.reversed`;
- unit-bearing axis values;
- coordinate display;
- label-bearing axis semantics;
- restricted slicing and copying behavior.

`Coord` explicitly rejects several operations inherited from the broader math
surface, including reductions and operations that are not meaningful for axis
objects. It also forces masks to `NOMASK` and reports `is_complex = False`.

This is a useful semantic boundary, but it is currently expressed through
overrides and runtime blocks rather than through a narrow coordinate-specific
base class.

### NDDataset

`NDDataset` is the main signal/data object.

Responsibilities include:

- data storage inherited through `NDComplexArray` / `NDArray`;
- complex and plugin-backed hypercomplex support;
- coordinate ownership through `CoordSet`;
- dataset metadata such as author, description, origin, history, timestamps,
  parent project, and timezone;
- slicing with coordinate propagation;
- sorting along coordinates;
- semantic display sections for summary, data, and dimensions;
- native save/load through `NDIO`;
- plugin namespace accessors and lazy method access.

`NDDataset` explicitly disables direct data labels with `_labels_allowed =
False`. Its user-facing labels normally live in its coordinates, not on
`NDDataset._labels`.

### NDMath

`NDMath` is the shared mathematical behavior layer.

Responsibilities include:

- NumPy ufunc integration;
- operator generation;
- many NumPy-like methods and reductions;
- unit compatibility and result unit resolution;
- mask propagation;
- result object assembly;
- title and history updates;
- partial coordinate reduction and propagation;
- plugin execution branches for non-core numeric backends.

This layer is powerful, but it combines numerical execution with metadata,
units, coordinates, masks, object identity, and plugin behavior. That is the
main reason mathematical semantics should be clarified before any hierarchy
split.

### NDIO

`NDIO` provides native persistence for dataset-like objects.

Responsibilities include:

- filename, directory, suffix, and filetype handling;
- native `.scp` / `.pscp` save and load;
- JSON serialization and deserialization;
- restoration of nested `CoordSet`, `NDDataset`, `Project`, and script state.

`NDDataset._attributes_()` and `NDIO.loads()` are compatibility-sensitive.
Any future class split must preserve serialized keys and restoration behavior.

## Responsibility Mismatches

### Labels Live Too Low

`NDArray` owns label storage and label-aware behavior. This is useful for
`Coord`, but less natural for `NDDataset`.

In practice there are two label concepts:

- direct labels on an `NDArray`-like object;
- coordinate labels stored through `Coord` objects in a `CoordSet`.

`NDDataset` conceptually needs the second model. It disables direct labels but
still inherits the property and much of the machinery. This is the strongest
case for considering a future `NDLabelled` or equivalent capability layer.

### Coord Shares a Broad Math Surface

`Coord` needs some arithmetic-like behavior, but it is an axis/support object,
not a signal-bearing dataset. The current implementation handles that by
overriding or rejecting operations that do not make sense for coordinates.

That works, but it means the coordinate contract is not obvious from the class
graph. Maintainers need to read both `Coord` and `NDMath` to know which math
operations are meaningful.

### Masks Are Inherited Then Neutralized by Coord

Mask storage is part of `NDArray`, but coordinates cannot be masked in the
current model. `Coord` therefore reports `is_masked = False` and forces `mask`
to `NOMASK`.

This is semantically reasonable, but structurally it is another inherited
responsibility that the subclass must disable.

### Metadata and Math Are Coupled

`NDMath` does not only compute numbers. It also participates in:

- result object construction;
- unit propagation;
- mask propagation;
- coordinate propagation;
- title updates;
- history updates;
- plugin dispatch.

This means the behavior of an operation depends on both numerical semantics and
object construction strategy. That is convenient, but it makes the contract
harder to reason about.

### Serialization Depends on Attribute Lists

`NDArray`, `Coord`, `NDDataset`, and `NDIO` each define or contribute to
`_attributes_()` behavior.

Those lists influence copying, comparison, and persistence. `NDDataset` also has
ordering assumptions in its attribute list for native save/load behavior.

Any responsibility split must treat `_attributes_()` as a compatibility
surface, not as an internal detail.

## Math Semantics Impact

The class hierarchy affects mathematical behavior because math is not purely
numeric in SpectroChemPy.

Binary arithmetic passes through `NDMath`, which handles:

- operand extraction;
- object type priority;
- unit compatibility;
- coordinate compatibility;
- plugin execution branches;
- masks;
- result assembly.

Ufuncs also route through `NDMath`, then update titles and metadata depending
on operation categories.

Reductions combine numerical reduction with dimension and coordinate changes.
Some reduction helpers inspect `coordset` dynamically rather than relying on a
narrow coordinate-aware protocol.

The effect is that mathematical behavior, metadata propagation, coordinate
propagation, and class responsibility are intertwined. This is why mathematical
semantics should be clarified before moving responsibilities between classes.

## Candidate Separation Models

### Documentation and Characterization Only

This would preserve the current hierarchy and improve maintainability through
tests and documentation.

Benefits:

- low risk;
- preserves public behavior;
- gives maintainers a clearer map.

Costs:

- does not reduce structural coupling;
- future contributors still inherit the same broad class responsibilities.

This is necessary, but probably not sufficient long term.

### Small Internal Cleanup Without a New Public Layer

This would keep the public hierarchy stable while introducing clearer internal
helpers or protocols for:

- label-bearing behavior;
- coordinate-bearing behavior;
- complex-capable behavior;
- result assembly;
- metadata propagation.

Benefits:

- lower risk than hierarchy changes;
- can happen incrementally;
- can support math semantics cleanup.

Costs:

- responsibilities remain physically located in the same classes;
- may not fully resolve the label boundary.

This is the best near-term implementation style if maintenance pressure appears.

### `NDLabelled` or Equivalent

A future labelled layer could own:

- `_labels`;
- `labels`;
- label validation;
- label-aware sorting;
- label-aware slicing;
- label-based location lookup.

Likely consumers:

- `Coord`;
- possibly `NDArray` as a compatibility facade;
- not directly `NDDataset`, except through compatibility or delegation to
  coordinates.

Benefits:

- clarifies that labels are primarily coordinate/axis semantics;
- reduces the need for `_labels_allowed = False`;
- gives math semantics a clearer distinction between signal data and labelled
  axes.

Costs:

- `NDArray.labels` is public;
- serialization, copying, equality, slicing, display, and readers currently
  know about labels;
- traitlets and MRO behavior could be affected.

This is worth serious consideration, but only as a staged compatibility
preserving extraction.

### `NDArrayCore` Plus Compatibility Facade

A deeper split could introduce an internal `NDArrayCore` for raw data, dtype,
shape, mask, and minimal indexing, while keeping public `NDArray` as a
compatibility facade.

Benefits:

- clearer conceptual model;
- public `NDArray` symbol could remain stable.

Costs:

- broad migration;
- high serialization risk;
- high test burden;
- likely premature before math semantics are clarified.

This should not be the first step.

## Risks and Compatibility

Any separation work must account for:

- MRO changes around `__init__`, traitlets defaults, validators, and properties;
- public compatibility of `NDArray.labels`, `Coord.labels`, and dataset
  coordinate labels;
- native serialization keys and attribute ordering;
- plugin accessors, math hooks, and display hooks;
- existing behavior of units, masks, metadata, and coordinates in operations;
- performance of slicing, reductions, binary operations, and save/load.

The largest practical risk is mixing behavior changes with class movement.
Maintainers should avoid combining a hierarchy change with a math semantics
change in the same PR.

## Relation to Mathematical Semantics Work

The related document
[`mathematical-semantics-and-metadata-propagation.md`](mathematical-semantics-and-metadata-propagation.md)
focuses on how operations should behave.

This document focuses on where responsibilities currently live.

The recommended order is:

1. characterize current mathematical behavior;
2. clarify the intended operation semantics;
3. identify result assembly and metadata propagation contracts;
4. only then consider whether a responsibility split such as `NDLabelled` is
   justified.

This ordering matters because class hierarchy cleanup should preserve a known
behavioral contract, not accidentally define one.

## Recommendation

Recommendation: **consider a future `NDLabelled` or equivalent layer, but do
not introduce it yet.**

The current hierarchy is serviceable, and there is no evidence that an immediate
deep redesign is required. However, labels and math/result assembly are
entangled enough that future maintainability work should keep responsibility
separation on the roadmap.

Near-term guidance:

- do not change the public hierarchy as part of the next math semantics work;
- document and test current math behavior first;
- keep hypercomplex semantics plugin-owned;
- treat `NDLabelled` as a candidate staged extraction, not a decision;
- preserve serialization compatibility and plugin APIs.

## Open Questions

- Should direct `NDArray.labels` remain a permanent public concept, or become a
  compatibility facade over a more explicit labelled capability?
- Should `Coord` keep inheriting the full `NDMath` surface, or should coordinate
  math become a narrower protocol?
- Should result assembly be extracted before any class split?
- Which metadata fields should be preserved, recomputed, overridden, or dropped
  for each operation family?
- Should complex/hypercomplex display and math hooks remain fully plugin-owned
  for non-core numeric backends?
- What characterization tests are required before any internal responsibility
  extraction can be considered safe?
