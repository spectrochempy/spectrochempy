# NDMath Maintainability Audit

## Status

Deferred technical risk map.

This document is not the mathematical semantics contract. For operation
behavior, result assembly, scientific object identity, provenance, and metadata
propagation, see
[`mathematical-semantics-and-metadata-propagation.md`](mathematical-semantics-and-metadata-propagation.md).

For class-level ownership and possible future responsibility splits, see
[`array-class-responsibility.md`](array-class-responsibility.md).

No immediate `NDMath` refactor is planned.

## Purpose

This audit records why `NDMath` is risky to change and which internal areas
future maintainers should approach carefully.

It deliberately avoids restating the full semantic model now captured by the
mathematical semantics RFC.

## Current Responsibility Concentration

`NDMath` currently participates in:

- public NumPy-like methods and reductions;
- Python operator dispatch;
- NumPy ufunc dispatch;
- operand ordering and result-type selection;
- unit compatibility and result-unit inference;
- mask extraction and propagation;
- coordinate compatibility checks;
- result construction for datasets, coordinates, quantities, scalars, masked
  arrays, and raw NumPy arrays;
- history/title updates;
- plugin execution hooks.

This concentration is not automatically wrong. It reflects SpectroChemPy's
domain needs. The maintenance risk is that numerical execution, validation,
result assembly, and metadata-adjacent updates are intertwined in the same
shared paths.

## Main Internal Paths

Binary operators and ufuncs use the central arithmetic path:

```text
__array_ufunc__() or generated operator
    ->
_check_order()
    ->
_op()
    ->
_preprocess_op_inputs()
    ->
_prepare_operation_quantities()
    ->
_resolve_operation_units()
    ->
_execute_operation()
    ->
_op_result()
```

Decorated NumPy-method wrappers use a separate path:

```text
public method decorated by _from_numpy_method
    ->
descriptor decides call mode
    ->
copy or construct an object
    ->
coerce data when needed
    ->
wrapped method mutates result
    ->
decorator adds history for selected cases
```

These paths overlap in purpose but not in structure. That overlap is the main
reason result assembly should be characterized before maintainability cleanup.

## Necessary Complexity

The following complexity is inherent to the project domain:

- unit-aware arithmetic;
- unit conversion for compatible operands;
- dimensionality errors for incompatible units;
- masked-array behavior;
- NumPy broadcasting;
- interaction between `NDDataset`, `Coord`, scalar, NumPy array, and
  `Quantity`;
- ufunc domain rules for dimensionless, logarithmic, and trigonometric inputs;
- coordinate compatibility for dataset-vs-dataset arithmetic;
- reductions that may return scalar `Quantity` or reduced datasets;
- generic plugin dispatch without plugin-specific semantics in core.

## Historical Complexity

The following complexity appears more historical than essential:

- two broad result pathways: `_op_result()` and `_from_numpy_method`;
- operand ordering logic split across `_preprocess_op_inputs()` and
  `_check_order()`;
- direct result mutation repeated across reduction and constructor wrappers;
- mixed responsibility in `_prepare_operation_quantities()`;
- constructors, reductions, operators, ufunc dispatch, and plugin hooks living
  in one file;
- comments and names that no longer fully match behavior.

One concrete example is `_COORDINATE_POLICY`: the nearby comment describes
multi-dimensional operands as being compared on every dimension, while current
implementation and tests document last-dimension validation.

## Risk Areas

Low-risk areas:

- adding documentation;
- adding characterization tests;
- adding narrow public behavior tests.

Medium-risk areas:

- changing one public NumPy-like wrapper;
- changing one reduction family;
- updating history/title behavior for one operation family;
- adjusting plugin hook plumbing while preserving fallback behavior;
- modifying mask propagation for a narrow operation family.

High-risk areas:

- `_op()`;
- `_preprocess_op_inputs()`;
- `_check_order()`;
- `_prepare_operation_quantities()`;
- `_resolve_operation_units()`;
- `_check_coordinate_compatibility()`;
- `_op_result()`;
- broad reduction result assembly.

Changes to high-risk areas should be preceded by focused characterization tests
covering scalar arithmetic, dataset-vs-dataset arithmetic, dataset-vs-`Coord`
arithmetic, NumPy array interop, compatible and incompatible units, masked
operands, coordinate mismatch behavior, reductions, and in-place operations.

## Possible Cleanup Directions

Future maintainability work may consider:

- separating public NumPy-method wrapping from core arithmetic execution;
- separating unit preparation from coordinate compatibility checks;
- reducing overlap between `_check_order()` and `_preprocess_op_inputs()`;
- centralizing reduction result assembly rules;
- documenting allowed result types by operation family;
- clarifying ufuncs that intentionally return raw arrays;
- isolating plugin extension points from ordinary execution flow.

These are cleanup directions, not implementation recommendations for the
current PR.

## Recommendation

Do not refactor `NDMath` yet.

Recommended sequencing:

```text
characterize mathematical behavior
    ->
decide result assembly / metadata contracts
    ->
identify safe helper boundaries
    ->
perform small internal cleanup
```

This document should remain a technical risk map. The semantic source of truth
should be the mathematical semantics RFC.
