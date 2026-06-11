# NDMath Maintainability Audit

## Status

Deferred maintenance reference.

No immediate `NDMath` refactor is planned. This document maps the risks and
responsibilities that future maintainers should account for before changing
shared arithmetic behavior.

## Executive Summary

`NDMath` is difficult to maintain because it concentrates several independent
responsibilities in one long mixin:

- public NumPy-like constructors and reductions;
- Python operator dispatch;
- NumPy ufunc dispatch;
- unit compatibility and result-unit inference;
- mask extraction and propagation;
- coordinate compatibility checks;
- result typing and result construction;
- metadata-adjacent updates such as history and title changes;
- plugin extension hooks.

The core arithmetic flow is coherent:

```text
operator / ufunc
    ->
operand ordering and return-type selection
    ->
unit and coordinate preparation
    ->
unit resolution
    ->
NumPy execution
    ->
mask extraction
    ->
copy-based result construction
```

The maintainability problem is that this flow is embedded alongside older
NumPy-method wrappers, result-shape mutations, reduction-specific coordinate
rules, history/title handling, and compatibility special cases.

## Responsibility Inventory

`NDMath` currently owns:

- public NumPy-like APIs such as `mean`, `sum`, `zeros`, `ones`, `linspace`,
  `diag`, and `coordmax`;
- module-level API export through `_update_api_funclist()`;
- Python arithmetic operator installation through `_set_operators()`;
- NumPy wrapping and `__array_ufunc__`;
- operand order normalization and result-type priority;
- unit compatibility, unit conversion, and result-unit inference;
- coordinate compatibility checks;
- coordinate updates for reductions and coordinate extrema;
- mask detection and propagation;
- result construction for `NDDataset`, `Coord`, `Quantity`, scalars, masked
  arrays, and raw NumPy arrays;
- history/title updates;
- plugin execution hooks.

## Main Execution Paths

Binary operators and ufuncs use a relatively centralized path:

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

Decorated NumPy-method wrappers use a different path:

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

These two result-construction paths overlap in purpose but not in structure.

## Necessary Complexity

The following complexity is inherent to SpectroChemPy's domain:

- unit-aware arithmetic;
- dimensionality errors for incompatible units;
- unit conversion for compatible operands with different scales;
- masked-array behavior;
- NumPy broadcasting;
- interaction between `NDDataset`, `Coord`, scalar, NumPy array, and
  `Quantity`;
- ufunc domain rules for dimensionless, logarithmic, and trigonometric inputs;
- coordinate compatibility for dataset-vs-dataset arithmetic;
- reductions that may return scalar `Quantity` or reduced datasets.

## Historical Complexity

The following complexity appears more historical than essential:

- two broad result pathways: `_op_result()` and `_from_numpy_method`;
- operand ordering logic in both `_preprocess_op_inputs()` and `_check_order()`;
- direct result mutation repeated across reduction and constructor wrappers;
- mixed responsibility in `_prepare_operation_quantities()`;
- constructors, reductions, operators, ufunc dispatch, and plugin hooks living
  in one file;
- comments and names that no longer fully match behavior.

One concrete example is `_COORDINATE_POLICY`: the comment near its definition
describes multi-dimensional operands as being compared on every dimension,
while implementation and tests document last-dimension validation.

## Risk Areas

Low-risk areas:

- adding characterization tests;
- adding documentation or audit notes;
- adding narrow public behavior tests.

Medium-risk areas:

- changing public NumPy-like wrappers;
- changing one reduction family;
- updating history/title behavior;
- adjusting plugin hooks;
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

Changes to high-risk areas should be preceded by focused characterization
tests for scalar arithmetic, dataset-vs-dataset arithmetic, dataset-vs-`Coord`
arithmetic, NumPy array interop, compatible and incompatible units, masked
operands, coordinate mismatch behavior, reductions, and in-place operations.

## Potential Simplification Areas

Future investigation may consider:

- separating public NumPy-method wrapping from core arithmetic execution;
- separating unit preparation from coordinate compatibility checks;
- reducing overlap between `_check_order()` and `_preprocess_op_inputs()`;
- centralizing reduction result assembly rules;
- documenting allowed result types by operation family;
- clarifying ufuncs that intentionally return raw arrays;
- isolating plugin extension points from ordinary execution flow.

## Recommended Next Investigation

The next useful investigation should be a characterization map, not a refactor:

```text
operation family
    ->
input types
    ->
result type
    ->
units behavior
    ->
mask behavior
    ->
coordset behavior
    ->
history/title behavior
```

This would distinguish stable public behavior from incidental internal
structure before any responsibility extraction begins.
