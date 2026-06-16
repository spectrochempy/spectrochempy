# Coordinate Arithmetic Audit

## Status

Completed architecture audit.

This document preserves the implementation map behind the
`../rfcs/coordinate-arithmetic-semantics.md` maintainer RFC.

## Executive Summary

SpectroChemPy currently uses a partially centralized but semantically narrow
coordinate arithmetic model:

```text
NumPy-style positional arithmetic
    +
unit-aware scalar/array handling
    +
last-dimension-only coordinate compatibility for dataset-vs-dataset arithmetic
```

The main source of truth is `NDMath`, especially
`src/spectrochempy/core/dataset/arraymixins/ndmath.py`.

The strongest finding is that coordinate compatibility is intentionally limited
to the last dimension for dataset-vs-dataset operations. Dimension names,
earlier dimensions, coordinate names, titles, and coordinate units do not
currently govern compatibility.

## Current Compatibility Rule

For arithmetic such as `A + B`, `A - B`, `A * B`, and `A / B`, the coordinate
compatibility check is only applied when both operands are treated as
`NDDataset` objects.

When both operands are datasets:

- exact `CoordSet` equality short-circuits the check;
- otherwise, only the coordinate on the last dimension of each operand is
  compared;
- the comparison uses `assert_coord_almost_equal(..., decimal=3, data_only=True)`.

The current rule is therefore:

- coordinate-value-based on the last dimension;
- approximate to three decimals;
- data-only, not metadata-rich.

## What Is Not Checked

The current model does not check:

- all dimensions;
- dimension name alignment;
- coordinate names;
- coordinate titles;
- coordinate units as a compatibility rule;
- earlier dimensions in multi-dimensional datasets.

There is no explicit rule for named-dimension alignment, partial coordinate
reconciliation, or distinguishing coordinate mismatch from shape mismatch at a
higher semantic level.

## Role of Dimensions

Dimensions are used to locate the last-dimension coordinate and to describe
array structure. They are not arithmetic alignment keys.

Matching dimension names do not by themselves make operands compatible, and
mismatched dimension names do not by themselves reject an operation. Shape and
last-dimension coordinate behavior dominate.

Today, dimensions are closer to structured labels than to arithmetic-semantic
alignment keys.

## Role of Coordinates

Coordinates influence arithmetic, but only narrowly.

The only coordinate property that directly affects binary dataset-vs-dataset
arithmetic is numeric coordinate data on the last dimension.

Ignored properties include:

- coordinate names;
- coordinate titles;
- coordinate units;
- earlier-dimension coordinate values;
- most `CoordSet` structure beyond the equality shortcut.

## Broadcasting Semantics

SpectroChemPy follows NumPy broadcasting closely for shape semantics, with a
small coordinate-aware overlay.

`dataset + scalar` follows NumPy scalar broadcasting and preserves coordinates
from the dataset result path.

`dataset + dataset` follows NumPy shape compatibility, then applies the
last-dimension coordinate check when both operands are datasets.

`dataset + reduced_dataset` remains shape-driven and last-dimension-centered;
reduced dimensions are not reconciled by name.

`dataset + coord` does not participate in the dataset-vs-dataset coordinate
compatibility check. That behavior is documented separately in
`dataset-vs-coord-arithmetic-audit.md`.

## Ambiguous Cases

The most ambiguous current cases are:

- multi-dimensional datasets whose earlier dimensions disagree but last
  dimensions match;
- dataset-vs-`Coord` arithmetic;
- coordinate units that differ but represent comparable axes;
- dimension names that suggest mismatch while shape and last-axis coordinates
  allow the operation.

## Sources of Truth

The primary source of truth is:

- `NDMath._preprocess_op_inputs()`;
- `NDMath._prepare_operation_quantities()`;
- `NDMath._check_coordinate_compatibility()`;
- `NDMath._resolve_operation_units()`;
- `NDMath._execute_operation()`;
- `NDMath._op()`.

Secondary contributors are `NDDataset`, `CoordSet`, and `Coord`, which provide
dimensions, coordinate access, equality semantics, and coordinate values.

The system is centralized in execution but incomplete in semantic coverage.

## Ecosystem Position

Relative to NumPy, SpectroChemPy is stricter because last-dimension coordinate
mismatch can reject an otherwise shape-compatible operation.

Relative to xarray, SpectroChemPy is much less semantic because it has no
general alignment rule and does not treat dimension names or coordinates as
full arithmetic compatibility objects.

The current model sits between permissive positional arithmetic and rich
labeled-array semantics.

## Decision Points

Future coordinate arithmetic work should explicitly decide:

- whether the last-dimension-only rule is legacy behavior to preserve or
  transitional behavior to replace;
- whether dimensions should remain labels or become arithmetic-semantic;
- whether coordinate units should participate in compatibility;
- whether dataset-vs-`Coord` arithmetic should have explicit rules;
- whether multi-dimensional compatibility should remain partial or become
  all-dimension-aware;
- how much deviation from NumPy broadcasting is acceptable.
