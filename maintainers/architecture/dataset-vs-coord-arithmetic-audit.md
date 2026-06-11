# Dataset-vs-Coord Arithmetic Audit

## Status

Completed semantic audit.

This document records the object-model reasoning behind removing or restricting
arithmetic between `NDDataset` and `Coord`.

## Executive Summary

Arithmetic between `NDDataset` and `Coord` existed through shared `NDMath`
machinery, but its scientific meaning was weak and only partially coherent
within the SpectroChemPy object model.

The strongest conclusion is:

```text
Coord is modeled primarily as an explicit axis or support structure,
not as a signal-bearing scientific dataset.
```

That makes `dataset + coord`, `dataset - coord`, `dataset * coord`, and
`dataset / coord` look less like central scientific features and more like
permissive consequences of giving `Coord` and `NDDataset` shared arithmetic
machinery.

## Role of Coord

`Coord` is documented and used as an explicit coordinate for a dataset axis.
It represents support information such as:

- time;
- wavelength;
- wavenumber;
- temperature;
- ppm;
- component index;
- labels along an axis.

It can carry numeric values and units, which makes it array-like. But its
responsibility is to describe where data lives and how an axis should be
interpreted.

## Scientific Meaning

For common spectroscopy workflows, adding or subtracting a coordinate from a
dataset usually has weak scientific meaning. Adding wavenumber to absorbance,
wavelength to intensity, or ppm positions to NMR signal values is not a
standard operation.

Multiplication and division can sometimes be interpreted as weighting or
normalization. Even then, the operand is usually better modeled as signal-like
data sampled along an axis, not as the axis itself.

## Coord vs 1D NDDataset

The semantic boundary is:

```text
Coord:
    Where is the data located?
    How should an axis be interpreted?

1D NDDataset:
    What are the measured or derived values along that axis?
```

Baselines, weighting profiles, instrument responses, calibration curves, and
model outputs are naturally signal-bearing arrays. They fit better as
`NDDataset` objects than as `Coord` objects.

## Historical Interpretation

The strongest evidence points to historical permissiveness:

- shared inheritance from `NDMath`;
- shared operator installation;
- shared unit-aware arithmetic support;
- broad permissiveness once `Coord` participated in the arithmetic machinery.

This does not prove the behavior was accidental, and some users may have relied
on it intentionally. But the codebase provides more evidence for shared
machinery than for an explicitly articulated scientific contract.

## Risk Assessment

Risks of allowing dataset-vs-`Coord` arithmetic:

- confusion between axis support and signal data;
- user misunderstanding of what `Coord` is for;
- silent use of coordinates where a 1D dataset would be clearer;
- surprising broadcasting and result typing inherited from shared machinery.

Risks of removing or restricting it:

- breaking workflows that used `Coord` as a convenient numeric vector;
- reducing flexibility for exploratory transformations;
- requiring users to convert axis-like vectors into `NDDataset` objects for
  arithmetic.

## Classification

The best classification is:

```text
Specialized feature with strong legacy-permissive characteristics.
```

It is not a core feature because the documented role of `Coord` is axis
support, not signal arithmetic.

It is not simply meaningless because multiplication and division can have niche
interpretations and existing users may rely on the behavior.

## Maintainer Guidance

Future discussions should start from this question:

```text
When SpectroChemPy needs a 1D numeric operand aligned with an axis,
what should make that operand a Coord,
and what should make it a 1D NDDataset instead?
```

That boundary is the real pressure point behind dataset-vs-`Coord` arithmetic.

