# Coordinate Arithmetic Decision Audit

## Status

Historical decision-space audit.

The current maintainer position is recorded in
`../rfcs/coordinate-arithmetic-semantics.md`. This document preserves the design
space and tradeoffs considered before that RFC.

## Executive Summary

SpectroChemPy currently occupies a middle position:

```text
NumPy broadcasting
    +
unit-aware arithmetic
    +
last-dimension coordinate validation
```

That position is coherent enough to describe, but not complete enough to serve
as a permanent arithmetic philosophy without explicit maintainer review.

The decision space is not binary. The project can choose among several models,
each trading off scientific safety, user friction, backward compatibility,
explainability, maintenance burden, and fit with spectroscopy workflows.

## Current Position

SpectroChemPy is more protective than NumPy because dataset-vs-dataset
arithmetic can reject mismatched last-dimension coordinates.

It is less semantic than a full dimension-aware or coordinate-aware system
because it ignores dimension names, earlier dimensions, coordinate names,
coordinate titles, and coordinate units for compatibility.

The strongest implicit assumption is:

```text
the last dimension is scientifically special
```

That assumption fits many spectroscopy workflows, but not all transformed or
multi-dimensional workflows.

## Candidate Models

### Preserve Current SpectroChemPy Behavior

```text
NumPy broadcasting
+
last-dimension coordinate validation
```

Strengths:

- preserves established behavior;
- protects against common spectral-axis mistakes;
- remains explainable for existing users;
- keeps maintenance burden low.

Weaknesses:

- leaves earlier dimensions under-specified;
- treats multi-dimensional datasets asymmetrically;
- makes dimension names and richer coordinate structure mostly irrelevant.

### Pure NumPy Semantics

```text
NumPy broadcasting only
```

This model maximizes array predictability and implementation simplicity, but it
removes the coordinate guardrail that currently prevents combining spectra
sampled on incompatible axes.

It is strongest for plain array semantics and weakest for spectroscopy-specific
scientific safety.

### All-Dimension Coordinate Validation

```text
NumPy broadcasting
+
coordinate validation on every dimension
```

This model is stronger and cleaner than the current rule, but it may reject
workflows that currently succeed and that users perceive as valid.

It remains positional in spirit unless paired with dimension-aware rules.

### Dimension-Aware Semantics

```text
dims become arithmetic-semantic
```

Dimension names could influence compatibility or alignment. This may reduce
accidental arithmetic across differently intended axes, but it introduces new
questions about whether labels should override array order and how much trust
to place in user-assigned names.

### Coordinate-Aware Semantics

Coordinates become first-class compatibility objects.

This gives the strongest scientific protection but also the highest complexity,
largest migration cost, and greatest risk of frustrating users who expect
permissive array arithmetic.

### Hybrid Model

A hybrid model could preserve NumPy broadcasting while strengthening selected
semantic checks.

This is likely the most realistic long-term direction if the current model
proves insufficient, but it needs concrete requirements before design work
starts.

## Strategic Conclusion

SpectroChemPy cannot maximize all goals at once.

The project must choose where it wants to sit between:

- permissive array arithmetic;
- protective spectroscopic semantics;
- generalized labeled scientific arithmetic.

For now, the current spectroscopy-oriented compromise remains acceptable.
Future changes should be driven by demonstrated scientific requirements rather
than by architectural preference alone.
