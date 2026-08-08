[Maintainer Docs](../README.md) · [RFC Index](INDEX.md)

# Coordinate Arithmetic Semantics

## Status

Accepted maintainer RFC.

This document records the current maintainer position on coordinate arithmetic
semantics in SpectroChemPy.

It may evolve based on practical experience, characterization work, and future
maintainer review.

This document is normative in intent, but not immutable. It defines the
current semantic position future discussions should start from.

## Purpose

The purpose of this RFC is to make SpectroChemPy's coordinate arithmetic model
explicit.

The problem is not that the current behavior is known to be wrong. The problem
is that its rationale, scope, and limits were not previously documented as a
maintainer position.

This RFC therefore documents:

- what coordinate arithmetic semantics SpectroChemPy currently relies on;
- why that model remains acceptable for now;
- which future evolution space remains open.

## Current Model

SpectroChemPy currently relies on the following arithmetic model:

```text
NumPy broadcasting
+
unit-aware arithmetic
+
last-dimension coordinate validation for dataset-vs-dataset arithmetic
```

Within this model:

- dimensions are not alignment keys;
- coordinate names are not compatibility keys;
- coordinate titles are not compatibility keys;
- coordinate units are not currently reconciled as part of compatibility;
- earlier-dimension coordinates are not validated in arithmetic.

This is a coherent but intentionally narrow model.

## Guiding Position

SpectroChemPy remains primarily a spectroscopy-first scientific dataset
library, not a general labeled-array alignment framework.

That position matters for arithmetic semantics. In many classic spectroscopy
workflows, the last dimension is the primary spectral axis. Treating that axis
as scientifically special is therefore a practical and historically grounded
choice.

The current model is accepted as a pragmatic spectroscopy-oriented compromise:

- permissive enough to preserve established workflows;
- protective enough to prevent a common class of spectral-grid mistakes;
- narrow enough to remain explainable without adopting a full alignment model.

This should not be read as a claim that the current model is final or ideal in
all cases.

## What This Contract Does Not Do

This RFC does not:

- introduce all-dimension validation;
- introduce dimension-name alignment;
- introduce coordinate-aware alignment;
- remove the existing last-dimension validation;
- define implementation details.

This RFC also does not define:

- metadata behavior;
- result assembly behavior;
- CoordSet redesign;
- display semantics;
- API changes.

## Accepted Current Semantics

The following semantics are accepted for now.

- Scalar arithmetic follows NumPy-like broadcasting behavior.
- Dataset-vs-dataset arithmetic follows NumPy-like shape semantics plus
  last-dimension coordinate validation.
- Metadata behavior is governed separately by Metadata Contract v1.
- Result assembly remains a separate architectural concern.

This means SpectroChemPy is neither a pure NumPy arithmetic model nor a full
coordinate-aware alignment system.

## Why Last-Dimension Validation Exists

Last-dimension validation exists because, in many spectroscopy workflows, the
last dimension is the primary spectral axis.

That makes some failures scientifically important even when array shapes remain
compatible. Adding or subtracting spectra sampled on different spectral grids
may be numerically possible while still being scientifically misleading.

Validating the last-dimension coordinate is therefore a useful low-cost
guardrail. It catches an important class of spectroscopy-specific mistakes
without imposing a full coordinate-alignment system on all arithmetic.

This guardrail is valuable, but limited. It does not cover all scientific
risks, and it should not be interpreted as proof that all relevant coordinates
were validated.

## Relationship to NumPy

SpectroChemPy intentionally follows NumPy-like broadcasting for shape
semantics.

It diverges from NumPy by adding a coordinate guardrail for
dataset-vs-dataset arithmetic. That divergence is intentional and
scientifically motivated.

The current project position is therefore:

- shape behavior should remain familiar to NumPy users;
- scientific datasets may justify extra compatibility checks;
- those checks should remain understandable and proportional to the domain.

## Relationship to xarray-style Alignment

SpectroChemPy does not currently implement named-dimension alignment.

It also does not currently implement coordinate joins or automatic coordinate
alignment.

xarray-style alignment represents a different design point: richer dimension
semantics, richer compatibility rules, and a different balance between
automation, protection, and complexity.

Adopting those semantics would be a major project-level change. It is not
recommended at this stage.

The current maintainer position is that SpectroChemPy should not move toward
full xarray-style coordinate alignment at this time.

## Future Evolution Space

The current model is acceptable, but it does not close future evolution.

Plausible future directions include:

- keeping the current model and documenting it more clearly;
- strengthening validation on more dimensions in selected cases;
- introducing optional stricter modes;
- adding limited dimension-aware checks;
- defining operation-specific compatibility rules.

This RFC does not choose among those paths. It only defines the current
maintainer position and the boundaries within which future discussion should
occur.

## Risks and Tradeoffs

The current model carries real tradeoffs.

- Last-dimension-only validation may create false confidence if users assume
  that all relevant coordinates were checked.
- Broader validation across all dimensions could introduce excessive friction
  for transformed, reduced, or analysis-derived datasets.
- Moving toward pure NumPy semantics would reduce friction but would also
  remove a valuable scientific guardrail.
- Moving toward hybrid or alignment-style semantics could increase scientific
  protection while also increasing semantic complexity and maintenance burden.

No currently visible direction eliminates all of these tradeoffs at once.

## Open Questions

The following questions remain open:

- Should last-dimension privilege remain a permanent rule?
- Should coordinate units participate in compatibility decisions?
- Should dataset-vs-Coord arithmetic get explicit semantic rules?
- Should transformed or analysis-derived datasets have relaxed compatibility
  rules?
- Should strict validation modes exist?

These questions are intentionally left open for future maintainer review.

## Adoption Guidance

Future pull requests affecting coordinate arithmetic should:

- reference this RFC;
- state whether they preserve or change the current model;
- include characterization tests for the behavior they rely on or modify;
- avoid broad semantic changes without explicit maintainer review.

The current maintainer position is therefore:

- SpectroChemPy should not move toward full xarray-style coordinate alignment
  at this stage.
- The current last-dimension validation model remains acceptable and valuable
  for classic spectroscopy workflows.
- Future evolution should preserve scientific safety while avoiding
  unnecessary friction for transformed, reduced, and analysis-derived
  datasets.
