[Maintainer Docs](../README.md) · [Architecture Index](INDEX.md)

# FitResult Scientific Contract

## Status

Implemented architecture note.

This note is the authoritative maintainer reference for the
current public scientific contract of `FitResult` after the completed 2026-07
enrichment campaign.

## Purpose

This note records the maintained answer to one practical question:

```text
After an optimization has run,
what scientific information belongs on FitResult,
and how is it surfaced?
```

It does not redesign `Optimize`, `ResultBase`, or the future parameter-solution
surface. Its job is narrower: document the stable `FitResult` contract that has
now emerged.

For the broader Result Object architecture, see
[`result-object-architecture.md`](result-object-architecture.md).

## Current Role

`FitResult` is the stable scientific result object exposed by `Optimize.result`.

It is not:

- the optimizer backend object
- the raw solver result
- the fit-configuration object
- the future parameter-solution abstraction

It is the grouped scientific interpretation of one completed fit.

Its current structure is:

```text
FitResult
│
├── outputs
├── diagnostics
├── covariance
├── variance
├── stderr
├── correlation
└── confidence_intervals
```

This split is deliberate. Scalar or small summary values stay grouped in
`diagnostics`. Matrix-valued uncertainty objects are exposed as dedicated
top-level properties.

## Outputs

`FitResult.outputs` is the primary grouped surface for dataset-like products of
the fit.

The maintained output names are:

- `fitted`
- `residuals`
- `components`

Attribute-style access remains supported through `ResultBase`:

- `result.fitted`
- `result.residuals`
- `result.components`

These outputs are scientific arrays and therefore remain naturally aligned with
`NDDataset`.

## Diagnostics

`FitResult.diagnostics` is the primary grouped surface for fit-quality and
solver-summary metrics.

The maintained diagnostics currently include:

- `cost`
- `niter`
- `ncalls`
- `n_observations`
- `n_varying_parameters`
- `degrees_of_freedom`
- `rss`
- `sse`
- `rmse`
- `r_squared`
- `reduced_chi_square`
- `adjusted_r_squared`
- `aic`
- `bic`
- `success`
- `status`
- `message`

The grouped dictionary is the primary public contract.

Attribute-style access remains available because `ResultBase` exposes
diagnostics through `__getattr__`, but maintainers should treat grouped access
as the semantic reference:

```python
result.diagnostics["rmse"]
result.diagnostics["r_squared"]
result.diagnostics["aic"]
```

not as a design invitation to keep flattening the object indefinitely.

## Information-Criterion Conventions

The current maintained Gaussian-residual conventions are:

```text
AIC = n * ln(RSS / n) + 2k
BIC = n * ln(RSS / n) + k * ln(n)
```

where:

- `n = n_observations`
- `RSS = rss`
- `k = n_varying_parameters`

These are model-comparison diagnostics. They are not uncertainty estimates and
do not belong to the covariance family.

## Uncertainty Surfaces

The maintained matrix/vector uncertainty surfaces are:

- `result.covariance`
- `result.variance`
- `result.stderr`
- `result.correlation`
- `result.confidence_intervals`

These are not stored inside `result.diagnostics`.

That separation is intentional:

- diagnostics are grouped summary values
- uncertainty matrices and vectors are richer scientific objects
- keeping them out of `diagnostics` avoids turning one mapping into a mixed bag
  of scalars, arrays, and future parameter-state fragments

### Availability

These uncertainty surfaces are available only when the least-squares-backed
uncertainty path is mathematically meaningful.

They return `None` when the necessary prerequisites are missing, including
cases such as:

- dry fits
- `simplex`
- `basinhopping`
- missing Jacobian
- non-positive residual degrees of freedom

`FitResult` therefore distinguishes two kinds of scientific enrichment:

1. diagnostics that are broadly available from residual statistics
2. uncertainty surfaces that require stronger numerical conditions

## Configuration vs Scientific Result

`FitResult.parameters` continues to mean estimator run configuration, not the
fitted parameter solution.

Typical contents include values such as:

- `method`
- `max_iter`
- `max_fun_calls`
- `autobase`
- `amplitude_mode`

This distinction must remain explicit:

- configuration belongs in `result.parameters`
- scientific outputs belong in `result.outputs`
- scientific summaries belong in `result.diagnostics`
- uncertainty surfaces belong on dedicated `FitResult` properties

The campaign deliberately did **not** introduce a `solution` surface yet.

## Runtime Boundary

`FitResult` is currently a runtime grouping object, not a durable scientific
record with an independent lifecycle.

In particular:

- `Optimize.result` recreates a `FitResult` view from estimator state
- no fit-generation identifier is retained
- no persistence contract is established here
- no dedicated export API is attached to `FitResult`

The current contract should therefore be read as:

```text
stable grouped scientific access
within a runtime result-object boundary
```

not as a commitment to snapshot persistence semantics.

## Notebook and Display Implications

The scientific contract is now rich enough that notebook representation matters.

Future display work should preserve the same conceptual grouping:

- outputs
- diagnostics
- uncertainty

without collapsing everything into one flat textual dump.

The display layer should make the contract easier to inspect, not redefine it.

## Non-goals

This architecture note does not:

- redefine `FitResult.parameters`
- introduce `result.solution`
- introduce `EstimationResult`
- attach plotting methods to `FitResult`
- define persistence semantics for result objects
- define optimizer backend expansion policy

Those remain separate topics.

## Maintainer Guidance

When extending `FitResult`, prefer the following order of questions:

1. Is this a scientific output dataset?
   - put it in `outputs`
2. Is this a fit-quality or solver-summary scalar?
   - put it in `diagnostics`
3. Is this an uncertainty matrix/vector/surface?
   - expose it as a dedicated `FitResult` property
4. Is this actually raw solver state?
   - keep it on `Optimize` unless and until a scientific interpretation is
     defined

This preserves the architecture that the enrichment campaign clarified.

## Follow-up Boundaries

The next natural work is no longer “add one more scalar”.

The highest-value follow-up topics are:

- a dedicated `FitResult` documentation page for users
- improved notebook / HTML representation
- a fresh audit of the user-facing `Optimize` API

Any future `solution` surface should emerge from actual pressure, not from a
desire to pre-abstract the current contract.
