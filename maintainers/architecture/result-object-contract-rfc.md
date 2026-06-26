[Maintainers](../../README.md) · [Architecture](../INDEX.md)

# RFC: Result Object Contract

**Status:** Implemented

**Campaign closed:** 2026-06-26

**Related RFC:** `maintainers/rfcs/analysis-fit-result-architecture.md`

## Motivation

SpectroChemPy 0.10.0 removed several historical state containers, including
`NDDataset.modeldata`, ROI, Project scripts, and `Project._others`. `Project`
is now a typed recursive container for `NDDataset` and `Project`.

The next architectural pressure point is analysis and fit output state.
Analysis and fitting methods routinely produce more than one scientific object:
scores, loadings, components, fitted curves, residuals, parameters, diagnostic
vectors, solver metadata, and reconstructed data. Today these outputs are
available, but they are not represented through a common result contract.

This RFC defines the contract for result objects. The contract has now been
implemented across the completed Result Object campaign; this document remains
the architectural reference for that contract rather than a change log of the
campaign.

## Non-goals

This RFC:

- does not redesign estimators
- does not redesign `NDDataset`
- does not redesign `Project`
- does not redesign metadata or provenance systems
- does not redesign serialization infrastructure
- does not redesign `DisplayItem` / `DisplaySection`

This RFC only defines a stable ownership, representation, and lifecycle
contract for analysis and fit outputs.

The contract is now implemented and the Result Alignment campaign is complete
for the maintained estimator-style scope.

## Current Runtime Boundary

The implemented Result classes are currently runtime grouping objects, not
durable records.

In particular:

- estimator `.result` properties construct a new Result on every access;
- parameters and outputs are read from the estimator's current state;
- no fit-generation identifier or parameter snapshot is retained;
- the returned dictionaries are mutable;
- the classes have no stable public import path outside the private
  `analysis._base._result` module;
- there is no schema version, persistence hook, Project ownership contract, or
  semantic HTML display.

The current implementation is therefore a live view suitable for grouped
runtime output discovery. No architectural decision has been made to replace
it.

Within that runtime boundary, `.result` is now the canonical grouped-output
API for estimator-style objects that expose multiple scientific outputs and/or
diagnostics.

Possible future lifecycle directions are:

1. retain the current live view;
2. use a cached view with invalidation on re-fit;
3. create a fit-time snapshot.

The choice is an open maintainer decision. It becomes necessary only if a
feature such as structured Result persistence or automatic stale-result
tracking requires stronger run identity.

## Current state

The current analysis base class stores post-fit state in
`AnalysisConfigurable._outfit`. `fit()` preprocesses `X` and optional `Y`, calls
the subclass `_fit()` implementation, assigns the returned object to `_outfit`,
sets `_fitted = True`, and returns `self`.

Observed result patterns include:

| Class | Current result carrier | Public result surfaces |
|-------|------------------------|------------------------|
| `PCA` | sklearn PCA object plus copied attributes; `_outfit` is the sklearn fitted object | `scores`, `loadings`, `components`, explained variance datasets, `inverse_transform()` |
| `SVD` | `_outfit` tuple from `np.linalg.svd()` | singular values as `NDDataset`; `U`, `VT`, `s` as raw arrays |
| `MCRALS` | long positional `_outfit` tuple | final `C` and `St`, constrained profiles, iteration lists, external-function outputs |
| `Optimize` | positional `_outfit` tuple plus mutable `FitParameters` and script state | fitted components, total fitted model, transform/inverse transform, fit parameters |

Several result arrays are converted to `NDDataset` lazily through
`_wrap_ndarray_output_to_nddataset()`. This keeps public surfaces convenient,
but it means semantic assembly currently lives partly in properties and
decorators rather than in a named result object.

There is no common serialization contract for analysis or fit result state.
There is no typed Project membership for result objects. Text and HTML
representations are class-specific or absent.

## Problems with `_outfit`

`_outfit` remains useful as a private implementation hook, but it is too weak as
the long-term architecture for result state.

The main issues are:

- It is positional. Meaning is carried by tuple order, not by field names.
- It is class-specific. `PCA`, `SVD`, `MCRALS`, and `Optimize` store different
  kinds of objects under the same attribute name.
- It is not self-describing. A maintainer must read each `_fit()` and each
  property to understand the result layout.
- It encourages fragile accessors. Several properties depend on indexes such as
  `_outfit[2]`, `_outfit[7]`, or `_outfit[9]`.
- It has no persistence boundary. There is no explicit decision about what
  belongs to serialized analysis state.
- It risks recreating the old `modeldata` problem if hidden result state becomes
  attached to datasets instead of remaining owned by analysis or fit objects.

This RFC does not require immediate removal of `_outfit`. `_outfit` may
continue to exist as an internal implementation detail during migration. The
goal is to move public semantics progressively from positional internal storage
to explicit result objects.

## Proposed concepts

### Estimator or model object

The estimator is the configurable object the user instantiates, configures, and
fits. Examples: `PCA`, `SVD`, `MCRALS`, `Optimize`.

It owns method parameters and runtime configuration. Before fitting, it is not a
result.

### Fitted estimator

A fitted estimator is the same estimator after successful `fit()`. Current
public behavior should remain: `fit()` returns `self`, and existing properties
continue to work during migration.

The fitted estimator may keep a reference to the last result object, but it
should not be the only structured representation of the result.

### Result object

A result object is the named, structured output record produced by a successful
fit or analysis operation.

It groups related output datasets and diagnostics from one run. It records
provenance, input summary, estimator identity, and result fields by name.

It is not an estimator. It should not expose mutating fit methods. It should not
silently recompute itself.

### Output `NDDataset`

An output `NDDataset` is one scientific array produced by the operation:
scores, loadings, components, reconstructed data, fitted curve, residuals, or a
diagnostic vector.

`NDDataset` remains the correct carrier for labeled scientific arrays. The
result object groups these datasets; it does not replace them.

### Diagnostics

Diagnostics are non-primary scientific outputs that summarize quality,
decomposition structure, solver state, or convergence.

Examples include explained variance, singular values, residual norms, solver
success flags, convergence messages, number of iterations, covariance, and
uncertainty estimates.

## Minimal hierarchy

The minimal future hierarchy should be:

```text
ResultBase
|-- AnalysisResult
`-- FitResult
```

`ResultBase` defines shared identity, provenance, display, named output access,
and optional serialization hooks.

`AnalysisResult` specializes the contract for decomposition and projection
outputs. Typical fields include `scores`, `loadings`, `components`,
`diagnostics`, and `reconstructed`.

`FitResult` specializes the contract for model fitting and optimization.
Typical fields include `parameters`, `fitted`, `components`, `residuals`,
`diagnostics`, and optional uncertainty state.

This hierarchy is intentionally small. It should not introduce separate classes
for every method in PR1.

The implemented campaign validated that this minimal hierarchy was sufficient
across the maintained estimator-style scope. No specialized Result subclasses
were required in core or aligned official plugins.

## Ownership model

The result object owns its named outputs.

Architectural responsibility remains separated:

- `NDDataset` owns scientific data
- estimators own algorithms and fitting procedures
- result objects own analysis and fit outputs
- `Project` owns organization and grouping

The original RFC proposed the following ownership rules. They remain relevant
to a possible snapshot-style Result, but the current live-view implementation
does not adopt them as lifecycle requirements:

- Under a snapshot model, output datasets could be independent objects rather
  than views that silently track future estimator or input mutation.
- The result may keep a lightweight input summary, but should not copy the full
  input dataset by default.
- The result may optionally keep a weak or private reference to the fitted
  estimator for developer inspection, but serialized result state must not
  depend on that live object reference.
- The fitted estimator may expose `result` or an internal `_result` during
  migration, while legacy properties continue to forward to the same underlying
  outputs.
- A result should never be stored inside `NDDataset` metadata or hidden array
  attributes. That would recreate the removed `modeldata` ownership problem.

If structured provenance or structured Result persistence is later pursued, an
input summary could identify what was fitted without duplicating the input:

- input class
- shape
- dimension names
- coordinate summaries where cheap and stable
- dataset name, title, units, and relevant metadata keys
- mask summary, not full mask copy unless explicitly required later

## Provenance model

Human-readable history and structured provenance should be distinct.

If structured provenance is added, candidate fields include:

- estimator class name, for example `PCA` or `Optimize`
- estimator display name if present
- selected public parameters used for the run
- input summary
- SpectroChemPy version
- optional timestamp
- optional backend/library summary, for example sklearn estimator class
- optional warnings or convergence notes

Human-readable history should remain suitable for display in output
`NDDataset.history`. It should summarize what was done, not serve as the only
machine-readable record.

The initial contract should avoid over-promising complete reproducibility.
Structured provenance identifies and explains a result; it is not yet a full
workflow replay system.

## Historical serialization boundary

The original RFC proposed the following serialization boundary. It was not
implemented, and structured Result persistence remains undecided and deferred.

Candidate serializable fields were:

- result type name
- schema version
- estimator name
- estimator parameters that are JSON-compatible or already supported by
  existing metadata serialization
- input summary
- named output datasets that already support native `NDDataset` serialization
- simple diagnostics: scalars, strings, lists, arrays, and `NDDataset`
  diagnostics

Candidate exclusions were:

- full estimator object serialization
- live sklearn object serialization
- callable serialization for advanced `MCRALS` hooks
- full `FitParameters` round-trip if it requires a dedicated schema
- covariance and uncertainty schemas where no stable public surface exists yet
- Project save/load round-trip for result objects

Possible internal representation:

```text
{
  "type": "AnalysisResult",
  "schema_version": 1,
  "estimator": {...},
  "input": {...},
  "outputs": {"scores": <NDDataset>, "loadings": <NDDataset>},
  "diagnostics": {"explained_variance": <NDDataset>},
  "provenance": {...}
}
```

This representation is preserved as design history, not as a required public
API or a commitment to structured Result persistence.

## Historical immutability proposal

The original RFC proposed logically immutable Result records. The implemented
runtime Result contract instead provides a live view assembled on access.
Live-view, cached-view, and fit-time snapshot semantics remain open maintainer
alternatives.

If a snapshot model is later selected, the original recommendations were:

- Result fields are assigned at construction time.
- Users should treat result outputs as read-only.
- Whether outputs are implemented using copies, read-only views, or shared
  objects remains an implementation detail.
- Performance and memory considerations may influence the chosen strategy.
- Structured provenance should not be mutated by normal user operations.
- Recalculation belongs to the estimator, not the result object.

These recommendations explain the original snapshot-oriented design direction;
they do not select that direction for future work.

## Display contract

Every result object should have both a compact text representation and an HTML
representation.

Minimum text representation:

- result type
- estimator name
- main output names and shapes
- compact diagnostics summary
- fitted input summary

Minimum HTML representation:

- Summary section
- Outputs section
- Diagnostics section
- Provenance section

The HTML implementation should align with the existing `DisplayItem` and
`DisplaySection` model used by Project and other display work. The result
object should expose semantic display sections first; renderers can turn those
sections into terminal or HTML output.

Large arrays should not be fully printed in the result representation. They
should be discoverable by name and inspected through the underlying
`NDDataset`.

## Project boundary

The original RFC deferred Project integration while the Result contract was
being established.

Current `Project` intentionally accepts only `NDDataset` and nested `Project`
instances. A result object should not be added to Project in PR1.

Historical integration options considered:

- allow `ResultBase` as a third typed Project member
- store result outputs as a nested Project with a result manifest
- keep result serialization separate from Project until save/load round-trip is
  designed

The key constraint is that Project must remain typed and explicit. Reintroducing
a generic `_others` bucket would conflict with the 0.10.0 cleanup.

The current maintained decision is stricter: Result objects remain
runtime-only and are not Project members. `Project` remains intentionally
restricted to `NDDataset` and nested `Project`.

When saved outputs are needed, the supported direction is dataset export to
named `NDDataset` objects or a nested dataset-only `Project`, using established
dataset persistence. Structured Result persistence and typed Project membership
are separate deferred possibilities, not required future steps.

Not every fitted or stateful object requires a Result. The completed campaign
established `.result` as the canonical grouped-output API only where it adds
meaningful grouped scientific value. Objects with sparse or processor-oriented
surfaces may remain without `.result`.

## Migration plan

### PR1: Minimal result contract and PCA prototype

Scope:

- introduce `ResultBase`
- introduce `AnalysisResult`
- introduce `FitResult` as an empty or minimal subclass for future fit work
- create a PCA-only prototype result object
- keep `PCA.fit()` returning `self`
- keep existing PCA properties and methods working
- add tests that legacy PCA accessors and the new result fields agree
- add display smoke tests for text and HTML representation
- do not serialize through Project

The PCA prototype should be deliberately modest. It should prove ownership,
named outputs, provenance, and display without migrating the whole analysis
stack.

### PR2: Second analysis migration

Migrate one method with different pressure from PCA.

Good candidates:

- `SVD`, because it currently exposes both raw factor arrays and diagnostic
  datasets
- `NMF`, because it is closer to PCA but has a different sklearn backend and
  output vocabulary

The goal is to test whether `AnalysisResult` handles more than one analysis
shape before touching fit code.

### PR3: Fit or Optimize prototype

Introduce a minimal `FitResult` prototype around `Optimize`.

Likely first fields:

- fitted total model
- components
- parameters
- residuals if already cheap and unambiguous
- simple diagnostics available from current code

This PR should not attempt full covariance, uncertainty, or optimizer replay
unless those already have stable internal surfaces.

### Historical later-work candidates

The original RFC listed these candidates. They are preserved as historical
context, not as commitments:

- serialization round-trip for result objects
- Project integration
- richer fit diagnostics
- result comparison helpers
- migration of `MCRALS` iteration lists and external-function outputs
- optional support for storing result collections in projects

## Open questions

1. Should `fit()` eventually return the result object, or should SpectroChemPy
   preserve sklearn-style `fit()` returning `self` indefinitely?
   Note: this RFC is built around current SpectroChemPy behavior. `fit()`
   continues to return `self` throughout the migration described here. Any
   future proposal to change that behavior would require a dedicated RFC.

2. Should the public surface be `estimator.result`, `estimator.results`, or a
   method such as `estimator.get_result()`?

3. Should result dataset accessors return copies always, or should some
   high-cost arrays use read-only views?

4. How much of `FitParameters` belongs in a serialized `FitResult` PR1 schema?

5. Should result objects support mapping-style access, attribute access, or
   both?

6. Should `SVD.U` and `SVD.VT` become `NDDataset` outputs during migration, or
   remain raw arrays with diagnostic wrappers?

7. Should result timestamps be enabled by default, given that deterministic
   tests and serialized fixtures may prefer stable output?

## Proposed first implementation PR

The first implementation PR should be titled along the lines of:

`MAINT: introduce analysis result contract prototype`

Recommended scope:

- add `ResultBase`, `AnalysisResult`, and `FitResult`
- implement a PCA-only `AnalysisResult`
- expose it from the fitted PCA object without changing `fit()` return value
- keep every current PCA public accessor intact
- add focused tests for ownership, named outputs, provenance, and display
- leave serialization and Project integration out of scope

This keeps the migration small enough to review while creating the contract
needed for later SVD/NMF and Optimize work.
