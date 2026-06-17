# RFC: Analysis and Fit Result Architecture

**Status:** Draft

**Audit reference:**
[`audit/~analysis-fit-result-architecture-audit.md`](../audit/~analysis-fit-result-architecture-audit.md)

**PR9 output semantics reference:**
[`audit/~pr9-analysis-output-semantics-report.md`](../audit/~pr9-analysis-output-semantics-report.md)

---

## Purpose

This RFC answers three questions:

1. What is the architectural nature of analysis and fit results in
   SpectroChemPy?
2. Who owns result state?
3. Where does `NDDataset` fit, and where is it insufficient?

This document is conceptual.

It does not propose API changes, implementation work, or a migration plan.

It formalizes vocabulary and architectural findings so that future maintainer
discussions can proceed from a shared understanding.

---

## Core Concepts

### 1. Result Surface

A **result surface** is the public way a user accesses an output of an analysis
or fit method.

Examples from the current codebase:

| Category | Example surfaces |
|----------|-----------------|
| Scores / latent coordinates | `PCA.transform()`, `NMF.transform()`, `MCRALS.transform()` |
| Loadings / components | `PCA.components`, `NMF.components`, `MCRALS.components` |
| Predictions | `PLSRegression.predict()`, `Optimize.predict()` |
| Fitted model curves | `Optimize.components`, `Optimize.inverse_transform()` |
| Diagnostics | `PCA.explained_variance`, `SVD.sv`, `EFA.f_ev` |
| Fit parameters | `Optimize.fp` |
| Factor arrays | `SVD.U`, `SVD.VT`, `SVD.s` |
| Reconstructions | `PCA.inverse_transform()`, `MCRALS.inverse_transform()` |

A single analysis or fit method typically produces **multiple result surfaces**.

---

### 2. Result Ownership

Result ownership answers: *where does the post-fit state live?*

Current reality in SpectroChemPy:

- **Primary owner:** the fitted analysis or fit object itself (`PCA`, `Optimize`,
  etc.)
- **Secondary carrier:** `NDDataset` — used for individual result arrays surfaced
  through accessors
- **Fallback carrier:** positional tuples (`_outfit`) and raw ndarrays for
  richer multi-part results
- **Exception carrier:** plain Python tuples returned from utility-style fits
  (`ActionMassKinetics.fit_to_concentrations()`)

The dominant pattern today is:

```
fit returns self
results live on the fitted object
individual surfaces are exposed via accessors or properties
dataset wrapping happens lazily per accessor
```

---

### 3. Analysis Result

An **analysis result** is an output produced by a decomposition, projection, or
latent-variable method.

Following the PR9 analysis output semantics characterization, analysis results
fall into three semantic families:

| Family | Description | Examples |
|--------|-------------|---------|
| **Latent output** | Derived analysis objects in component/latent space | `scores`, `loadings`, `components`, `transform()` |
| **Diagnostic output** | Model-quality or factor-summary objects | `explained_variance`, `singular_values`, `ev_ratio` |
| **Reconstructed output** | Source-space objects built from latent representation | `inverse_transform()`, `predict()` |

Current analysis classes:
- `PCA`, `NMF`, `PLSRegression` — sklearn-estimator-style, result state on the
  fitted object
- `SVD` — decomposition subclass but does **not** implement the generic latent
  transform contract; behaves more like a factorization + diagnostics object
- `EFA` — exposes a small internal tuple as concentration profiles and evolving
  diagnostic vectors
- `MCRALS` — richest current result surface: multiple concentration/spectral
  profiles, constrained versions, iteration history lists, external function
  outputs

---

### 4. Fit Result

A **fit result** is an output produced by a model-fitting or optimization
workflow.

Fit results put stronger architectural pressure than analysis results because a
scientifically complete fit is inherently multi-part:

- fitted parameters
- fitted model curve(s)
- individual model components
- residuals
- covariance / uncertainty
- solver diagnostics (success, message, iterations)
- goodness-of-fit summaries

Current fit-related objects:

- `Optimize` — the clearest fit-result architecture in core. Subclasses
  `DecompositionAnalysis` even though its semantics are fitting/modeling.
  Internal state includes `fp` (`FitParameters`), `script`, component datasets,
  and a positional `_outfit` tuple. The raw optimizer result is **not**
  preserved as a public surface.
- `LinearRegressionAnalysis` / `LSTSQ` / `NNLS` — sparse result surfaces
  (coefficients, intercept, predictions). No explicit residual or uncertainty
  object.
- `ActionMassKinetics.fit_to_concentrations()` — utility-style method that
  returns a plain tuple mixing fitted arrays, mapping metadata, and a raw
  `scipy.optimize.OptimizeResult`.

---

### 5. Parameter Object

`FitParameters` is a custom `UserDict` subclass that stores:

- parameter values
- lower and upper bounds
- fixed flags
- reference links
- common-block membership
- model mapping
- experiment-variable structure

Classification:

- **Parameter state:** yes — it defines fit configuration.
- **Fit-configuration state:** yes — it carries model structure.
- **Fitted-parameter result:** partially — optimization mutates it in place, but
  it is not paired with covariance, uncertainty, or residual surfaces.

`FitParameters` is a parameter container. It is not currently a full fit result
object.

---

### 6. Diagnostic Object

A **diagnostic** is an output that summarizes model quality, factor importance,
or solver behavior.

Current examples:

- explained variance and ratio (`PCA.ev`, `PCA.ev_ratio`)
- singular values (`SVD.sv`)
- forward/backward eigenvalues (`EFA.f_ev`, `EFA.b_ev`)
- score / R² (`PLSRegression.score()`)
- Optimize solver result (`fopt`) — computed but **not surfaced** as public
  result state
- `scipy.optimize.OptimizeResult` — returned by
  `ActionMassKinetics.fit_to_concentrations()` but **discarded** by
  `Optimize.fit()`

Diagnostic representation is uneven across the codebase:

- some diagnostics are `NDDataset` objects with coordinates
- some are plain scalars
- some are raw arrays
- some are computed but not exposed

---

### 7. Reconstructed / Predicted Source-Space Object

A **reconstructed source-space output** is a result that returns to the geometry
of the original input data:

- `inverse_transform()` — reconstructs data from latent representation
- `predict()` — generates predicted target values
- `Optimize.inverse_transform()` — returns the fitted total model curve

These are generally `NDDataset`-compatible because they share the source data's
axis structure. Their provenance should record which model or fit method
produced them, but current history rewriting does not always preserve this
information in a structured way.

---

## Current State

SpectroChemPy's current result representation is **functionally workable but
semantically fragmented**.

The dominant pattern is:

```
object-owned state with dataset accessors,
augmented by ad hoc tuples and raw arrays where the result is richer than one
dataset
```

Specific analysis and fit objects:

| Object | Result ownership | Internal carrier | Public surface richness |
|--------|-----------------|-----------------|----------------------|
| `PCA` | sklearn estimator | `self._pca` attributes | ~7 accessors |
| `SVD` | decomposition object | `_outfit` tuple | ~6 accessors + raw arrays |
| `NMF` | sklearn estimator | `self._nmf` attributes | ~3 accessors |
| `EFA` | decomposition object | `_outfit` tuple | ~4 accessors |
| `MCRALS` | decomposition object | large positional `_outfit` tuple | ~10 accessors + raw arrays |
| `PLSRegression` | sklearn estimator | `self._plsregression` + custom arrays | ~12 accessors |
| `Optimize` | decomposition subclass (fit semantics) | `_outfit` tuple + `fp` + `script` | ~5 accessors |
| `LinearRegressionAnalysis` | sklearn-style estimator | internal arrays | ~3 accessors |
| `ActionMassKinetics` | plain object | raw tuple | returned tuple |

---

## Architectural Findings

### 1. Result ownership is object-centric, not dataset-centric

Since `modeldata` removal, result state lives primarily on analysis/fit objects
rather than on `NDDataset`. This is a genuine architectural improvement in
clarity.

### 2. `modeldata` was addressing a real problem, but with the wrong owner

The historical need — grouped multi-part model outputs belonging together — was
real. Attaching hidden model state to `NDDataset` was the wrong architectural
placement. The removal corrected the ownership, but the underlying scientific
need for grouped result representation remains.

### 3. `NDDataset` is a good carrier for single scientific result arrays

For one latent matrix, one component matrix, one reconstructed dataset, or one
diagnostic vector, `NDDataset` is natural and well-suited.

### 4. `NDDataset` is insufficient as the only abstraction for grouped results

When the scientific output is several aligned datasets plus parameter state plus
diagnostics plus solver metadata, `NDDataset` alone does not provide a natural
grouping abstraction.

### 5. `_outfit` is useful internally but weak as architecture

The generic `_outfit` tuple provides a common implementation hook, but it is:

- positional
- class-specific
- semantically opaque
- not self-describing

For simple cases it is manageable. For rich result surfaces (`MCRALS`,
`Optimize`) it becomes difficult to read and fragile to extend.

### 6. Fit diagnostics are underrepresented on the public surface

In current fit-like code:

- residuals are often implicit (computed manually in tests)
- solver results may be discarded
- uncertainty is mostly absent
- covariance is not surfaced
- goodness-of-fit summaries are sparse

### 7. SpectroChemPy currently mixes several result idioms

| Idiom | Where |
|-------|-------|
| Fitted object + accessors | `PCA`, `NMF`, `PLSRegression`, `LinearRegressionAnalysis` |
| Fitted object + raw arrays | `SVD.U`, `SVD.VT`, `Optimize.modeldata` |
| Fitted object + positional tuple | `MCRALS`, `EFA`, `SVD._outfit`, `Optimize._outfit` |
| Returned plain tuple | `ActionMassKinetics.fit_to_concentrations()` |
| Standalone `NDDataset` per accessor | All `transform()` / `components` / `inverse_transform()` surfaces |

This is the strongest evidence of semantic fragmentation in result
representation.

---

## Ecosystem Comparison

This section records how other scientific Python projects handle related
concerns. It is provided for context only — not as a recommendation.

### NumPy

- Functional numerical API.
- Factorization-style operations return arrays or tuples of arrays.
- No persistent fitted-object architecture.

### SciPy

- Solver functions often return explicit result objects
  (`scipy.optimize.OptimizeResult`).
- Structured result objects carry success flags, messages, and diagnostic
  fields alongside numerical outputs.

### scikit-learn

- One predominant pattern: estimator-owned state.
- `fit()` returns `self`.
- Result surfaces come from `predict()`, `transform()`, `score()`, and learned
  attributes.
- This is the strongest influence on current SpectroChemPy analysis
  architecture.

### xarray

- `DataArray`: one labeled array plus coordinates and attributes.
- `Dataset`: dict-like container of aligned labeled arrays.
- SpectroChemPy has no equivalent of a grouped multi-array result container for
  analysis outputs.

### lmfit

- Explicit `ModelResult` class grouping:
  - fitted parameters
  - best-fit curve
  - initial fit
  - residual
  - covariance
  - uncertainty
  - component evaluation
  - fit reports / summaries
- SpectroChemPy `Optimize` solves a closely related problem but spreads
  equivalent concerns across several ungrouped surfaces.

---

## Candidate Directions

This section records architectural options without choosing one. Listed in no
preferred order.

### Candidate A: Keep estimator-owned pattern, strengthen conventions

- Fitted objects remain the primary result owner.
- No new container class required.
- Consistency would come from clearer conventions about what each accessor
  should expose and how result surfaces should be named.

**What it would clarify:**
- naming and access conventions across analysis and fit classes

**What it would not solve:**
- grouped multi-part result representation
- semantic opacity of `_outfit`

**Likely risk / complexity:** Low. Mostly documentation and convention.

---

### Candidate B: Introduce a grouped result collection abstraction

- A new collection-style object for multi-array analysis outputs.
- `NDDataset` would remain the carrier for individual arrays.
- The collection would own the group and expose named, labeled members.

**What it would clarify:**
- grouping of related outputs (scores + loadings + diagnostics)
- explicit membership instead of positional `_outfit`

**What it would not solve:**
- fit-specific concerns (covariance, uncertainty, solver metadata)

**Likely risk / complexity:** Medium-high. New type requires design, tests,
serialization, and ecosystem integration.

---

### Candidate C: Introduce a dedicated fit-result object

- A `FitResult` abstraction for optimization / modeling workflows.
- Would group: parameters, model curves, residuals, diagnostics, uncertainty,
  solver metadata.
- Closest ecosystem analogue: `lmfit.ModelResult`.

**What it would clarify:**
- fit results as first-class objects
- separation of fit outputs from decomposition outputs
- natural home for residuals, covariance, uncertainty

**What it would not solve:**
- decomposition/analysis result grouping (unless combined with B)

**Likely risk / complexity:** Medium. Domain is narrower than a general
collection, so scope is clearer.

---

### Candidate D: Keep datasets for scientific arrays, keep diagnostics separate

- Scientific result arrays stay as `NDDataset`.
- Solver and parameter state remain non-dataset objects (plain dicts, custom
  parameter objects, or SciPy-style result objects).
- No new grouped abstraction.

**What it would clarify:**
- clear boundary between scientific data and optimization metadata

**What it would not solve:**
- fragmented result surface landscape
- implicit diagnostics and discarded solver state

**Likely risk / complexity:** Low. Preserves current trajectory.

---

## Open Questions

1. Should the fitted object itself remain the primary result object, or should
   result state be transferred to a dedicated result container after `fit()`
   completes?

2. Should rich fit workflows (`Optimize`) expose a dedicated `FitResult`
   object?

3. Should MCRALS / PLSRegression-like multi-part outputs be grouped in a named
   collection?

4. Should solver diagnostics (`success`, `message`, `covariance`,
   `uncertainty`) be considered public result state or internal
   implementation detail?

5. Are residuals and uncertainties first-class scientific results in
   SpectroChemPy?

6. Should analysis and fit results share one result architecture, or is the
   pressure different enough to warrant separate approaches?

7. Should `_outfit` remain an internal implementation detail only, or should it
   be formalized?

8. Should future APIs distinguish between:
   - latent results
   - diagnostic results
   - reconstructed results
   - fit results?

9. Is `Optimize` better interpreted as a decomposition-like object, a
   model-fitting object, or a hybrid historical compromise?

10. Is `FitParameters` only a parameter container, or is it already acting as a
    partial fit-result abstraction?

---

## Maintainer Assessment

No immediate implementation work is recommended.

The audit justifies a shared vocabulary and a future design discussion, but does
not identify an urgent maintenance problem or user-facing limitation.

The strongest near-term conclusion is:

- Result state should remain analysis/fit-object-owned, not dataset-owned. The
  `modeldata` removal was architecturally correct.
- The current fragmentation is real but tolerable. It does not block current
  development.

Future work should focus on:

1. clarifying conventions for result surfaces before introducing new classes
2. deciding whether fit workflows justify a dedicated result object
3. determining whether MCRALS-class multi-output objects need a collection
   abstraction

Any of these decisions should be deferred until a concrete maintenance problem
or user-facing limitation motivates the work.

**Core question to revisit when the time comes:**

> The need that modeldata addressed was real.
> What should own grouped multi-part scientific results
> when the result is richer than one dataset?
