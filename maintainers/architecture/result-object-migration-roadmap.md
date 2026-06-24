# Result Object Migration Roadmap

**Status:** Campaign complete for core estimators

**Related RFC:** `result-object-contract-rfc.md`

**Campaign review context:** This roadmap preserves the tracked maintainer
summary for the completed campaign; detailed implementation history remains in
the local audit trail.

---

This roadmap is now a post-campaign maintainer summary. It preserves the
contract, migration outcome, architectural lessons, and explicit boundaries for
future work. Detailed PR-by-PR history remains in the audit trail.

## 1. Campaign Outcome

The Result Object Contract campaign is complete for core estimators.

Initial objectives preserved by the completed campaign:

- introduce a stable public Result contract for analysis and fit outputs
- preserve existing estimator APIs and behavior
- keep estimator storage as an internal implementation detail
- validate the contract across heterogeneous estimator implementations

Nine migrations were completed:

| PR | Estimator | Result type | Status |
|---|---|---|---|
| PR1 (#1208) | PCA | `AnalysisResult` | Merged |
| PR2 (#1209) | SVD | `AnalysisResult` | Merged |
| PR3 (#1211) | Optimize | `FitResult` | Merged |
| PR4 (#1213) | NMF | `AnalysisResult` | Merged |
| PR5 (#1215) | MCRALS | `AnalysisResult` | Merged |
| PR6 (#1217) | SIMPLISMA | `AnalysisResult` | Merged |
| PR7 (#1218) | EFA | `AnalysisResult` | Merged |
| PR8 (#1219) | FastICA | `AnalysisResult` | Merged |
| PR9 (#1220) | PLSRegression | `AnalysisResult` | Merged |

`ResultBase`, `AnalysisResult`, and `FitResult` are defined in
`src/spectrochempy/analysis/_base/_result.py` (98 lines total). All three
classes were used without subclass adaptation.

Campaign summary:

- 9 migrations completed
- 0 modifications to `ResultBase` since PR1
- 0 specialized `AnalysisResult` or `FitResult` subclasses introduced
- contract validated across multiple storage strategies

Validated storage strategies:

- sklearn object stored in `_outfit`
- tuple stored in `_outfit`
- private estimator attributes
- `_outfit = None`

## 2. Final contract

The following contract is now stable and documented for maintainers:

### `ResultBase`

| Member | Type | Required |
|---|---|---|
| `estimator` | `str` | Yes, via `__init__` |
| `parameters` | `dict` | Optional, defaults to `{}` |
| `outputs` | `dict` | Optional, defaults to `{}` |
| `diagnostics` | `dict` | Optional, defaults to `{}` |
| `__repr__` | `str` | Compact multi-line summary |

### `AnalysisResult`

- Empty subclass of `ResultBase`
- Used unchanged across decomposition and cross-decomposition migrations

### `FitResult`

- Empty subclass of `ResultBase`
- Used unchanged for fitting and optimization

### Common estimator pattern

```python
@property
def result(self):
    if not self._fitted:
        raise NotFittedError(...)
    return ResultType(
        estimator="Name",
        parameters={...},
        outputs={...},
        diagnostics={...},
    )
```

### Output and parameter conventions

Values in `outputs` and `diagnostics` may be:

- `NDDataset`
- raw `ndarray`
- simple scalar values such as `float`, `int`, `str`, `bool`

Values in `parameters` should remain JSON-compatible or Traitlets-native
configuration values.

## 3. Key Architectural Findings

### Contract vs Storage

Result objects are public API.
Internal estimator storage remains an implementation detail.

The campaign validated the same Result contract against multiple storage
patterns, including:

- sklearn object in `_outfit`
- tuple in `_outfit`
- private attributes
- `_outfit = None`

Public semantics now live in the `result` contract, not in the storage shape.

### No `ResultBase` evolution required

Across all migrations:

- `ResultBase` remained unchanged
- `AnalysisResult` remained unchanged
- `FitResult` remained unchanged

This remained true even for the most structurally different estimators and for
the first `_outfit`-free migration.

### `_fit_meta` pattern

The `_fit_meta` convention emerged as the lightweight solution for
estimator-local diagnostics that `_fit()` would otherwise discard.

The pattern is:

- optional
- estimator-local
- used only when diagnostics would otherwise be discarded

This pattern was sufficient without any base-class change.

## 4. Migration summary

Each migration validated the same contract against a distinct architectural
scenario:

| Estimator | Storage pattern | What it validated |
|---|---|---|
| PCA | sklearn object in `_outfit` | first prototype of the property pattern |
| SVD | tuple in `_outfit` | raw ndarray outputs plus NDDataset diagnostics |
| Optimize | tuple in `_outfit` | first `FitResult` use and `_fit_meta` capture |
| NMF | sklearn object in `_outfit` | repeatability on another sklearn-backed estimator |
| MCRALS | large tuple in `_outfit` | contract stability under the most complex internal layout |
| SIMPLISMA | tuple in `_outfit` | iterative diagnostics captured via `_fit_meta` |
| EFA | tuple in `_outfit` | sparse diagnostics and dynamic component computation |
| FastICA | sklearn object in `_outfit` | callable parameter handling and whitening edge cases |
| PLSRegression | private attributes on `self` | first `_outfit`-free migration |

## 5. Deferred Optional Infrastructure

The following topics do not belong to the completed Result campaign. They are
optional or deferred concerns rather than unfinished Result architecture:

- structured Result persistence
- typed Project membership
- provenance enrichment
- HTML / display integration
- caching

These are not remaining per-estimator Result migrations. Dataset persistence is
already established; structured Result persistence and typed Project membership
remain deferred.

## 6. Known limits and optional follow-up

Known deliberate limits of the completed campaign:

- no caching; `result` creates a fresh object on each access
- no serialization support in `ResultBase`
- no Project integration
- no provenance enrichment beyond current estimator-facing data
- no HTML or display integration beyond `__repr__`

Optional follow-up candidates that do not change campaign completion status:

| Estimator | Classification | Rationale |
|---|---|---|
| Baseline | Optional | Processor-specific, outside the main `analysis/` migration set |
| LSTSQ | Optional | Thin wrapper, low architectural risk |
| NNLS | Optional | Same profile as LSTSQ |

## 7. Audit Trail

Detailed implementation history remains in the local audit trail. The tracked
maintainer references for this completed campaign are this roadmap and
[`result-object-contract-rfc.md`](result-object-contract-rfc.md).

## 8. 0.11 Result Alignment

The Result Object campaign is complete for core estimators. The remaining work
before `.result` is described as the canonical grouped-output API is alignment
and completion work around the implemented runtime Result contract.

Before `.result` is described as the canonical grouped-output API for 0.11, the
following alignment tasks should be completed:

- publish `ResultBase`, `AnalysisResult`, and `FitResult` through a documented
  public import path rather than only the private `_base._result` module;
- document the current live-view behavior and decide whether it should remain
  live, become cached, or become a fit-time snapshot; no direction is currently
  preferred;
- define the minimum scientifically complete `FitResult` payload;
- align the remaining maintained stateful core candidates (`Baseline`,
  `LSTSQ`, and `NNLS`) or document them as explicit exceptions;
- keep the completed IRIS plugin alignment covered by compatibility tests;
- complete TENSOR/CP plugin alignment;
- update analysis and official-plugin documentation to teach `.result`;
- coordinate compatible IRIS and TENSOR releases before the core 0.11 release,
  because both plugins currently declare an upper core bound below 0.11.

The recommended 0.11 scope is completion of the canonical **runtime Result
contract** across core and official plugins. Structured Result persistence and
typed Project membership remain deferred optional directions. Dataset export
and dataset persistence are sufficient for 0.11.

### Recommended sequence

1. Document the selected live, cached, or fit-time snapshot lifecycle semantics
   and publish the public Result type path.
2. Complete Baseline, LSTSQ, and NNLS alignment.
3. Complete TENSOR/CP plugin alignment and release compatible official plugins;
   IRIS Result alignment is implemented.
4. Update user documentation and remove only confirmed deprecated aliases.
5. Keep Results runtime-only in 0.11, using dataset export and established
   dataset persistence when saved outputs are needed.
6. Keep structured Result persistence and typed Project membership deferred
   unless future use cases justify them.
