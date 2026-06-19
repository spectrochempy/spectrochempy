# Result Object Migration Roadmap

**Status:** Campaign complete for core estimators

**Related RFC:** `result-object-contract-rfc.md`

**Campaign review:** `audit/~result-object-campaign-closure-review.md`

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

## 5. Deferred Infrastructure Work

The following topics do not belong to the completed Result campaign.
They belong to separate architecture efforts:

- serialization
- Project integration
- provenance enrichment
- HTML / display integration
- caching

These are cross-cutting infrastructure concerns, not remaining per-estimator
Result migrations.

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

Detailed implementation history remains in the audit files:

- `audit/~result-object-campaign-closure-review.md`
- `audit/~result-object-contract-pr1-audit.md`
- `audit/~result-object-contract-pr2-audit.md`
- `audit/~result-object-contract-pr3-audit.md`
- `audit/~result-object-contract-pr4-audit.md`
- `audit/~result-object-contract-pr5-audit.md`
- `audit/~result-object-contract-pr6-audit.md`
- `audit/~result-object-contract-pr7-audit.md`
- `audit/~result-object-contract-pr8-audit.md`
- `audit/~result-object-contract-pr9-audit.md`
