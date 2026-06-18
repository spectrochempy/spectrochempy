# Result Object Migration Roadmap

**Status:** Active roadmap note

**Related RFC:** `result-object-contract-rfc.md`

**Audit trail:** `audit/~result-object-migration-roadmap-notes.md`

---

## 1. Validated state

Three prototypes have been implemented and merged:

| PR | Estimator | Result type | Commit |
|---|---|---|---|
| PR1 (#1208) | PCA | `AnalysisResult` | `a9157d1aa7` |
| PR2 (#1209) | SVD | `AnalysisResult` | `9aa59cf088` |
| PR3 | Optimize | `FitResult` | `6872bbed56` |

`ResultBase`, `AnalysisResult`, and `FitResult` are defined in
`src/spectrochempy/analysis/_base/_result.py` (98 lines total). All three
classes were used without subclass adaptation — the base classes sufficed.

Infrastructure validated:

- `ResultBase.__init__` accepts `estimator`, `parameters`, `outputs`,
  `diagnostics` as named dict-valued parameters.
- `ResultBase.__repr__` displays estimator name, parameters, output names
  with shapes, and diagnostic names with shapes.
- No subclass of `AnalysisResult` or `FitResult` was needed for PCA, SVD, or
  Optimize.

Common pattern across all three prototypes:

```python
@property
def result(self):
    if not self._fitted:
        raise NotFittedError(...)
    return SomeResult(
        estimator="Name",
        parameters={...},
        outputs={...},
        diagnostics={...},
    )
```

Cache approach (deliberately deferred):

- All three create a new result object on every access.
- `pca.result is not pca.result` — documented as intentional for PR phase.
- See Open Questions below.

---

## 2. Stabilised contract

The following contract is now stable (changing it would break three
estimators):

### `ResultBase`

| Member | Type | Required |
|---|---|---|
| `estimator` | `str` | Yes, via `__init__` |
| `parameters` | `dict` | Optional, defaults to `{}` |
| `outputs` | `dict` | Optional, defaults to `{}` |
| `diagnostics` | `dict` | Optional, defaults to `{}` |
| `__repr__` | `str` | Multi-line compact summary |

### `AnalysisResult` (`ResultBase`)

- No additional fields.
- Used for decomposition / projection estimators.
- Three prototype estimators show no pressure to extend.

### `FitResult` (`ResultBase`)

- No additional fields.
- Used for optimisation / curve-fitting estimators.
- One prototype shows no pressure to extend.

### Output value types

Values in `outputs` and `diagnostics` may be:

- `NDDataset` (preferred for scientific arrays)
- raw `ndarray` (acceptable — SVD `U`, `s`, `VT`)
- scalar `float` / `int` (acceptable — Optimize `cost`, `niter`, `ncalls`)

### Parameter value types

Values in `parameters` must be JSON-compatible or Traitlets-native types:

- `str`, `int`, `float`, `bool`, `None`
- No live objects, no callables, no large arrays in `parameters`

---

## 3. Remaining migration candidates

### 3.1 NMF (low risk)

**File:** `src/spectrochempy/analysis/decomposition/nmf.py`

**`_outfit` structure:** The sklearn `NMF` fitted object (one element).

**Current public surface:**

- `components` (via `_get_components()` → `self._nmf.components_`)
- `transform()` → `self._nmf.transform(X)`
- `inverse_transform()` → `self._nmf.inverse_transform(X_transform)`
- No `scores` property, no `loadings` property (unlike PCA)

**Outputs for `result`:**

| Key | Source | Type |
|---|---|---|
| `components` | `self._nmf.components_` | ndarray |
| `W` | `self._nmf.fit_transform(X)` or `transform()` | ndarray |
| `H` | alias for `components` | ndarray |

**Diagnostics:**

- `reconstruction_err_` from sklearn `NMF` object
- `n_iter_` from sklearn `NMF` object

**Parameters:** `n_components`, `init`, `solver`, `beta_loss`, `tol`,
`max_iter`, `random_state`, `alpha_W`, `alpha_H`, `l1_ratio`, `shuffle`

**Risk:** Low. Structure is closer to PCA than to MCRALS. The sklearn object
stores everything. `_outfit` is already a single sklearn object, not a
positional tuple.

### 3.2 MCRALS (high risk)

**File:** `src/spectrochempy/analysis/decomposition/mcrals.py`

**`_outfit` structure** (returned by `_fit`):

```python
return (
    C,                      # [0] ndarray — final concentrations
    St,                     # [1] ndarray — final spectra (also `_components`)
    C_constrained,          # [2] ndarray — last constrained C
    St_ls,                  # [3] ndarray — last unconstrained St
    extraOutputGetConc,     # [4] list — external function extras
    extraOutputGetSpec,     # [5] list — external function extras
    C_constrained_list,     # [6] list — iteration history (if storeIterations)
    C_ls_list,              # [7] list — iteration history
    St_constrained_list,    # [8] list — iteration history
    St_ls_list,             # [9] list — iteration history
)
```

**Current public surface:**

| Property | Source |
|---|---|
| `C` | `transform()` → `_outfit[0]` |
| `St` / `components` | `_outfit[1]` (via `_get_components`) |
| `C_constrained` | `_outfit[2]` (wrapped via decorator) |
| `St_ls` | `_outfit[3]` (wrapped via decorator) |
| `extraOutputGetConc` | `_outfit[4]` |
| `extraOutputGetSpec` | `_outfit[5]` |
| iteration lists | `_outfit[6..9]` |

**Unique challenges:**

1. **External callables** (`getConc`, `getSpec`) — serialized via `dill` +
   `base64`. Result object must not require live callables.
2. **Iteration history** — lists of arrays over ALS steps. Large, optional
   (controlled by `storeIterations`). May need deferred or conditional
   inclusion in result.
3. **Constraint parameters** — ~30 Traitlets config params. Many are lists,
   enums, or callables. The `parameters` dict needs careful curation.
4. **Wrapped outputs** — `C_constrained` and `St_ls` use
   `_wrap_ndarray_output_to_nddataset` decorator. Result object would need
   access to the estimator's `_C_2_NDDataset` / `_St_2_NDDataset` methods or
   replicate the wrapping.

**Outputs for `result`:**

| Key | Source | Risk |
|---|---|---|
| `C` | `_outfit[0]` → wrapped | Low |
| `St` | `_outfit[1]` → wrapped | Low |
| `C_constrained` | `_outfit[2]` → wrapped | Low |
| `St_ls` | `_outfit[3]` → wrapped | Low |
| `extraOutputGetConc` | `_outfit[4]` | Medium — arbitrary content |
| `extraOutputGetSpec` | `_outfit[5]` | Medium — arbitrary content |

**Diagnostics:**

- Convergence info: `tol`, `max_iter`, `maxdiv` parameters + `niter` from ALS
  loop
- Residual norm change at convergence
- PCA comparison reconstruction error

**Risk:** High. This is the most complex `_outfit` in the codebase. Requires
an explicit audit before migration.

---

## 4. Open questions

### 4.1 Cache strategy

All three prototypes create new result objects on every access. Decisions
needed:

- Add a `_result` attribute with invalidation on re-fit?
- Keep lazy creation but memoize within a fit session?
- Profile first — does repeated creation have measurable cost?

### 4.2 Provenance enrichment

The RFC defines structured provenance (version, timestamp, input summary).
No prototype implements it. Should provenance be added as a PR5 or deferred
entirely (post-MCRALS)?

### 4.3 DisplaySection / HTML

No prototype implements `DisplayItem` / `DisplaySection` integration. The
RFC defines minimum HTML representation. Should this be a separate PR or
folded into each migration PR?

### 4.4 Serialization

No prototype implements `__getstate__` / `__setstate__` or Project save/load
for result objects. The RFC defines serialization boundaries but defers
full round-trip.

Recommended approach: deferred until after all AnalysisResult migrations
(PR4–PR6), then add serialization in a single pass.

### 4.5 Project integration

Should result objects become a third typed Project member? Deferred by
RFC. Recommend re-evaluating after serialization exists.

### 4.6 FitParameters for Optimize

Optimize has a rich `FitParameters` object. PR3 explicitly excluded it from
`result.parameters`. Should a future PR integrate `FitParameters` into
`FitResult.parameters`?

### 4.7 Residuals in FitResult

PR3 noted no stable public surface for residuals. If one emerges, should
`residuals` be added to `FitResult.outputs`?

---

## 5. Proposed sequence

```
PR4: NMF AnalysisResult
  └─ low risk, quick win, validates PCA pattern for a second decomposition

PR5: Provenance enrichment (optional — see §4.2)
  └─ add structured provenance to ResultBase if deemed necessary before MCRALS

PR6: MCRALS audit + AnalysisResult
  └─ explicit audit before any code changes
  └─ highest risk migration, requires careful planning

PR7+: FitResult improvements
  └─ residuals, FitParameters integration, richer solver diagnostics

Later:
  └─ Serialization round-trip for result objects
  └─ Project integration
  └─ DisplaySection / HTML integration
  └─ Cache strategy
```

### Rationale

- NMF is the natural next step: same module (`decomposition`), same base
  class (`DecompositionAnalysis`), same sklearn backend pattern. Low risk,
  high confidence.
- MCRALS is deferred because its complexity would slow progress if tackled
  before NMF. An explicit audit (PR6) must precede any MCRALS code changes.
- Provenance enrichment could either precede MCRALS (so MCRALS result gets
  provenance from the start) or wait until after all initial migrations.
  Decision deferred to maintainers.
- FitResult improvements are post-MCRALS because `FitResult` is already
  stable and Optimize is the only consumer.
- Serialization, Project integration, and DisplaySection are deferred to a
  later phase when all result-producing estimators implement the contract.

---

## 6. Non-goals (for this roadmap)

- No redesign of `_outfit` — remains as internal implementation detail
- No redesign of `AnalysisConfigurable._fit()` signature
- No change to `fit()` return value (continues to return `self`)
- No change to existing public properties (backward compatible)
- No full serialization round-trip
- No Project integration
- No DisplaySection / HTML migration
- No caching — deferred until profiling data exists
