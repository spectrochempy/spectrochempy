# Dataset Math Protocols: Diagnostic and Direction

This note is a low-risk study branch for future work on mathematical operations
on `NDDataset`, `Coord`, and related array objects. It is intentionally not a
redesign proposal yet. The goal is to clarify the current behavior, compare it
with NumPy and xarray conventions, and identify incremental directions before
tackling larger pieces such as quaternion support.

## Context

SpectroChemPy already has a substantial math layer in
`spectrochempy.core.dataset.arraymixins.ndmath.NDMath`. It supports:

- Python arithmetic operators through dynamically installed operator methods.
- NumPy ufunc dispatch through `__array_ufunc__`.
- Unit-aware arithmetic through Pint.
- Mask propagation.
- Basic coordinate compatibility checks.
- Special handling for complex and quaternion-like data.
- Public API functions generated from `NDMath` methods.

This gives users a convenient experience, but the implementation has grown into
a single dense path where several independent policies are interleaved:

- operand normalization and result type selection;
- unit compatibility and unit result calculation;
- coordinate compatibility;
- mask handling;
- quaternion and complex handling;
- NumPy ufunc execution;
- history/title metadata updates.

That makes future changes risky, especially for quaternion data, plugin-provided
domain behavior, and NumPy 2.x compatibility.

## Relevant External Conventions

The most relevant NumPy NEPs and xarray conventions are:

- NumPy NEP 13, `__array_ufunc__`: custom array objects should use
  `__array_ufunc__(self, ufunc, method, *inputs, **kwargs)` for ufunc dispatch,
  return `NotImplemented` when they cannot handle an operation, and maintain a
  coherent type hierarchy for mixed-type operations.
  Reference: https://numpy.org/neps/nep-0013-ufunc-overrides.html
- NumPy NEP 18, `__array_function__`: high-level NumPy functions can dispatch to
  custom array implementations, but the surface is broad and should be adopted
  selectively.
  Reference: https://numpy.org/neps/nep-0018-array-function-protocol.html
- NumPy NEP 22, duck typing overview: libraries should interoperate through
  explicit protocols rather than relying only on ndarray subclassing.
  Reference: https://numpy.org/neps/nep-0022-ndarray-duck-typing-overview.html
- NumPy NEP 50, scalar promotion: scalar dtype promotion is now more explicit in
  NumPy 2.x and should be tested at the SpectroChemPy boundary.
  Reference: https://numpy.org/neps/nep-0050-scalar-promotion.html
- NumPy NEP 56, Array API support: the Array API namespace is useful as a future
  compatibility target, but it is not a drop-in replacement for SpectroChemPy
  because units and coordinates remain first-class behavior here.
  Reference: https://numpy.org/neps/nep-0056-array-api-main-namespace.html
- xarray arithmetic: xarray vectorizes NumPy-style math over labelled arrays,
  aligns index coordinates automatically for binary arithmetic, and separates
  coordinate alignment rules from raw array execution.
  References:
  https://docs.xarray.dev/en/stable/user-guide/computation.html and
  https://docs.xarray.dev/en/stable/user-guide/duckarrays.html

The main lesson is not that SpectroChemPy should become xarray. The useful
lesson is architectural: dispatch, alignment, unit policy, and metadata policy
should be separate enough to test independently.

## Current SpectroChemPy Diagnosis

### Strengths

- `NDDataset` and `Coord` already behave naturally with Python operators.
- `np.sin(dataset)` and many other ufuncs return SpectroChemPy objects, which is
  the right user-facing behavior.
- Unit compatibility is central, which is important for scientific users.
- Existing tests cover many unary, binary, comparison, unit, and scalar cases.
- The behavior is mostly backward compatible with common NumPy idioms.

### Friction Points

1. `NDMath._op()` is doing too much.

   It simultaneously normalizes operands, chooses return type, checks units,
   checks coordinate compatibility, executes NumPy operations, handles masks,
   handles quaternion data, and computes metadata. This makes it hard to change
   one policy without affecting the others.

2. `__array_ufunc__` only handles the direct call path robustly.

   NumPy's protocol includes methods such as `reduce`, `accumulate`, `outer`,
   and `at`. SpectroChemPy currently routes ufunc calls through `_op()` in a way
   that is mainly oriented toward elementwise calls. Unsupported ufunc methods
   should be explicitly handled or return `NotImplemented`.

3. `__array_function__` is not implemented for `NDDataset`.

   This is not necessarily a bug. Implementing it broadly would be high risk.
   But SpectroChemPy should decide which high-level NumPy functions are part of
   the public interoperability contract, instead of relying on accidental
   behavior.

4. Coordinate compatibility is more permissive and less explicit than xarray.

   Current binary operations compare selected coordinate data, especially the
   last dimension for 1D-vs-2D cases. This is useful for spectroscopy, but the
   policy is embedded in `_op()` and is not named as an alignment policy.

5. Mixed operands need a clearer type hierarchy.

   NEP 13 recommends coherent type relations for mixed operations. SpectroChemPy
   currently uses local type-name checks and operand reversal. This works for
   common cases, but it should be made explicit before adding more array-like
   plugin types.

6. Quaternion behavior is special-cased in the central operation path.

   That is exactly the area where a future refactor could accidentally regress
   current behavior. It should be isolated behind a small execution policy before
   changing quaternion semantics.

7. Metadata propagation is implicit.

   History, title, units, mask, and coordinates are updated in several places.
   A clearer result-construction phase would make future changes easier to
   reason about.

## Recommended Direction

### Phase 0: Define the Contract Before Refactoring

Add targeted tests that describe the desired public behavior without changing
implementation yet:

- operator and ufunc equivalence, for example `a + b` and `np.add(a, b)`;
- scalar, ndarray, Quantity, Coord, and NDDataset mixed operands;
- in-place operations and explicit non-alignment behavior;
- coordinate mismatch and accepted spectroscopy-specific broadcasting;
- unit conversion and unit failure cases;
- representative complex and quaternion cases;
- unsupported ufunc methods returning `NotImplemented` or raising clear errors.

This phase is cheap and protects the existing user experience.

### Phase 1: Extract Operation Planning Internals

Without changing public behavior, split `_op()` into small internal phases:

1. `OperationRequest`
   Captures function name, ufunc method, inputs, kwargs, reflexive/in-place
   flags, and whether the call came from NumPy or Python operators.

2. `OperandPlan`
   Normalizes operands, decides the primary SpectroChemPy object, records raw
   magnitudes, masks, units, coordinate sets, and operand roles.

3. `UnitPlan`
   Computes compatibility and result units using Pint.

4. `AlignmentPlan`
   Encapsulates coordinate compatibility and future alignment choices.

5. `ExecutionPlan`
   Runs the numeric operation on magnitudes, including complex/quaternion
   policies.

6. `ResultPlan`
   Builds the returned object and applies units, mask, coordinates, title, and
   history.

The first extraction can remain private and should live near `NDMath` to keep
the diff reviewable.

### Phase 2: Make Alignment Policy Explicit

Introduce a small internal vocabulary:

- `strict`: all indexed coordinates must match;
- `spectroscopic-last-dim`: current permissive behavior for common 1D/2D
  spectroscopy operations;
- `none`: no coordinate comparison, only shape/broadcast checks.

Do not switch the default yet. First, express current behavior as one named
policy and test it. Later, we can decide whether a user-facing option is needed.

### Phase 3: Tighten NumPy Protocol Compliance

Keep `__array_ufunc__`, but make unsupported paths explicit:

- support `method == "__call__"` first;
- return `NotImplemented` for methods that are not supported yet, or raise a
  clear `TypeError` when returning `NotImplemented` would leak to users;
- respect `out` only when it can be done correctly;
- add tests for `np.add`, `np.multiply`, `np.equal`, reductions, and selected
  failure cases.

Do not implement broad `__array_function__` in the first pass. A safer first
step is a small registry of explicitly supported high-level functions if needed
later, such as `np.concatenate`, `np.stack`, or `np.trapz` replacements.

### Phase 4: Isolate Quaternion Execution

Before changing quaternion behavior, move the current quaternion-specific data
execution into a private helper with tests. The helper should receive normalized
magnitudes and return only numeric data. Unit and coordinate policy should not
live in the quaternion branch.

### Phase 5: Consider xarray Interoperability, Not Replacement

xarray's main value here is the labelled-array architecture:

- operation dispatch;
- alignment;
- computation;
- result metadata.

SpectroChemPy should not adopt xarray's exact defaults blindly because
spectroscopy often benefits from domain-specific tolerance and unit behavior.
But the separation of concerns is worth following.

## Near-Term PR Proposal

A first real PR should be limited to:

1. Add focused tests documenting current operator, ufunc, unit, coordinate, and
   quaternion behavior.
2. Introduce private `OperationRequest` and `OperationPlan` helpers.
3. Move only operand normalization and result-type selection out of `_op()`.
4. Keep behavior unchanged.
5. Run only targeted tests:

   ```bash
   conda activate scpy
   pytest tests/test_core/test_dataset/test_mixins/test_ndmath.py -q -ra
   pytest tests/test_core/test_dataset/test_coord.py -q -ra
   pytest tests/test_core/test_dataset/test_dataset.py -q -ra
   ```

This gives us a safer base for quaternion and future plugin/domain extensions.

## Open Questions

- Should coordinate-aware binary arithmetic eventually align by coordinate label,
  like xarray, or keep the current spectroscopy-first comparison rules?
- Should in-place operations deliberately avoid alignment, following xarray's
  rationale, or keep current behavior only for backward compatibility?
- Which high-level NumPy functions should be public SpectroChemPy dispatch
  contracts?
- Should plugin-provided array-like objects participate in the same math
  hierarchy, or should plugins expose conversion functions instead?
- How should units interact with NumPy scalar promotion in NumPy 2.x edge cases?

## Current Recommendation

Do not start with quaternion. Start by naming and testing the operation pipeline.
Once dispatch, unit policy, coordinate policy, and result construction are
separated, quaternion support becomes a contained numeric execution problem
rather than a cross-cutting change to the whole math layer.
