# Dataset Math Protocols: Diagnostic and Direction

This note is a low-risk study branch for future work on mathematical operations
on `NDDataset`, `Coord`, and related array objects. It is intentionally not a
redesign proposal yet. The goal is to clarify the current behavior, compare it
with NumPy and xarray conventions, and identify incremental directions before
tackling larger pieces such as quaternion support. In the context of the plugin
architecture work, this also means identifying which numerical capabilities
must remain core behavior and which ones should become optional or
domain-specific extensions.

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

The plugin work, especially the direction explored around PR 952, adds an
important constraint: optional scientific domains should not force all users to
pay for domain-specific behavior at import time or in the central execution
path. Quaternion handling is the clearest example, but the same question applies
to some complex-data behavior. Infrared users should keep a simple, fast, and
predictable real-valued path, while NMR or future domains can opt into richer
numeric semantics when they need them.

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

7. Complex-data behavior is also mixed into the common path.

   Complex numbers are more general than quaternion data and should remain well
   supported, but not every scientific workflow needs the same complex-aware
   behavior. The future operation layer should allow a real-valued default path,
   a core complex path, and plugin/domain extensions without making every
   operation branch through all possible numeric modes.

8. Metadata propagation is implicit.

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
- representative real, complex, and quaternion cases, with separate tests for
  the real-valued default path and optional/domain-specific numeric paths;
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
   Runs the numeric operation on magnitudes. This should start with a small
   real-valued/default execution path and delegate complex or quaternion behavior
   to explicit numeric policies instead of embedding those branches in the
   common path.

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

### Phase 4: Decouple Numeric Execution Policies

Before changing quaternion behavior, move the current complex and
quaternion-specific data execution into explicit private helpers with tests. The
helpers should receive normalized magnitudes and return only numeric data. Unit
and coordinate policy should not live in those branches.

The target shape is:

- a core real-valued/default execution policy used by most spectroscopy
  operations;
- a core complex execution policy only where complex data are actually present
  or explicitly requested;
- a quaternion execution policy that can later be moved behind a plugin or
  optional extension boundary if that is the cleanest outcome;
- no hard dependency from ordinary dataset arithmetic on quaternion-specific
  imports or assumptions.

This mirrors the plugin architecture principle: the core may expose a generic
extension point, but domain-specific numeric behavior should be registered or
selected explicitly.

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
   numeric-mode behavior.
2. Introduce private `OperationRequest` and `OperationPlan` helpers.
3. Move only operand normalization and result-type selection out of `_op()`.
4. Name the current real, complex, and quaternion branches as internal execution
   policies, without moving them to plugins yet.
5. Keep behavior unchanged.
6. Run only targeted tests:

   ```bash
   conda activate scpy
   pytest tests/test_core/test_dataset/test_mixins/test_ndmath.py -q -ra
   pytest tests/test_core/test_dataset/test_coord.py -q -ra
   pytest tests/test_core/test_dataset/test_dataset.py -q -ra
   ```

This gives us a safer base for complex/quaternion decoupling and future
plugin/domain extensions.

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
- Should quaternion execution remain in core as an optional internal policy, or
  become a plugin-provided numeric policy once the extension point is stable?
- Which complex-data operations are truly generic core behavior, and which ones
  are domain-specific convenience behavior?

## Current Recommendation

Do not start with quaternion. Start by naming and testing the operation pipeline.
Once dispatch, unit policy, coordinate policy, and result construction are
separated, complex and quaternion support become contained numeric execution
policies rather than cross-cutting changes to the whole math layer. This is
aligned with the plugin direction: the core owns the generic protocol and the
common real-valued path; optional domains should be able to provide richer
numeric behavior without making that behavior central for every user.
