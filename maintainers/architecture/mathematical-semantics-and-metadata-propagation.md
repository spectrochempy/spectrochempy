[Maintainers](../../README.md) · [Architecture](../INDEX.md)

# Mathematical Semantics and Metadata Propagation

Status: Draft RFC / characterization largely complete

## Purpose

This document records the current maintainer-facing understanding of
SpectroChemPy mathematical semantics and metadata propagation.

The goal is to define the behavior contract before any `NDMath` refactor,
result-assembly cleanup, or class hierarchy split.

This RFC now treats result assembly as a first-class audit axis. The emerging
concern is not primarily numerical execution. The larger architectural risk is
how a computed result is reattached to scientific context.

This document focuses on behavior, not class redesign. It does not introduce
`NDLabelled`, does not redesign `NDArray` / `Coord` / `NDDataset`, and does not
propose production code changes.

The central question is:

```text
When an operation produces a result, what happens to data, units, masks,
coordinates, labels, metadata, title, name, and history?
```

## Relationship to Array Class Responsibility

The related document
[`array-class-responsibility.md`](array-class-responsibility.md)
explains where responsibilities currently live.

This document explains how operations currently behave and how their behavior
should be characterized.

The distinction matters:

- Array Responsibility maps ownership across `NDArray`, `NDComplexArray`,
  `Coord`, `NDDataset`, `NDMath`, and `NDIO`.
- Mathematical Semantics maps operation behavior across arithmetic, ufuncs,
  reductions, shape changes, combinations, labels, masks, units, coordinates,
  metadata, and plugin-backed numeric behavior.

Current maintainer guidance:

```text
Clarify math semantics before introducing NDLabelled or restructuring
NDArray / Coord / NDDataset.
```

Hierarchy cleanup should preserve a known operation contract, not accidentally
define one.

## Method

This draft was prepared from:

- source review of `NDMath`, `NDArray`, `NDComplexArray`, `Coord`,
  `NDDataset`, `CoordSet`, `concatenate`, and representative processing /
  analysis operations;
- review of existing core, processing, and analysis tests;
- focused behavioral checks in the local `scpy-core` environment for
  representative arithmetic, ufunc, reduction, metadata, and `Coord` cases;
- prior architecture audits on metadata propagation, coordinate arithmetic,
  dataset-vs-coordinate arithmetic, NDMath maintainability, array
  responsibilities, and hypercomplex display.

No production code changes were made for this RFC.

PR1 characterization tests (144 tests) were added in
`tests/test_core/test_dataset/test_math_semantics_baseline.py` covering
arithmetic and ufunc semantics.

PR2 characterization tests (101 tests) were added in
`tests/test_core/test_dataset/test_shape_semantics_baseline.py` covering
shape operations and CoordSet semantics.

PR3 characterization tests (100 tests) were added in
`tests/test_core/test_dataset/test_reduction_semantics_baseline.py` covering
dimension reduction, keepdims, units, masks, CoordSet, metadata, history,
 ROI, and identity/provenance for sum, mean, std, min, max, argmin,
and argmax.

PR4 characterization tests (63 tests) are in
`tests/test_core/test_dataset/test_combination_semantics_baseline.py` covering
concatenate and stack data, CoordSet, units, masks, metadata, history,
 ROI, identity/provenance, coordinate-unit preservation, compatible
coord-unit conversion, stack origin and meta propagation, and edge cases.

PR5 characterization tests (135 tests) are in
`tests/test_core/test_dataset/test_indexing_semantics_baseline.py` covering
return type, shape/dims, CoordSet, units, masks, metadata, history, ROI,
identity, provenance, scalar extraction, label indexing,
ellipsis, step slicing, float indexing, and edge cases for integer, slice,
label, bool, and list indexing forms.

PR6 characterization tests (81 tests) are in
`tests/test_processing/test_processing_wrapper_semantics_baseline.py` covering
processing wrapper semantics for smooth, savgol, whittaker, basc, detrend,
asls, and denoise: return type, shape/dims, CoordSet, units, masks, metadata,
history, ROI, identity, provenance, and shared behavior. Reveals
two distinct assembly patterns (Group A: Filter/PCA-based, Group B:
Baseline-based).

PR8 characterization tests (38 tests) are in
`tests/test_analysis/test_integration/test_integration_semantics_baseline.py`
covering integration semantics for `trapezoid()` / `simpson()`: return type,
single-axis dimensional reduction, CoordSet reduction, unit transformation,
metadata overrides, rewritten provenance, identity as derived quantity, label
survival on non-integrated dimensions, mask behavior, and representative edge
cases.

See Audit Policy in `AGENTS.md` for test-first refactoring requirements.

> **Note (June 2026):** `modeldata` has been removed from `NDDataset` as
> orphaned historical infrastructure.  All references to `modeldata` below
> describe historical characterization-test behavior that is no longer active.
> See `maintainers/rfcs/modeldata-semantic-contract.md` and issue `#1168` for
> the full audit and maintainer decision.
>
> **Decision update (June 2026):** `roi` has also been classified as orphaned
> historical interactive-selection state and removed from the public runtime
> `NDDataset` model.  References to `roi` below describe the historical
> behavior characterized during PR2–PR7 before that removal.  Legacy
> serialized `roi` / `modeldata` fields should be ignored on load rather than
> restored onto live objects.  See `maintainers/rfcs/roi-semantic-contract.md`
> and `maintainers/rfcs/modeldata-semantic-contract.md`.

## Semantics Matrix

This matrix summarizes the current behavior by operation family.

It is intentionally compact. Details and open questions are expanded in the
following sections.

| Operation family | Data | Units | Masks | Coordinates | Labels | Metadata | Title | History | Return type |
|---|---|---|---|---|---|---|---|---|---|
| Binary arithmetic | NumPy-like operation on magnitudes | compatible or algebraic rules | propagated through masked data | dataset-vs-dataset last-dim check | mostly copied through coordinates | broad copy-first preservation | preserved or operation-adjusted | appended | usually left semantic object |
| Comparisons | elementwise comparison | compatible units required where relevant | propagated or raw masked result | last-dim check for datasets | copied only if object result | often dropped for raw result | usually not meaningful | usually none | dataset-like or raw array depending path |
| Ufuncs | elementwise operation | function-specific requirements | propagated for dataset-returning paths | generally copied | generally copied through coordinates | broad copy-first preservation | may be wrapped, e.g. `sqrt(title)` | ufunc history where dataset-like | dataset-like or raw array |
| Reductions | reduced data or scalar | usually preserved, operation-specific | reduced through masked arrays | reduced, removed, or selected | reduced/removed with coordinates | partly copied, partly operation-specific | varies | varies | scalar, `Quantity`, or dataset |
| Shape operations | shape/dims changed | preserved | shape-aligned | transformed, dropped, or rebuilt | selection-like paths preserve best | broad copy-first preservation | usually preserved | some operations append | same object type |
| Interpolation | resampled data | data units preserved | interpolated/reordered locally | target coordinates rebuilt | exact matches preserve labels, resampled points unlabelled | copy-first plus local edits | preserved | appended | `NDDataset` |
| Integration | integrated data | data units times coordinate units | operation-specific | integrated dim removed | removed with integrated coord | copy-first plus local edits | often operation-defined | rewritten locally | reduced dataset or scalar-like |
| Indexing / selection | sliced data | preserved | sliced with data | sliced along indexed dim; CoordSet preserved | labels follow coord slice | copy-first preservation | preserved | appended with "Slice extracted" | always `NDDataset` (even single-element) |
| Concatenation | concatenated data | compatible, converted to first units | concatenated with data | concatenated selected dim | concatenated selected coord labels | synthesized multi-source metadata | first title wins | reset/synthesized | `NDDataset` |
| Stack | new leading dim plus concatenation | same as concatenate | stacked with data | new coord labels from dataset names | new stack labels from names | follows concatenate | follows concatenate | follows concatenate | `NDDataset` |
| Analysis outputs | derived result | local operation rules | local operation rules | local operation rules | local operation rules | often synthesized locally | often synthesized locally | often rewritten | operation-specific |
| Processing outputs (Group A: Filter/PCA) | processed data | preserved | wrapper-dependent | preserved (unchanged) | mostly preserved | name appended/rewritten, roi recomputed, history rewritten | author preserved (except denoise where overridden), title preserved | rewritten to single method entry | `NDDataset` (wrapper-assembled) |
| Processing outputs (Group B: Baseline) | baseline-corrected data | preserved | wrapper-dependent | preserved (unchanged) | mostly preserved | name/title preserved, roi preserved | author preserved, title preserved | appended | `NDDataset` (wrapper-assembled) |

The main pattern visible in this table is that arithmetic and many ufuncs have
a relatively coherent center in `NDMath`, while processing, analysis,
integration, interpolation, and combination operations often carry local
semantic policy.

## Binary Arithmetic Semantics

Binary arithmetic is centered in `NDMath`.

The current model is:

```text
NumPy-style broadcasting
+
unit-aware arithmetic
+
mask-aware execution
+
spectroscopy-oriented coordinate validation
+
copy-based result assembly
```

Representative observed behavior:

| Operation family | Current result | Units | Coordinates | Metadata/title/history |
|---|---|---|---|---|
| `NDDataset + scalar` | `NDDataset` | preserved/compatible | copied from left operand | broad metadata survives by copy; history appended |
| `NDDataset * scalar` | `NDDataset` | transformed by unit algebra when needed | copied from left operand | broad metadata survives by copy; title may be operation-dependent |
| `NDDataset + NDDataset` | `NDDataset` | compatible units required | last-dimension coordinate check | result assembled from a copy; history appended |
| `NDDataset * NDDataset` | `NDDataset` | product units | last-dimension coordinate check | result assembled from a copy; history appended |
| `Coord + scalar` | `Coord` | current coordinate units preserved | not applicable | label/metadata behavior follows `Coord` copy path |
| `Coord * scalar` | `Coord` | current coordinate units preserved in observed behavior | not applicable | scientifically meaningful but should be reviewed |
| `Coord + Coord` | `Coord` | compatible units required | not applicable | labels and metadata need characterization |
| `Coord` with `NDDataset` | unsupported | compatible-unit checks may happen first | not applicable | explicit semantic `TypeError` when unit checks do not fail first |

Important current rule:

Arithmetic between `NDDataset` and `Coord` is intentionally unsupported when the
operation reaches the semantic check:

```text
Coord represents an axis, not signal data.
```

This reflects the Dataset-vs-Coord semantic decision: use a 1D `NDDataset` for
signal-like correction vectors, not a `Coord`.

### Dataset vs Scalar

Observed current behavior:

- returns an `NDDataset`;
- starts from a copy of the left operand;
- preserves broad metadata such as `name`, `description`, and custom `meta`
  through copying;
- appends operation history when the result has history support;
- preserves or transforms units according to operation type;
- preserves masks through masked-array execution when masks are present.

Open questions:

- Which copied fields are intentionally preserved versus incidentally
  preserved?
- Should `name` survive scalar arithmetic unchanged?
- Should history wording be standardized by operation family?

### Dataset vs Dataset

Observed current behavior:

- returns an `NDDataset`;
- compatible units are required for additive/comparison-style operations;
- multiplication/division use unit algebra;
- coordinate compatibility is checked under the current
  `spectroscopic-last-dim` policy;
- only the last dimension is validated for coordinate compatibility in
  dataset-vs-dataset arithmetic;
- earlier dimensions are not full alignment keys.

Open questions:

- Is last-dimension-only validation the long-term policy or only the legacy
  baseline?
- Should all-dimension validation be considered for some operation families?
- Should coordinate units be converted or compared more explicitly?

### Coord vs Scalar

Observed current behavior:

- `Coord + scalar`, `Coord - scalar`, `Coord * scalar`, and `Coord / scalar`
  use the shared math machinery and return a `Coord`;
- coordinate units remain attached in observed scalar cases;
- coordinate labels and metadata behavior are not yet fully characterized.

Semantic concern:

`Coord` is an axis/support object, so arithmetic on coordinates can be
scientifically meaningful for axis transforms, but not all dataset-style math is
appropriate.

Open questions:

- Should scalar multiplication/division of coordinate values preserve units,
  transform units, or be operation-specific?
- Should labels survive coordinate arithmetic, or should transformed coordinates
  drop labels unless the operation is a pure selection?

### Coord vs Coord

Observed current behavior:

- compatible-unit coordinate arithmetic can return a `Coord`;
- incompatible units fail via unit compatibility;
- labels and metadata propagation need stronger tests.

Open questions:

- Is `Coord + Coord` conceptually an axis transform, a quantity operation, or a
  discouraged operation?
- Should labels survive when coordinate values are arithmetically combined?

### Coord vs Dataset

Observed current behavior:

- arithmetic between `Coord` and `NDDataset` is blocked by an explicit semantic
  `TypeError` once operands pass earlier unit checks;
- incompatible units may raise a unit error before the semantic `TypeError`.

Current interpretation:

`Coord` should not be used as signal data in dataset arithmetic.

Open question:

- Should error ordering be standardized so the semantic Coord-vs-Dataset error
  always wins over unit errors?

### Unitful vs Unitless and Compatible vs Incompatible Units

Observed current behavior:

- additive operations require compatible dimensionality;
- compatible operands may be converted to the left operand's units;
- multiplication/division produce algebraic units;
- some ufuncs require dimensionless or angular units;
- logical/testing operations may remove units or return raw arrays.

Open questions:

- Should unitless scalar addition to unitful datasets remain accepted through
  implicit left-unit interpretation?
- Should coordinate units participate in arithmetic compatibility more strongly?

## Ufunc Semantics

`NDMath.__array_ufunc__()` supports ordinary ufunc calls and rejects unsupported
ufunc methods by returning `NotImplemented`.

Representative behavior:

| Ufunc family | Current behavior | Unit behavior | Return type |
|---|---|---|---|
| `abs` / `absolute` | elementwise magnitude | preserves input units | generally `NDDataset` |
| `sqrt` | elementwise square root; negative values may promote to complex | square-root units | generally `NDDataset` |
| `log`, `exp` | requires dimensionless input | incompatible units raise | generally `NDDataset` when accepted |
| `sin`, `cos`, `tan` | requires angular or dimensionless-compatible input | angular rules apply | generally `NDDataset` when accepted |
| `real`, `imag` | component extraction for complex data | depends on method path | `NDDataset`-like component where supported |
| `isfinite`, `isnan`, `isinf`, `signbit`, `logical_not` | raw testing/logical behavior | units removed | raw NumPy/masked-array result |

Observed examples:

- `np.sqrt(NDDataset(..., units="m"))` returns an `NDDataset` with square-root
  units.
- `np.log(NDDataset(..., units="m"))` raises a dimensionality error.
- `np.isfinite(NDDataset(...))` returns a masked array rather than an
  `NDDataset`.

Consistent behaviors:

- many elementwise ufuncs preserve the dataset object model;
- unit requirements are enforced for dimension-sensitive functions;
- masks are respected through masked-array data where applicable.

Surprising or ambiguous behaviors:

- some ufuncs return raw arrays while nearby numeric ufuncs return datasets;
- title updates are operation-specific;
- metadata survival is mostly copy-driven;
- ufunc support is not complete for `reduce`, `accumulate`, `outer`, or `at`.

Complex/hypercomplex note:

- ordinary complex behavior is handled in core through `NDComplexArray` and
  `NDMath`;
- plugin-backed hypercomplex execution uses plugin hooks for non-core numeric
  branches;
- this document does not redesign those plugin hooks.

## Reduction Semantics

Reductions are higher risk because they change shape and often change scientific
meaning.

Representative behavior:

| Reduction | Current result pattern | Units | Coordinates | Metadata/title/history |
|---|---|---|---|---|
| `sum` | scalar quantity or reduced object | generally preserved | reduced or removed | history/title behavior varies |
| `mean` | scalar quantity or reduced object | generally preserved | reduced or removed | history/title behavior varies |
| `std` / `var` | scalar quantity or reduced object | operation-specific | reduced or removed | needs characterization |
| `min` / `max` | scalar quantity or reduced object | preserved when scalar quantity | may preserve selected/reduced coords | special coordinate selection for global extrema |
| `argmin` / `argmax` | index/coordinate-related result | not always dataset-like | operation-specific | needs characterization |
| cumulative ops | shape-preserving or dimension-aware | operation-specific | likely copied | some are unsupported for `Coord` |
| `trapezoid` / `simpson` | integrated dataset, including 0-d dataset results | multiplies data units by coordinate units | integrated dimension removed | local title/description/history rewrite for derived quantity |

Observed current behavior:

- reducing all dimensions can return a `Quantity` rather than an `NDDataset`;
- named-dimension reductions can return reduced datasets;
- `keepdims=True` preserves singleton dimensions where implemented;
- many ordinary reductions use masked-array calculation, but integration is a
  notable exception: mask information survives on the returned object while
  masked source values still appear to contribute numerically to the integral;
- coordinate propagation is handled by reduction helpers and local operation
  code.

Consistent behaviors:

- scalar reductions tend to return scalar quantities when units are present;
- reduced dimensions are removed unless `keepdims=True`;
- many reductions preserve units when the physical dimension of the value is
  unchanged.

Ambiguous or surprising behaviors:

- title/history behavior is not uniform;
- global extrema may rebuild coordinates differently from ordinary reductions;
- integration has local physical semantics and result assembly rules;
- integration is the clearest current example of a derived scientific quantity
  generated through reduction;
- cumulative operations are not uniformly meaningful for `Coord`.

Open questions:

- Which reductions should return `Quantity`, `NDDataset`, or plain scalars?
- Should metadata preservation differ between statistical and physical
  reductions?
- Should integration be governed by a shared reduction/geometry contract?

### Reduction Families

PR3 characterization tests (100 tests) identify three distinct reduction
categories with different semantic behaviors:

**Aggregating reductions (`sum`, `mean`, `std`):**
  Return type depends on dimensionality: global reduction (`dim=None`) returns
  a plain scalar (or `Quantity` if units present), named-dim reduction returns
  `NDDataset`, and `keepdims=True` preserves singleton dimensions. History is
  appended with the method name. Units are preserved consistently. CoordSet is
  reduced (dropped for global, surviving coords preserved for named-dim).
  Extremum coordinates are not reconstructed.

**Selection reductions (`min`, `max`):**
  Same return-type pattern as aggregating reductions for most cases. However,
  `max(dim=None, keepdims=True)` and `min(dim=None, keepdims=True)` reconstruct
  coordinate values at the extremum location rather than dropping all
  coordinates. This asymmetric behavior is not present in aggregating
  reductions, where `sum(dim=None, keepdims=True)` drops all coordinates.
  History records `amax`/`amin` (not `max`/`min`), reflecting the underlying
  numpy dispatch.

**Index reductions (`argmin`, `argmax`):**
  Never return `NDDataset`. Return type depends on dimensionality: `ndarray`
  for named-dim reductions, `tuple` of ints for global reduction (unraveled
  multi-index), and scalar `int` for 1D datasets. The `keepdims` parameter is
  accepted but has no observable effect on return type or shape — it is
  effectively a no-op for index reductions.

## Shape Operation Semantics

Shape operations include slicing, transpose, swapdims, squeeze, reshape, and
dimension expansion.

Representative behavior:

| Operation | Data/dims behavior | CoordSet behavior | Metadata behavior |
|---|---|---|---|
| slicing | data sliced via array path | `NDDataset` slices coordinates afterward | broad metadata copied; history records slice |
| `transpose` | dims permuted | coordinates follow dims | broad metadata copied; history appended |
| `swapdims` | two dims swapped | coordinates follow dims | broad metadata copied; history appended |
| `squeeze` | singleton dims removed | squeezed coords dropped | broad metadata copied |
| `reshape` | data reshaped; dims may be provided | coordinate policy controls preserve/drop/strict | broad metadata copied; structural metadata recomputed |
| `atleast_2d` / expansion | new singleton dims introduced | default coords created | broad metadata copied |

Consistent behaviors:

- data, masks, dims, and coordinates are generally kept aligned;
- shape operations usually start from a copy and therefore preserve scientific
  context broadly;
- `reshape` has an explicit coordinate policy because old coordinates may
  become misleading.

Ambiguous or surprising behaviors:

- history is not equally explicit for every structural operation;
- label propagation depends on whether labels are direct labels or coordinate
  labels;
- reshape has clearer coordinate policy than some older shape paths.

Open questions:

- Which shape operations should append history?
- Should structural metadata such as `roi` always be dropped or
  recomputed?
- Should label preservation be limited to selection-like operations?

## Combination Operation Semantics

Combination operations include `concatenate`, `stack`, append-like behavior, and
local merge-like assembly paths in analysis/processing code.

Representative current behavior:

| Operation | Data/mask behavior | Units | Coordinates/labels | Metadata |
|---|---|---|---|---|
| `concatenate` | masked arrays concatenated | compatible units required; converted to first units | concatenated along selected dim; coord units preserved; compatible coord units auto-converted to first dataset's coord units | title from first dataset; authors combined; description/history rewritten; origin/meta/roi from last dataset (copy artifact) |
| `stack` | adds new leading dimension then delegates to concatenate | same compatibility as concatenate | new coordinate labels derive from dataset names | metadata follows concatenate path; origin from last dataset; meta from last dataset (deep-copied) |
| analysis assembly | operation-specific | operation-specific | operation-specific | often locally synthesized |

Consistent behaviors:

- concatenation checks non-concatenated shapes;
- data units must be compatible;
- masks are concatenated with the data;
- coordinate labels along the concatenated dimension can be concatenated;
- coordinate units are preserved for both concatenated and non-concatenated dimensions;
- compatible coordinate units (e.g., m and cm) are auto-converted to the first
  dataset's coordinate units, with data values converted accordingly;
- `origin` propagates from the last dataset for both concatenate and stack;
- custom `meta` propagates from the last dataset and is deep-copied (not aliased
  to the source object).

Ambiguous or surprising behaviors:

- title from the first dataset wins when titles differ;
- `description`, `author`, and `history` are synthesized locally;
- custom `meta` merge/drop behavior is not yet a clear contract — current
  behavior takes meta from the last dataset, which may or may not be intended
  for multi-source operations;
- `modeldata` (removed from `NDDataset`) retained the shape of the last input
  dataset, not the concatenated output shape (stale);
- plugin post-processing can affect concatenation semantics.

Open questions:

- Should custom `meta` be merged, preserved from first input, or dropped for
  multi-source operations?
- Should title/name behavior be standardized across all combination operations?
- Should stack/concatenate expose a documented metadata policy?

## Indexing / Selection Semantics

Indexing and selection operations access subsets of an `NDDataset` using
positional integer indexing, slice notation, ellipsis, label-based slicing,
float-based nearest-value lookup, and fancy indexing (boolean arrays and
integer lists).

All indexing is through the unified `ds[...]` syntax. There are no separate
`loc`/`iloc`/`sel`/`isel` accessors. The type of indexer is auto-detected
and dispatched through `_make_index` -> `_get_slice` -> `_loc2index`.

PR5 characterization tests (135 tests) are in
`tests/test_core/test_dataset/test_indexing_semantics_baseline.py`.

Compact summary:

```text
Return type:
    always NDDataset (even single-element extraction)

Shape/dims:
    dims preserved with singleton axes (not squeezed)

CoordSet:
    sliced with data, titles/units preserved

Labels:
    follow coordinate slicing

Units:
    preserved for all indexing forms

Masks:
    sliced with data; scalar False if no masked elements

Metadata (title, name, author, description, origin, meta):
    preserved via copy-first assembly (deep-copied)

History:
    appended with "Slice extracted: ..." entry

ROI:
    preserved unchanged (stale-field risk)

Identity:
    preserved (same object with restricted support)

Provenance:
    preserved + history append
```

### Return Type

All indexing forms that return a result return `NDDataset`, even
single-element extraction (`ds[0, 0]` returns shape `(1, 1)`). The result
never leaves the `NDDataset` surface.

### Shape and Dims

- Dimensions are preserved with singleton axes, not squeezed.
- `ds[0]` on shape `(5, 7)` returns shape `(1, 7)` with dims `["y", "x"]`.
- `ds[:, 0]` returns shape `(5, 1)`.
- `ds[0, 0]` returns shape `(1, 1)`.

### CoordSet Behavior

- Coordinates are sliced with the data along the indexed dimension.
- Non-indexed dimensions preserve their coordinates unchanged.
- Coordinate titles and units are preserved.
- Step slices and negative steps correctly slice coordinates.
- Boolean and list fancy indexing slice coordinates to match the selected
  rows/columns.
- Labels follow coordinate slicing.

Classification: **Preserve with slice** for all indexing forms.

### Units

Data units are preserved through all indexing forms: integer, slice, ellipsis,
label, bool, and list fancy indexing.

### Masks

- Masks are sliced with the data.
- If the resulting slice contains no masked elements, `r.mask` may be a scalar
  `False` rather than an array.
- Mask shape matches the sliced data shape when masked elements are present.

### Metadata

All metadata fields propagate through all indexing forms:

- `title`, `name`, `author`, `description`, `origin` are preserved unchanged.
- Custom `meta` is preserved and deep-copied (not aliased).

This is a copy-first pattern: the result is assembled from a copy of the
input, so all non-geometric metadata survives.

### History

History is **appended** for all indexing forms. The original entry is
preserved and a new `"Slice extracted: ..."` entry is added. History is never
rewritten or reset for indexing operations.

### ROI

ROI is copied unchanged through all indexing forms. It is NOT sliced or
adjusted to match the new data shape. This is a clear stale-field risk.

### Modeldata

`modeldata` was previously preserved unchanged through all indexing forms
(retaining the original full shape even after subsetting — a clear stale-field
risk).  It has since been **removed from `NDDataset`** as orphaned historical
infrastructure.  See the modeldata RFC for details.

### Identity

- Slicing preserves identity: title, name, author, description, origin, and
  meta all survive via copy-first assembly.
- Even single-element extraction (`ds[0, 0]`) returns an `NDDataset` with
  full identity — identity never leaves the dataset surface.

### Provenance

- `origin` and `author` are preserved via copy-first.
- History is extended (appended), not rewritten.

### Result Assembly Interpretation

Indexing follows a **Copy-First with Sliced Assembly** pattern:

```text
copy input dataset
slice data, mask, and CoordSet along indexed dimensions
preserve all metadata unchanged
append history
keep roi unchanged (stale risk)
```

This is closest to Pattern A (Copy-First) with a systematic geometry-aware
slice step that the general copy-first path does not explicitly name.

## Result Assembly Patterns

Current propagation differences mostly come from result assembly, not from
numerical computation itself.

Four recurring patterns appear in the codebase.

### Pattern A: Copy-First Result Assembly

Typical users:

- binary arithmetic;
- many ufuncs;
- many shape operations;
- some interpolation and analysis paths before local edits.

Pattern:

```text
new = self.copy()
replace data / units / mask
adjust history or title locally
return new
```

Fields commonly preserved:

- `name`
- `title`, unless operation changes it;
- `author`
- `description`
- `origin`
- custom `meta`
- `coordset`, unless shape/domain changes;
- `roi`, even when it may become stale.

Semantic consequence:

This is generous and user-friendly for scientific context, but some preservation
is likely accidental. A copied `roi` or `name` may no longer be
valid after derived operations. (`modeldata` has been removed from `NDDataset`
— see the modeldata RFC.)

### Pattern B: Rebuild or Synthesize Result

Typical users:

- concatenation;
- stack through concatenation;
- some analysis outputs;
- some operations that assemble new domains.

Pattern:

```text
construct or copy one representative object
replace data / coordinates / metadata explicitly
synthesize title, description, author, or history
```

Fields commonly preserved:

- data units if compatible;
- selected coordinates;
- title from a primary input, often the first dataset.

Fields commonly synthesized:

- `description`
- `author`
- `history`
- sometimes coordinate labels.

Semantic consequence:

This pattern makes multi-source provenance more explicit, but field policies
are local to each operation.

### Pattern C: Reduction-Specific Assembly

Typical users:

- `sum`, `mean`, `min`, `max`, `std`, and related reductions;
- integration-like operations.

Pattern:

```text
reduce data along one or more dimensions
return scalar/Quantity if fully reduced
otherwise update dims and coordset
apply operation-specific metadata rules
```

Fields commonly changed:

- `dims`
- `coordset`
- `mask`
- units for physical reductions such as integration.

Semantic consequence:

Reduction semantics are highly dependent on whether the result remains a
dataset or becomes a scalar/quantity. This split should be characterized before
any result assembly cleanup.

### Pattern D: Wrapper-Based Processing Assembly

Typical users:

- processing classes and helpers that operate on raw arrays;
- filter-like transformations;
- selected analysis wrappers.

Pattern:

```text
run numeric algorithm on ndarray-like data
wrap raw output back into NDDataset
restore selected fields explicitly
```

Fields commonly restored:

- data units;
- selected dimensions / coordinates when shape-compatible;
- title or history by wrapper policy;
- masks where wrapper support exists.

Fields at risk:

- custom `meta`;
- `description`;
- `author`;
- `origin`;
- `roi`;

> **Note:** `modeldata` was previously listed among fields at risk but has been
> removed from `NDDataset` — see the modeldata RFC.

Semantic consequence:

Wrapper-based paths are the clearest source of metadata fragmentation. They may
produce scientifically valid data while preserving less context than copy-first
paths.

### Observed Wrapper Assembly Patterns (PR6)

PR6 characterization tests (`test_processing_wrapper_semantics_baseline.py`)
reveal two distinct assembly patterns among the current processing wrappers,
driven by the underlying algorithm class rather than by any shared policy.

#### Group A: Filter/PCA-Based Wrappers

Wrappers: `smooth`, `savgol`, `whittaker`, `denoise`

Pattern:

```text
name:              appended with "_Filter.transform" or "_PCA.inverse_transform"
modeldata:         dropped (set to None)  [Note: modeldata removed from the runtime array model]
roi:               recomputed from data range after processing
history:           rewritten (original entries lost; single entry "Created using method ...")
title:             preserved
author:            preserved by filter wrappers; overridden to system hostname by denoise
description:       preserved
origin:            preserved
meta:              preserved
coordset:          preserved
units:             preserved
```

Key observation: Group A wrappers behave as if producing a new or derived
dataset, not the original dataset after a transformation. The history rewrite
and name suffix are the clearest signals.

Within Group A, `denoise` (PCA-based) diverges from the filter wrappers in one
field: `author` is set to `"user@hostname"` (system-dependent) rather than
preserved from the source dataset. This may indicate an incidental difference
in how `denoise` constructs its output object versus `Filter.transform`.

#### Group B: Baseline-Based Wrappers

Wrappers: `basc`, `detrend`, `asls`

Pattern:

```text
name:              preserved unchanged
roi:               preserved unchanged
history:           appended (original entries survive, operation appended)
title:             preserved
author:            preserved
description:       preserved
origin:            preserved
meta:              preserved
coordset:          preserved
units:             preserved
```

> **Note:** `modeldata` (previously preserved, shape and values unchanged) has
> been removed from `NDDataset` — see the modeldata RFC.

Key observation: Group B wrappers behave as same-object transformations. The
history append is consistent with copy-first arithmetic assembly.

#### Summary

| Property | Group A (Filter/PCA) | Group B (Baseline) |
|---|---|---|
| Name | appended with method suffix | preserved unchanged |
| Modeldata | dropped | preserved |
| ROI | recomputed from data range | preserved unchanged |
| History | rewritten (single entry) | appended (original survives) |
| Author | preserved (except denoise) | preserved |

#### Interpretation

Group A and Group B do not appear to differ because of an explicit
processing policy.

The split currently follows implementation families
(Filter/PCA versus Baseline).

Whether this reflects an intentional semantic distinction
(derived object vs same-object transformation)
or historical implementation divergence remains unknown.

Further architectural work should avoid treating either pattern as
normative until the semantic intent is clarified.

## Result Assembly by Operation Family

Result assembly is the step that turns computed values back into a
SpectroChemPy object with units, masks, coordinates, metadata, identity, and
provenance.

The operation families reviewed in this RFC suggest that result assembly is an
implicit architectural concept already present in the codebase.

| Operation family | Dominant pattern | Intentionally preserved | Structurally recomputed | Copy-surviving fields | Stale-field risk |
|---|---|---|---|---|---|---|
| Arithmetic | Copy-first | units, masks, coordset, scientific context, history append | data, units for algebraic operations | `name`, `origin`, custom `meta`, `roi` | `roi`, possibly `name` after derived arithmetic |
| Ufuncs | Copy-first with ufunc-local edits | units when valid, masks, coordset, scientific context | data, sometimes title/history/units | `name`, `origin`, custom `meta`, `roi` | geometry-dependent fields when a ufunc changes semantic domain |
| Reductions | Reduction-specific | units where meaningful, selected context, provenance | data, dims, coordset, mask, sometimes units | copied metadata on dataset-returning reductions | stale `roi`; ambiguous `name` and title |
| Shape operations | Copy-first plus geometry edits | units, scientific context, masks when shape-aligned | dims, coordset, sometimes mask geometry | `name`, `origin`, custom `meta`, `roi` | structural metadata after reshape/squeeze/transpose |
| Interpolation | Copy-first plus domain rebuild | data units, broad context, target coordinate semantics | coordset along interpolated domain, data, mask/labels locally | `name`, `origin`, custom `meta`, possibly `roi` | copied region state tied to old coordinate grid |
| Integration | Reduction-specific / local rebuild | scientific context, integrated provenance | units, dims, coordset, title/history locally | `name`, `origin`, custom `meta` | source-domain `roi`; title/name ambiguity |
| Indexing / selection | Copy-first with sliced assembly | units, masks, coordset (sliced), scientific context, history append | coordset sliced along indexed dims | all metadata preserved, deep-copied | `roi` stale after subsetting |
| Concatenate | Rebuild / synthesize | compatible units, selected coordinates, masks | data, concatenated coord, description/author/history | first input title/name in some paths | misleading first-input identity; custom `meta` ambiguity |
| Stack | Rebuild / synthesize via concatenate | compatible units, masks, source names as stack labels | new leading dim, stack coord, multi-source provenance | concatenate-dependent fields | first-input title/name; ambiguous multi-source context |
| Processing outputs (Group A: Filter/PCA) | Wrapper-based — identity-changing | data units, coordset, scientific context after wrapper | data, name, roi, history rewritten | title, author (except denoise), description, origin, meta | name overwritten with method suffix; history lost; roi recomputed |
| Processing outputs (Group B: Baseline) | Wrapper-based — identity-preserving | data units, coordset, name, roi, scientific context | data, history appended | title, author, description, origin, meta | roi may become stale for baseline-corrected domain |

> **Note:** `modeldata` was previously listed across the matrix above as a
> copy-surviving field and stale-field risk, but has been removed from
> `NDDataset` — see the modeldata RFC.
| Analysis outputs | Rebuild / synthesize | operation-defined context | data, coordset, title, description, history, units | operation-specific | preserving source identity when output is a new object |

This table should not be read as a proposed policy. It is a map of the current
implicit behavior observed from source review and representative checks.

Important field-level observations:

- `title` is usually preserved for same-object transformations but may be
  wrapped or overwritten by ufuncs, integrations, processing, and analysis.
- `history` is the most explicit provenance mechanism, but append versus
  replacement remains operation-local.
- `coordset` behaves like geometry-dependent metadata: it is preserved only
  when geometry survives, and otherwise reduced, rebuilt, or synthesized.
- coordinate labels usually follow the coordinate object, not the data array
  directly.
- `modeldata` (removed from `NDDataset`) was previously the clearest stale-field
  risk because copy-first paths preserved it without proving it remained valid.
- `metadata`, `origin`, and `author` behave more like scientific/provenance
  context and generally should not disappear just because an internal assembly
  path changes.

## CoordSet Semantics

`CoordSet` is not only storage for coordinates. In practice, it is the
geometry contract for an `NDDataset`.

Observed CoordSet behavior can be grouped as follows.

| Operation family | CoordSet behavior | Semantic reading |
|---|---|---|
| Arithmetic | preserved from the left dataset after last-dim compatibility checks | signal values changed, coordinate geometry preserved |
| Ufuncs | generally preserved for dataset-returning ufuncs | elementwise transform over the same support |
| Reductions | reduced, dropped, or selected according to reduced dimensions | result geometry changes with dimensionality |
| Shape operations | transformed with dims or rebuilt to match new shape | coordinate structure follows array geometry |
| Interpolation | partially rebuilt along the interpolated domain | same scientific object on a new coordinate grid |
| Integration | integrated dimension removed or transformed | coordinate support changes because a domain was collapsed |
| Indexing / selection | sliced along indexed dimension, preserved elsewhere | same object with restricted support |
| Concatenate | rebuilt along concatenated dimension, preserved/validated elsewhere | multi-source geometry is synthesized |
| Stack | synthesized with a new leading coordinate, often from dataset names | result gains a provenance-like stack axis |

The current model is mostly coherent:

```text
Preserve CoordSet when the physical support is unchanged.
Reduce or rebuild CoordSet when geometry changes.
Synthesize CoordSet for multi-source or newly introduced axes.
```

The main limitation is that this model is not yet expressed as a formal
contract. Some operations validate only the spectroscopic last dimension,
whereas shape and combination operations operate more directly on full
geometry. That difference may be intentional for spectroscopy workflows, but it
should remain visible during future `NDMath` work.

CoordSet is therefore best classified as structural information, not plain
scientific context. It may carry scientifically meaningful labels and units, but
its validity depends on shape, dims, coordinate values, and domain.

## Processing vs Analysis Result Assembly

Processing and analysis outputs behave differently because they answer
different scientific questions.

Processing usually transforms an existing dataset:

```text
input dataset -> transformed dataset on the same or related support
```

Examples include filtering, baseline-like transformations, interpolation, and
other operations whose output is still understood as the same measured object
after a processing step.

Analysis often produces a new scientific object:

```text
input dataset(s) -> derived object, scores, components, features, or model output
```

This distinction explains much of the current metadata behavior:

- Processing tends to preserve units, title, description, author, origin, and
  custom metadata when the result remains the same scientific object.
- Processing must recompute or drop geometry-dependent structures when the
  support changes.
- Analysis is more likely to synthesize titles, descriptions, coordinates, and
  history because the output is not simply the input dataset with changed
  values.
- Analysis may need different units, labels, and coordinate meanings because
  the result represents model space, component space, feature space, or another
  derived domain.

The current codebase does not always label this distinction explicitly, but the
behavior often follows it. Wrapper-based processing paths are the main weak
spot: they may behave like processing conceptually while preserving context more
like a rebuilt analysis output.

PR6 characterization reveals an internal split: Group A wrappers (Filter/PCA:
`smooth`, `savgol`, `whittaker`, `denoise`) follow the analysis-like pattern —
they rewrite history, append method suffixes to names, and
recompute roi. Group B wrappers (Baseline: `basc`, `detrend`, `asls`) follow
the processing pattern — they preserve name, roi, and append
history. (modeldata was also dropped/preserved respectively but has since been
removed from the runtime array model — see the modeldata RFC.) This suggests the split is driven by the underlying algorithm class
(Filter vs. Baseline) rather than by an explicit processing-vs-analysis policy.

Object identity explains this distinction more directly than metadata
propagation alone. Processing usually preserves context because the output is
still interpreted as the same measured object after a transformation. Analysis
often synthesizes context because the output is interpreted as a new scientific
object derived from one or more inputs.

Provenance refines the distinction:

```text
Processing:
    usually preserves identity
    usually preserves and extends provenance

Analysis:
    often creates a new identity
    should still preserve source provenance
```

Counterexamples are possible. A processing operation that changes domain may
behave more like a changed-representation result, while an analysis helper may
return a dataset closely tied to the original support. The useful distinction is
therefore not the module name, but whether the output represents the same
scientific object and how source lineage remains visible.

PR9 refines the "analysis" side of this distinction further:

```text
analysis methods do not produce one semantic class of output
```

They currently produce three families:

- latent derived analysis objects;
- diagnostic / model-summary outputs;
- reconstructed source-space outputs.

## Analysis Output Families

PR9 characterized representative decomposition APIs:

- `PCA`
- `SVD`
- `EFA`
- `NMF`
- `MCRALS`

The main conclusion is architectural rather than algorithmic:

```text
analysis outputs do not require a new top-level category yet,
but they do require an explicit internal split
```

Current analysis outputs fall into three families:

| Family | Typical examples | Identity reading | CoordSet / dims reading | Units / provenance reading |
|---|---|---|---|---|
| Latent derived analysis objects | PCA scores, PCA loadings/components, NMF transform outputs, NMF components, EFA concentration-like profiles, MCRALS concentration and spectral profiles | derived analysis object | synthetic `k` axis; surviving source axis partially preserved; source-space support partly reduced away | usually unitless; `name` / `history` synthesized; source lineage often copied selectively |
| Diagnostic / model-summary outputs | explained variance, explained variance ratio, cumulative explained variance, singular values, EFA forward/backward eigenvalue matrices | derived diagnostic object | often component-indexed; not source-space datasets | summarize fitted model structure; provenance often method-centric and rewritten |
| Reconstructed source-space outputs | `PCA.inverse_transform()`, `NMF.inverse_transform()`, conceptually `MCRALS.inverse_transform()` | same scientific object in a modeled / reconstructed representation | source shape, dims, and coordinate metadata restored | source units usually preserved; `name` / `history` remain synthesized around the reconstruction method |

This does **not** require a new top-level taxonomy node.  The existing
identity classes remain sufficient if maintainers distinguish:

- latent derived outputs;
- diagnostic derived outputs;
- reconstructed source-space outputs.

Implementation note:

- many analysis outputs are assembled through
  `_wrap_ndarray_output_to_nddataset()`;
- the wrapper class `_set_output()` currently acts as a semantic assembly
  layer for metadata, coords, names, and provenance;
- this is an observational note about current behavior, not a refactor
  proposal.

Metadata and provenance should be documented conservatively:

- `meta`, `origin`, and `filename` often survive;
- `name` and `history` are usually synthesized;
- `author` and `description` may differ between apparent implementation intent
  and observed runtime behavior;
- provenance is often method-centric and rewritten rather than appended.

`SVD` is a current exception and should not be forced into the same contract as
`PCA` / `NMF`:

- it exposes diagnostic vectors and raw factor arrays;
- it does not currently implement the generic `transform()` reduction API;
- it behaves more like a decomposition-diagnostic surface than a full latent
  representation API.

## Scientific Context vs Structural Information

The current behavior broadly distinguishes two kinds of attached information,
even when the distinction is not enforced centrally.

Scientific context describes interpretation that usually remains meaningful
across single-source transformations:

- `title`
- `description`
- `units`, when the operation does not change physical dimensions;
- custom `meta`, when user supplied and not geometry-specific;

Structural information describes object geometry or derived state whose
validity depends on shape, coordinate support, or model domain:

- `dims`
- `coordset`
- coordinate labels;
- masks;
- `roi`;
- transposition state and other geometry-derived flags.

Of these, `roi` deserves special mention:

- **`roi`** is likely historical UI/interactive selection state, not stable
  scientific metadata. Its propagation through shape operations, reductions, and
  copy-first assembly should be reassessed once its current usage is understood.

  Current observed ROI behaviors include:

  - preserved through shape operations;
  - preserved through indexing and slicing;
  - recomputed by some processing wrappers (Group A).

  No global ROI semantic contract is currently visible.

- **`modeldata`** was derived model or fit information, historically linked to
  fitting workflows.  It has since been **removed from `NDDataset`** as orphaned
  historical infrastructure — see the modeldata RFC (`#1168`).  The descriptions
  below document what characterisation tests observed before removal.

Neither field should be treated as ordinary scientific context in the long term.

Provenance describes lineage and attribution:

- `history`
- `origin`
- `author`
- `created`
- `modified`
- `filename`

Some fields are mixed:

- `units` are scientific context for value-preserving transforms but must be
  recomputed for algebraic arithmetic, integration, and domain transforms.
- `title` is context for ordinary processing but may become a derived output
  name for analysis.
- `meta` may contain either stable user context or geometry-dependent
  application state.
- `author` may describe the source dataset, the result creator, or a combined
  multi-source attribution.
- `origin` may identify scientific source, file lineage, or processing origin.
- coordinate labels are scientifically meaningful, but their validity is
  structural because labels belong to coordinate support.

Current behavior is consistent with this distinction in broad strokes, but not
uniformly. Copy-first result assembly tends to preserve both scientific context
and structural information. Rebuild/wrapper paths tend to preserve only fields
that the operation author remembered to restore. Provenance is usually handled
through explicit history updates or local synthesis, but not by one shared
policy.

## Scientific Object Identity

The codebase appears to contain an implicit concept of scientific object
identity, even though it is not named as a public API or formal internal
abstraction.

The question is:

```text
Does the result still represent the same scientific object, the same object in
a transformed representation, a derived object, or a synthesized multi-source
object?
```

Identity is distinct from provenance:

```text
Identity asks: what scientific object does this result represent?
Provenance asks: where did this result come from?
```

The same operation can therefore change identity while preserving provenance,
or preserve identity while extending provenance.

Observed identity classes:

| Identity class | Meaning | Typical operations | Metadata tendency |
|---|---|---|---|
| Same scientific object | Values changed, but the measured object and support are still the same | scalar arithmetic, many elementwise ufuncs, preserve-geometry processing | preserve title/name/context, append history, preserve CoordSet |
| Same object, changed representation | Same object, but grid, domain, dimensionality, or representation changed | interpolation, reshape-like operations, some integrations and domain transforms | preserve context, recompute structural fields, update history/title as needed |
| Derived scientific object | Output has distinct scientific meaning from the input | latent analysis outputs, scores, components, features, model-derived outputs | synthesize title/description/coords/history, preserve provenance selectively |
| Multi-source synthesized object | Result combines multiple scientifically meaningful inputs | concatenate, stack, multi-input analysis | synthesize provenance, avoid pretending first input identity is the whole result |

PR6 reveals two different identity signals among processing wrappers.

Group A wrappers modify identity markers (name suffixes, rewritten
history), suggesting a derived-object reading.

Group B wrappers preserve identity markers and extend provenance,
suggesting a same-object transformation.
(modeldata has since been removed from the runtime array model.)

It is currently unclear whether this distinction is intentional or
emergent from implementation history.

Operation-family identity reading:

| Operation family | Identity interpretation | Current metadata consistency |
|---|---|---|
| Arithmetic | usually same scientific object | mostly consistent: copy-first preserves context and appends history |
| Ufuncs | usually same object, sometimes changed representation or domain | mostly consistent for dataset-returning ufuncs; raw-return ufuncs intentionally leave object identity |
| Reductions | mixed: same object summarized, or changed representation if dimensions collapse | partially consistent; dims/CoordSet are reduced, but title/name/provenance need characterization |
| Shape operations | same object, changed representation | mostly consistent when CoordSet is transformed with dims; stale structural fields remain a risk |
| Interpolation | same object on a new coordinate grid | conceptually consistent: context survives, CoordSet is rebuilt locally |
| Integration | derived scientific quantity generated through reduction | notably coherent: unit transformation, rewritten title/description/history, CoordSet reduction |
| Indexing / selection | same object with restricted support | consistent: copy-first preserves identity, CoordSet sliced, metadata unchanged |
| Concatenate | multi-source synthesized object | partly consistent: provenance is synthesized, but first-title/name behavior can overstate identity |
| Stack | multi-source synthesized object with new stack axis | partly consistent: source names become labels, other identity fields follow concatenate |
| Processing outputs (Group A: Filter/PCA) | processed or transformed — identity partially changed (name suffix, history rewrite suggest derived identity) | wrapper paths create derived identity while preserving most context; denoise further overrides author |
| Processing outputs (Group B: Baseline) | same object, baseline-corrected — identity preserved | consistent: same-object identity preserved with history append |
| Analysis outputs: latent family | usually derived scientific object | broadly consistent: local synthesis is expected, but provenance preservation varies |
| Analysis outputs: diagnostic family | derived diagnostic or model-summary object | consistent at a high level: component-indexed summaries are assembled locally and are not source-space datasets |
| Analysis outputs: reconstructed family | same scientific object in a modeled / reconstructed representation | geometry and units often return, while `name` / `history` remain method-synthesized |

Identity and provenance combinations:

| Case | Typical operations | Identity | Provenance |
|---|---|---|---|---|
| Identity survives, provenance grows | arithmetic, many ufuncs, baseline-like processing (Group B), indexing/slicing | same object | history appended |
| Identity partially changed, provenance reset | filter/PCA-like processing (Group A: smooth, savgol, whittaker, denoise) | same object with derived-identity name suffix; context otherwise preserved | history rewritten (original lost); denoise also overrides author |
| Identity survives with changed representation | interpolation, reshape-like operations | same object in new representation | lineage preserved and operation recorded |
| Identity changes with reduction-derived quantity | integration | derived quantity from the same source dataset | rewritten history; transformed units; surviving broad context |
| Identity changes, provenance survives | latent analysis outputs, diagnostic outputs, decomposition-derived datasets | derived object | source attribution/history should remain visible |
| Synthesized identity, synthesized provenance | concatenate, stack, multi-source analysis | multi-source result | lineage combined or synthesized |
| Object identity leaves the dataset surface | raw-return logical/testing ufuncs, scalar reductions | no dataset identity | provenance usually unavailable in returned raw value |
| Neither materially changes | pure inspection/display or no-op-like access paths | no new result identity | no new provenance event expected |

The last category is not a major mathematical result family. Most operations
that return a new `NDDataset` either update provenance, change representation,
or both.

Interpolation is currently the clearest example identified during the
characterization campaign of *same scientific object + changed representation*.
Scientific context survives, provenance is extended, while coordinate geometry
is rebuilt.

Field-level identity implications:

| Field | Primary role | If identity is preserved | If identity changes | Current concern |
|---|---|---|---|---|
| `title` | identity / scientific context | usually preserve or wrap | synthesize/override when meaning changes | operation-local wording |
| `name` | identity | may preserve workflow identity | may misidentify derived or multi-source outputs | copy-first can preserve too much |
| `description` | scientific context | should usually survive | may need synthesized description | wrapper/analysis paths vary |
| `author` | provenance, mixed | should usually survive as attribution | should remain provenance, not necessarily authorship of new object | concatenate synthesis is local; denoise overrides to system hostname |
| `origin` | provenance, mixed | should usually survive as lineage | should remain lineage/provenance when meaningful | may be copied without explicit decision |
| custom `meta` | mixed | should not disappear silently | should preserve, namespace, or explicitly drop by rule | copy-first vs wrapper inconsistency |
| `history` | provenance | should append operation | should record derivation/provenance | append/reset/synthesize varies |
| `coordset` | structural information | preserve if support unchanged | reduce/rebuild/synthesize if support changes | mostly coherent but implicit |
| labels | structural / scientific context | preserve with coordinate support | recompute/synthesize with new axes/domains | follows CoordSet but not separately contracted |
| `roi` | structural information | preserve only if still valid on same support | drop/recompute if support/domain changes | high stale-field risk |
| `modeldata` | *(removed from the runtime array model)* | — | — | — |

Identity and CoordSet are related but not identical.

A `CoordSet` can change while scientific object identity remains mostly
preserved. Interpolation is the clearest case: the object can remain the same
measured spectrum or dataset while being represented on a new coordinate grid.
Reshape can also preserve object identity when it is only a representation
change and the coordinate structure remains valid. Integration is more mixed:
collapsing a dimension can be read as summarizing the same object or producing
a derived quantity, depending on operation semantics. Concatenate and stack
generally create synthesized multi-source identity because no single input
fully owns the result.

Identity and result assembly are correlated but not equivalent:

| Result assembly pattern | Identity tendency | Caveat |
|---|---|---|
| Copy-first | often identity-preserving | may accidentally preserve stale structural fields |
| Rebuild / synthesize | often identity-changing or multi-source | can also represent same-object domain changes such as interpolation |
| Reduction-specific | mixed | depends on whether reduction is a summary of the same object or a derived quantity |
| Wrapper-based | depends on operation: Group A (Filter/PCA) tends toward derived identity; Group B (Baseline) preserves identity | processing wrappers should often preserve identity; analysis wrappers may create new identity; current Group A behavior contradicts that expectation |

This suggests that object identity is the conceptual layer connecting metadata,
coordinates, result assembly, and mathematical semantics. The metadata contract
answers what should happen to fields; object identity explains why those field
rules differ by operation category.

Emerging implicit identity contract:

```text
Preserve identity for ordinary value transformations over the same scientific
support.

Preserve identity but recompute structure when representation or coordinate
support changes.

Create or synthesize identity for analysis outputs and multi-source results.

Preserve provenance across identity changes.

Do not let a construction path alone decide object identity.
```

Assessment: **C. Object identity should eventually become an explicit
architectural concept.**

It should not become a new API or implementation plan now. However, it is
already useful enough to name in maintainer-facing architecture work because it
explains otherwise disconnected questions about `title`, `name`, `history`,
`CoordSet`, `roi`, custom metadata, and multi-source provenance.

## Provenance Semantics

Provenance is the lineage of a result. It is related to identity, but not the
same concept.

Observed provenance carriers:

- `history`, which records many operations explicitly;
- `origin`, which often survives by copy as source lineage;
- `author`, which often survives as attribution and may be synthesized for
  multi-source operations;
- `filename`, `created`, and `modified`, which are provenance-like but need
  field-specific policy;
- operation-local descriptions and titles, which sometimes encode derivation
  informally.

Observed provenance rules:

```text
Single-source transformations usually preserve lineage and append or rewrite
history.

Representation changes preserve lineage even when structural information is
recomputed.

Derived analysis outputs may change identity but should still preserve source
attribution in some form.

Analysis-output provenance should be read by family:

- latent outputs usually preserve some copied lineage while synthesizing
  method-local `name` / `history`;
- diagnostic outputs are often even more method-centric and should not be read
  as source-space provenance trails;
- reconstructed outputs return to source space but still carry synthesized
  reconstruction provenance rather than a pure append-only lineage.

Multi-source operations synthesize provenance instead of inheriting one source
unchanged.

Raw scalar or ndarray returns generally leave the dataset provenance surface.
```

Provenance currently survives through a mixture of copy-first behavior, history
updates, and local synthesis. That is useful but uneven. The important
distinction for maintainers is:

```text
Changing scientific object identity does not imply discarding provenance.
Preserving identity does not imply provenance remains unchanged.
```

Examples:

- arithmetic preserves identity while adding operation history;
- interpolation preserves identity in a changed representation while retaining
  lineage;
- latent analysis outputs and diagnostics create a derived identity while still
  needing source provenance;
- reconstructed analysis outputs return to source space without becoming
  latent outputs;
- concatenation and stack create synthesized identity and should synthesize
  lineage from multiple inputs;
- scalar reductions may produce scientifically useful values but leave the
  normal `NDDataset` provenance mechanism.

## Emerging Result Assembly Contract

The following contract is not a new proposal. It is the implicit policy that
appears to emerge from existing behavior and from the related metadata RFC.

```text
Preserve scientific context whenever the result remains the same scientific
object.

Preserve provenance whenever the result remains scientifically traceable to one
or more inputs, even if identity changes.

Recompute, reduce, rebuild, synthesize, or drop structural information when
shape, dims, coordinate support, or scientific domain changes.

Track provenance explicitly through history rather than relying only on copied
identity fields.

Avoid silently discarding user metadata just because a result was assembled by
a wrapper or reconstruction path.

Treat multi-source results as new provenance objects, not as plain copies of
the first input.
```

This contract is currently implicit and unevenly implemented. It is strongest
for arithmetic and many ufuncs, weaker for processing/analysis wrappers, and
most ambiguous for multi-source provenance metadata and structural fields such
as `roi`. (`modeldata` has since been removed from `NDDataset`.)

## Relationship to Metadata and Responsibility RFCs

Result assembly sits between two existing architecture documents.

The related [`../rfcs/metadata-contract.md`](../rfcs/metadata-contract.md)
defines the desired field-level metadata semantics:

```text
Preserve scientific context.
Recompute geometry-dependent metadata.
Never silently drop user metadata.
```

This RFC explains why that contract is difficult to satisfy today: metadata
behavior depends on how a result object is constructed.

Scientific object identity provides the missing rationale behind the metadata
contract categories. If identity is preserved, scientific context should
usually survive. If representation changes, structural information must be
recomputed while provenance remains attached. If a derived or multi-source
identity is created, titles, coordinates, descriptions, and history may need to
be synthesized rather than inherited.

Provenance provides a second rationale that should not be collapsed into
identity. A PCA score dataset, decomposition component, concatenated dataset,
or model output may not be the same scientific object as its source data, but
it remains scientifically traceable to those sources. This is why identity
fields can be synthesized while provenance fields should still preserve or
record lineage.

The related
[`array-class-responsibility.md`](array-class-responsibility.md)
maps where responsibilities currently live. Result assembly cuts across those
responsibilities:

- `NDMath` owns much of the central arithmetic and ufunc assembly behavior.
- `NDDataset` owns dataset-specific scientific context and `CoordSet`.
- `CoordSet` owns coordinate structure but is not part of the class hierarchy.
- processing and analysis code often perform local wrapping or synthesis.
- plugin-backed behavior depends on generic core dispatch and plugin-owned
  semantics.

This is why result assembly emerged from the semantics audit rather than from a
pure class-responsibility review. It is not one class responsibility. It is the
cross-cutting step where numerical results, coordinate geometry, metadata, and
provenance become a user-facing scientific object.

## Should Result Assembly Become Its Own RFC?

Assessment: **B. Yes, but only after mathematical semantics is stabilized.**

Result assembly is clearly significant enough to deserve maintainer attention:

- it is the main source of metadata propagation differences;
- it determines whether `CoordSet`, masks, labels, `roi`, and other fields
  remain valid;
- it explains why processing and analysis outputs differ from arithmetic;
- PR9 shows that analysis outputs themselves split into multiple assembly
  families rather than one uniform semantic class;
- it is where `NDMath`, `NDDataset`, `CoordSet`, processing, analysis, and
  plugin contracts meet.

However, a standalone Result Assembly Contract RFC would be premature before
the characterization tests in this document exist. Without those tests, a
dedicated RFC could accidentally canonize historical accidents or overcorrect
behavior that users currently rely on.

Recommended sequencing:

```text
Mathematical Semantics characterization
    ->
Metadata Contract field mapping
    ->
Result Assembly Contract RFC
    ->
small implementation cleanup, if needed
```

For now, result assembly should remain inside this RFC as a first-class audit
axis. It should become a dedicated RFC once maintainers can distinguish stable
scientific behavior from incidental construction-path behavior.

## Coord Semantics

`Coord` is best understood as an axis/support object.

Current behavior:

- `Coord + scalar`, `Coord - scalar`, `Coord * scalar`, and `Coord / scalar`
  can return `Coord`;
- `Coord + Coord` and related operations can return `Coord` when units are
  compatible;
- `Coord` with `NDDataset` is semantically unsupported;
- several dataset-style operations are explicitly unsupported for `Coord`;
- `Coord` cannot be masked;
- `Coord` is always treated as non-complex.

Interpretation:

`Coord` arithmetic can be meaningful for axis transformations, but `Coord`
should not become a signal-bearing operand by accident.

Operations that appear meaningful:

- shifting an axis by a scalar;
- scaling an axis when the unit semantics are clear;
- subtracting compatible coordinates to produce offsets or deltas.

Operations that are blocked or questionable:

- `Coord` with `NDDataset` arithmetic;
- dataset-style reductions on `Coord`;
- mask operations on `Coord`;
- treating coordinate labels as if they were data samples.

Open questions:

- Should coordinate arithmetic preserve labels?
- Should coordinate scalar multiplication alter units or only values?
- Should `Coord + Coord` remain supported as a general operation?
- Should unsupported `Coord` operations be documented in one explicit contract?

## Label Semantics

There are two label systems:

- direct labels inherited from `NDArray`;
- coordinate labels stored in `Coord` / `CoordSet`.

Current behavior:

- direct labels are mainly useful for 1D labelled arrays and coordinates;
- `NDDataset` disables direct data labels;
- dataset labels usually live inside coordinates;
- slicing selected coordinates preserves labels;
- interpolation may preserve labels only for exact original-coordinate matches
  and leave genuinely resampled points unlabelled;
- concatenation can concatenate coordinate labels along the concatenated
  dimension.

Consistent behaviors:

- selection-like operations are the safest label-preserving operations;
- generated/resampled coordinates are less likely to preserve labels;
- dataset-level label semantics are mostly coordinate-label semantics.

Ambiguous or surprising behaviors:

- direct `NDArray.labels` remains public while `NDDataset` uses coordinate
  labels;
- label propagation is not yet governed by one policy;
- multi-coordinate label behavior is operation-specific.

Relationship to possible `NDLabelled`:

The possible future `NDLabelled` layer belongs to class responsibility work,
not to this RFC. This document only records that labels need a behavior
contract before any extraction is attempted.

## Mask Semantics

Masks are numerical data validity metadata for datasets.

Current behavior:

- arithmetic uses masked data when either operand is masked;
- masks propagate through many arithmetic paths;
- reductions calculate through masked-array behavior;
- slicing slices masks with data;
- concatenation concatenates masks with data;
- interpolation reconstructs masks through numerical float interpolation of
  the source mask followed by 0.5 thresholding — not a copy operation
  (mask handling is closer to a rebuild policy than a preservation policy);
- integration preserves mask information on the returned object, but current
  characterization suggests masked source values are still included in the
  numerical integration itself;
- `Coord` rejects masks and always behaves as unmasked.

Consistent behaviors:

- masks are generally aligned with data shape;
- arithmetic and slicing preserve masks in expected array-like ways;
- concatenation keeps data/mask alignment.

Ambiguous or surprising behaviors:

- integration currently mixes mask preservation with apparent numeric inclusion
  of masked values;
- geometry-changing processing operations can recompute masks locally;
- analysis outputs may not preserve masks in the same way as arithmetic;
- mask semantics for derived quantities are not centrally documented.

Open questions:

- When should masks be preserved versus recomputed?
- Should derived analysis results carry masks from input data?
- Should integration exclude masked values numerically, or is the current
  historical behavior intentional?
- Should interpolation mask behavior be part of a general geometry-changing
  operation contract?

## Unit Semantics

Units are part of the operation contract, not decoration.

Current behavior:

- additive operations require compatible units;
- compatible units may be converted to the first operand's units;
- multiplication/division compute algebraic result units;
- powers and roots can transform units;
- logarithmic/exponential functions require dimensionless input;
- trigonometric functions require angular or dimensionless-compatible input;
- reductions often preserve units;
- integration multiplies data units by coordinate units;
- concatenation requires compatible data units and converts to first units;
- coordinate units are used for coordinate values and location slicing.

Consistent behaviors:

- dimensional analysis is meaningful in `NDMath`;
- incompatible additive units are rejected;
- many ufunc unit requirements are explicit.
- integration is the clearest current unit-transforming reduction family.

Ambiguous or surprising behaviors:

- coordinate units are not full arithmetic alignment keys;
- unitless scalar arithmetic with unitful objects needs explicit policy;
- coordinate scalar arithmetic needs clearer unit semantics;
- processing/analysis functions sometimes define unit behavior locally.

Open questions:

- Should coordinate-unit compatibility be strengthened in dataset arithmetic?
- Should coordinate scalar multiplication/division alter units?
- Should all physical transforms use shared unit-policy helpers?

## Metadata Semantics

Metadata propagation is currently path-dependent.

This section now acts as a bridge between the field-level metadata contract and
the identity/provenance model above. Detailed classification lives in:

- `Scientific Context vs Structural Information`;
- `Scientific Object Identity`;
- `Provenance Semantics`;
- [`../rfcs/metadata-contract.md`](../rfcs/metadata-contract.md).

The current short version is:

| Concern | Current pattern | Risk |
|---|---|---|
| Copy-first result assembly | preserves many fields implicitly | can keep stale structural fields such as `roi` |
| Rebuild/wrapper result assembly | restores selected fields explicitly | can drop context or provenance accidentally |
| Identity fields | `title` and `name` often survive by copy | can misidentify derived or multi-source outputs |
| Provenance fields | `history`, `origin`, and `author` are appended, copied, or synthesized locally | no single provenance policy yet |
| User metadata | custom `meta` often survives copy-first paths | merge/drop behavior is unclear for wrappers and multi-source results |

The normative direction remains:

```text
Preserve scientific context.
Recompute geometry-dependent metadata.
Never silently drop user metadata.
```

Remaining open questions:

- Which copied fields are intentional behavior versus construction-path
  accidents?
- Should identity fields and provenance fields have separate review rules?
- Which multi-source fields should merge, synthesize, namespace, or drop?

## Intentional vs Accidental Behavior

This section classifies observed behavior by confidence.

### Units

| Classification | Observation | Reasoning |
|---|---|---|
| Intentional | additive operations require compatible units | enforced centrally by `NDMath` |
| Intentional | multiplication/division compute algebraic units | core unit machinery is built around this |
| Likely intentional | trigonometric/log/exp unit restrictions | explicit function requirements exist |
| Unclear | coordinate scalar arithmetic preserving units | useful, but the coordinate contract is not yet explicit |
| Unclear | coordinate units not being full alignment keys | consistent with current coordinate arithmetic RFC, but still a policy question |

### Masks

| Classification | Observation | Reasoning |
|---|---|---|
| Intentional | arithmetic uses masked data when operands are masked | central masked-array path |
| Intentional | slicing and concatenation keep masks aligned with data | implemented with data-shape operations |
| Likely intentional | reductions use masked-array reductions | consistent with NumPy masked semantics |
| Unclear | processing/analysis derived masks vary by operation | local implementations define behavior |
| Likely accidental | stale masks after some geometry-changing derived outputs, if any exist | no central mask policy yet |

### Coordinates

| Classification | Observation | Reasoning |
|---|---|---|
| Intentional | dataset-vs-dataset arithmetic checks last dimension | documented current policy |
| Intentional | `Coord` is not a signal operand for `NDDataset` arithmetic | explicit semantic error |
| Likely intentional | shape operations keep data/dims/coordset aligned | recent CoordSet lifecycle work supports this |
| Unclear | earlier dimensions are not validated in arithmetic | accepted current behavior, but may be legacy |
| Unclear | coordinate labels after interpolation | recently improved but still operation-specific |

### Labels

| Classification | Observation | Reasoning |
|---|---|---|
| Intentional | `Coord` can carry labels | core coordinate feature |
| Intentional | `NDDataset` disables direct data labels | `_labels_allowed = False` |
| Likely intentional | selection-like slicing preserves labels | follows coordinate slicing semantics |
| Unclear | labels after coordinate arithmetic | no explicit contract yet |
| Unclear | direct `NDArray.labels` long-term role | responsibility audit leaves this open |

### Metadata

| Classification | Observation | Reasoning |
|---|---|---|
| Intentional | `history` records many operations | explicit assignments throughout operation paths |
| Likely intentional | copy-first paths preserve scientific context | useful behavior, but not always documented |
| Unclear | preserving `name` after derived operations | may help workflows or misidentify results |
| Likely accidental | custom `meta` survives some paths but not wrappers | depends on construction path |
| Likely accidental | `roi` may survive geometry changes by copy | structural fields need recomputation policy |

### History

| Classification | Observation | Reasoning |
|---|---|---|
| Intentional | arithmetic appends operation history | explicit `NDMath` behavior |
| Likely intentional | concatenate resets/synthesizes history | multi-source provenance differs from unary operations |
| Unclear | exact history wording | inconsistent across operation families |
| Likely accidental | history replacement versus append differences | often path-local rather than contract-driven |

## Complex and Hypercomplex Semantics

Ordinary complex data are handled by core `NDComplexArray` plus `NDMath`.

Current ordinary complex behavior:

- `NDDataset` inherits complex support through `NDComplexArray`;
- `real`, `imag`, `absolute`, and conjugation behavior are part of the math
  surface;
- complex display splits real/imaginary values for ordinary complex data;
- some ufuncs promote negative real inputs to complex for mathematically valid
  results.

Plugin-backed hypercomplex behavior:

- non-core numeric execution branches are plugin-provided;
- hypercomplex/quaternion semantics should remain plugin-owned;
- core should provide generic dispatch/fallback, not quaternion-specific
  operation semantics;
- the restored RR/RI/IR/II display behavior is relevant as an example of this
  ownership boundary, but display is not the main topic of this RFC.

Open questions:

- Which core math methods must remain generic for plugin-backed numeric types?
- Which hypercomplex operations need characterization tests before future
  `NDMath` cleanup?
- Should ordinary complex and plugin-backed hypercomplex behavior share a
  documented result contract where possible?

## Current Semantic Model

The current model that emerges is:

```text
Object identity:
    NDDataset is signal/data.
    Coord is axis/support.
    CoordSet is coordinate structure.
    Operations may preserve the same scientific object, change its
    representation, derive a new object, or synthesize a multi-source object.

Stable user context:
    units, title, description, author, origin, and custom meta often survive by
    copy, but this is not always an explicit semantic decision.

Structural information:
    dims, coordset, mask, and roi depend on shape/domain and may
    need recomputation.

Derived information:
    history, title, units, coordinates, and descriptions may be operation
    outputs rather than preserved inputs.

Execution model:
    NumPy-like computation, unit-aware operation rules, mask-aware execution,
    spectroscopy-oriented coordinate checks, plugin dispatch, and copy-heavy
    result assembly.

Result assembly:
    The cross-cutting step where computed values become a scientific object
    again. It determines whether context is preserved, geometry is recomputed,
    provenance is recorded, and stale structural fields are avoided.

Scientific object identity:
    The conceptual reason why some operations preserve context while others
    synthesize titles, coordinates, descriptions, or provenance.

Provenance:
    The lineage of the result. It can be preserved, appended, or synthesized
    independently from whether scientific object identity is preserved.
```

Strong areas:

- core arithmetic and many reductions have a meaningful center in `NDMath`;
- unit handling is substantial and scientifically valuable;
- mask propagation is generally sensible for arithmetic;
- `Coord` vs `NDDataset` arithmetic is now semantically clearer;
- `CoordSet` has a mostly coherent preserve/reduce/rebuild/synthesize model;
- object identity explains much of the processing-vs-analysis distinction;
- provenance explains why derived or multi-source outputs should remain
  traceable even when identity changes;
- CoordSet lifecycle work reduced coordinate propagation risk;
- hypercomplex plugin ownership is clearer after the display restoration work.

Weak areas:

- metadata behavior depends on construction path;
- result assembly is implicit rather than a documented operation contract;
- object identity is implicit and therefore inconsistently reflected in
  `title`, `name`, history, and multi-source provenance;
- provenance is present through history/copy/synthesis paths but not separated
  cleanly from identity in all operation families;
- title/history rules are operation-local;
- coordinate compatibility is intentionally narrower than full alignment;
- direct labels and coordinate labels need clearer contracts;
- processing and analysis functions still contain local semantic policies.

## Ambiguities and Inconsistencies

### High

- Metadata propagation is path-dependent: copy-first operations preserve more
  than wrapper/reconstruction paths.
- Result assembly mixes numerical execution, metadata updates, unit handling,
  mask handling, coordinate propagation, object-type decisions, and plugin
  dispatch.
- Coordinate compatibility in dataset arithmetic validates the spectroscopic
  last dimension but does not provide full labelled alignment.
- Processing and analysis operations synthesize metadata locally, so similar
  operation categories can preserve different fields.
- Structural metadata such as `roi` can survive by copy even
  when geometry-changing operations may make it stale.  (`modeldata` has been
  removed from `NDDataset` — see the modeldata RFC.)
- Plugin-backed operations depend on generic dispatch contracts; weak fallback
  or unclear result assembly semantics could regress optional backends.

### Medium

- Direct labels and coordinate labels are both present but not governed by one
  behavior contract.
- `Coord` arithmetic is partly meaningful and partly blocked, but the boundary
  is not yet documented as a complete contract.
- Unit behavior is strong in core arithmetic but less centralized for
  processing/analysis and coordinate transforms.
- Ufunc return types differ: some operations return `NDDataset`, while
  logical/testing operations return raw arrays.
- Geometry-changing operations differ in how explicitly they handle
  coordinates, labels, `roi`, and history.
- Error ordering in mixed semantic/unit failures may make behavior harder to
  explain even when final rejection is correct.

### Low

- History wording varies across operation families.
- Title naming is locally defined and sometimes surprising.
- Error ordering can expose unit errors before semantic errors in mixed
  `Coord` / `NDDataset` operations.
- Some behavior is likely stable in practice but underdocumented.

## Recommended Characterization Tests

Before behavior changes, add focused characterization tests for current
semantics.

### Priority 1: Must Characterize Before Refactoring

These tests protect behavior that sits directly under future `NDMath`, result
assembly, or metadata work.

- `NDDataset` with scalar: return type, units, mask, title, name, history,
  `description`, `origin`, and custom `meta`. — [x] delivered
- `NDDataset` with `NDDataset`: compatible units, incompatible units,
  coordinate match, coordinate mismatch, mask propagation. — partial coverage
- result assembly field comparison for arithmetic, reduction, shape operation,
  wrapper processing, and analysis-style output using the same seeded context.
- identity-class characterization for representative same-object,
  changed-representation, derived-object, and multi-source operations.
- provenance characterization for identity-preserving, identity-changing,
  multi-source, and raw-return operations.
- `Coord` with `NDDataset`: all operator directions and error types.
- `sum`, `mean`, `std`, `min`, `max` with `dim=None`, named dims, and
  `keepdims=True`;
- slicing with coordinate labels and masks;
- `transpose`, `swapdims`, `squeeze`, `reshape`, `atleast_2d`; — [x] delivered
- propagation of `dims`, `coordset`, labels, masks, title, history, and custom
  metadata. — [x] delivered
- stale-field behavior for `roi` across preserve-geometry and
  modify-geometry operations. — [x] delivered
  (`modeldata` was also characterised but has since been removed from
  `NDDataset`.)
- `title`, `name`, `history`, `origin`, `author`, and custom `meta` behavior
  when an operation preserves identity versus creates a derived identity.
  — [x] delivered
- separation of identity fields (`title`, `name`) from provenance fields
  (`history`, `origin`, `author`) in representative analysis and combination
  outputs. — [x] delivered
- wrapper-based processing metadata preservation versus copy-first arithmetic.
- ordinary complex and plugin-backed hypercomplex operation branches.

### Priority 2: Important but Not Blocking

These tests clarify policy but need not block all maintainability work.

- `Coord` with scalar: value, units, labels, title, metadata.
- `Coord` with `Coord`: compatible/incompatible units and label behavior.
- representative dataset-returning ufuncs: `abs`, `sqrt`, `sin`, `cos`, `tan`;
  — partial coverage
- unit-restricted ufuncs: `log`, `exp`; — [x] delivered
- raw-return ufuncs: `isfinite`, `isnan`, `logical_not`;
- `argmin` / `argmax` return and coordinate semantics;
- cumulative operations and unsupported `Coord` reductions;
- `trapezoid` / `simpson` unit and coordinate behavior.

### Priority 3: Nice to Have

- `concatenate` with compatible and incompatible units;
- `concatenate` with masks and coordinate labels;
- `stack` coordinate labels from dataset names;
- metadata/title/author/history behavior for multi-source results.
- interpolation labels and masks;
- filtering wrapper metadata preservation;
- integration title/description/history/units;
- peak finding coordinate and unit behavior;
- dot-like transformation result assembly.

The purpose of these tests is not to freeze every current quirk permanently.
The purpose is to make intentional changes visible and reviewable.

## Semantic Building Blocks

The PR1–PR8 campaign revealed a small set of recurring semantic patterns.
These patterns are not proposed as a new architecture.  They describe the
reusable building blocks already present in the codebase.

### 1. CoordSet Semantic Categories

CoordSet behavior across operations falls into four categories:

| Category | Meaning | Representative operations |
|----------|---------|--------------------------|
| **Preserve** | CoordSet survives unchanged in shape and content | binary arithmetic, ufuncs, elementwise processing (baseline group) |
| **Reduce** | CoordSet is reduced, dropped, or selected along reduced dimensions | reductions (sum, mean, std, min, max, argmin, argmax), integration |
| **Rebuild** | CoordSet is reconstructed along the affected dimension(s) | interpolation, geometry-changing operations (reshape, squeeze, transpose) |
| **Synthesize** | A new CoordSet is built from multiple sources | concatenate, stack |

These categories correspond to how the coordinate geometry relates to the
result.  Arithmetic preserves geometry.  Reductions collapse it.  Interpolation
replaces it.  Concatenation combines multiple geometries into one.

### 2. Scientific Object Identity Categories

The campaign confirmed four distinct identity classes:

**Same scientific object**
:   The result is the same measured entity with modified values.
    *Arithmetic, ufuncs, baseline processing.*
    Metadata preserved, history appended.

**Same object, changed representation**
:   The entity is unchanged but its coordinate grid or dimensionality is
    different.
    *Interpolation, reshape-like operations.*
    Scientific context survives, provenance extended, geometry rebuilt.

**Derived scientific object**
:   The result has a distinct scientific meaning from the input.
    *Analysis outputs (PCA scores, components), model-derived datasets,
    integration-derived quantities.*
    Title/description/coords/history synthesised locally, provenance
    selectively preserved.

**Multi-source synthesized object**
:   The result combines multiple scientifically meaningful inputs.
    *Concatenate, stack, multi-input analysis.*
    Provenance combined or synthesised; first-input identity can overstate
    the result's nature.

*Interpolation* provides the clearest example of *same object, changed
representation*: scientific context survives, provenance is extended,
while coordinate geometry is rebuilt.

#### Identity vs Provenance

One of the major outcomes of the campaign is that *identity* and *provenance*
are distinct concerns:

- **Identity** answers *what the result is*: same entity, derived entity,
  or synthesised entity.
- **Provenance** answers *where it came from*: history, origin, author,
  filename, creation time.

An operation can preserve identity while extending provenance (arithmetic),
change representation while preserving lineage (interpolation), generate a
derived quantity through reduction (integration), or synthesise both identity
and provenance (concatenate).  Treating them as a single concept obscures the
semantic distinctions that result assembly must handle.

### 3. Result Assembly Categories

Result assembly describes *how* an operation builds its return value —
the mechanical pattern that determines which fields are copied, recomputed,
or dropped.

| Pattern | How the result is built | Representative operations | Metadata tendency |
|---------|------------------------|--------------------------|-------------------|
| **Copy-First Assembly** | Copy the input, then modify data locally | arithmetic, ufuncs | Broad preservation; history appended |
| **Sliced Assembly** | Slice/copy, then assign sliced data into the copy | indexing, selection | Full preservation; history appended |
| **Reduction Assembly** | Build result from reduced data, attach surviving context | reductions, integration | Partially preserved; operation-specific, with integration as the strongest derived-quantity case |
| **Domain-Rebuild Assembly** | Copy, then rebuild coordinate domain and interpolate/recompute data | interpolation, reshape-like ops | Scientific context preserved; geometry rebuilt |
| **Wrapper Assembly** | Call an underlying library, then wrap the result back into NDDataset | processing wrappers (smooth, savgol, etc.) | Two sub-patterns: Group A rewrites identity markers; Group B appends history |
| **Synthesize Assembly** | Combine multiple inputs into a new result | concatenate, stack | Multi-source metadata; provenance synthesised |

The distinction between Group A and Group B wrappers is a notable divergence
within the same assembly pattern:

- Group A (Filter/PCA-based: smooth, savgol, whittaker, denoise): name
  overwritten, roi recomputed, history rewritten.
- Group B (Baseline-based: basc, detrend, asls): name preserved,
  roi preserved, history appended.
  (`modeldata` was also dropped/preserved respectively but has since been
  removed from `NDDataset`.)

This divergence is likely emergent from implementation history rather than
intentional design.

### 4. Provenance Categories

Provenance refers to the record of how a dataset was produced.  Observed
patterns:

| Pattern | Meaning | Examples |
|---------|---------|----------|
| **Preserve** | The existing provenance trail survives unchanged | arithmetic, indexing (history appended but original entries kept) |
| **Extend** | A new provenance entry is added to the existing trail | interpolation, baseline processing (Group B), slicing |
| **Rewrite** | The existing trail is replaced by a single new entry | filter/PCA processing (Group A), some analysis outputs |
| **Synthesise** | Provenance is built from multiple sources | concatenate, stack |

History is the most explicit provenance mechanism.  The append-vs-rewrite
distinction is the most practically significant: it determines whether the
original processing trail survives in the result.

### 5. Structural Information Categories

CoordSet, labels, and masks are active structural information.
ROI and modeldata were historical structural-like fields removed after audit.

CoordSet, labels, and masks share a common property: their validity depends on
geometry, domain, shape, or representation.  They are not ordinary scientific
context.

| Field | Dependency | Behaviour after geometry change |
|-------|-----------|--------------------------------|
| **CoordSet** | Coordinate geometry | Rebuilt or synthesised |
| **Labels** | Coordinate value equality | Carried over on exact match only |
| **Masks** | Data shape and position | Reconstructed via float interpolation + threshold; not copied |
| **ROI** | Historical data-domain range state | Removed from the runtime array model after audit |
| **modeldata** | Historical derived model state | Removed from the runtime array model after audit |

These fields differ from title, name, author, description, origin, and meta,
which survive geometry changes unconditionally.

The stale-field risk for ROI was the most actionable finding in the campaign.
That issue is now resolved by removal from the runtime dataset model rather
than by introducing a new propagation contract.  The characterization remains
useful as historical evidence of why removal was chosen.

### 6. Campaign Conclusions

The PR1–PR8 characterisation campaign clarified the following:

- **CoordSet semantics** are now reasonably understood across four categories
  (preserve, reduce, rebuild, synthesise).
- **Identity and provenance** emerged as distinct concepts during the campaign.
  Identity answers *what* the result is; provenance answers *where* it came
  from.
- **Result assembly** is a first-class architectural concern, not a detail of
  `NDMath` internals.  The assembly pattern determines which fields survive
  and how.
- **Integration** is now the clearest characterized example of a derived
  scientific quantity produced through reduction: geometry is reduced, units
  transform, and identity/provenance markers are rewritten locally.
- **ROI** was the least well-defined runtime structure identified by the
  campaign.  It was preserved verbatim by most operations, including those
  that changed the coordinate grid, which made it frequently stale.  That
  ambiguity has now been resolved by removing ROI from `NDDataset` rather than
  defining a new propagation policy.
- **Processing wrappers** expose an implementation-family semantic divergence:
  Group A rewrites identity markers while Group B extends provenance.  Whether
  this is intentional is unknown.
- **Interpolation** provides the clearest example of *same object, changed
  representation* — the pattern that the campaign set out to identify.

The characterization campaign itself changed no behavior.  Follow-up maintainer
decisions later removed both `modeldata` and `roi` from the runtime dataset
model and kept only load-compatibility handling for legacy serialized fields.

### Needs Decision Now

These decisions are blockers for any result assembly or `NDMath` refactor.

- Metadata preservation baseline: which fields are preserved by default for
  copy-first single-source operations?
- Structural metadata policy: when should coordinate structures be recomputed
  or dropped?  `roi` and `modeldata` have now both been removed from
  `NDDataset`, resolving those specific stale-field concerns by removal rather
  than by propagation policy.
- Wrapper parity: should wrapper-based processing preserve the same scientific
  context as copy-first operations?
- Result assembly scope: should maintainers treat result assembly as a
  behavior contract before extracting or simplifying `NDMath` internals?
- Object identity scope: should same-object, changed-representation,
  derived-object, and multi-source operations become explicit maintainer
  categories for metadata/result-assembly review?
- Provenance scope: should history, origin, author, filename, created, and
  modified be reviewed separately from object identity fields?
- Coord-vs-Dataset arithmetic: should the current semantic rejection remain the
  long-term contract, and should its error precedence be standardized?

### Needs Decision Later

These decisions can wait until characterization tests expose the tradeoffs more
clearly.

- Whether last-dimension coordinate validation remains final policy or becomes
  a legacy baseline.
- Whether `Coord` arithmetic should remain broadly supported or be narrowed.
- How direct `NDArray.labels` and coordinate labels should evolve.
- Whether logical/testing ufuncs should continue returning raw arrays.
- How multi-source custom `meta` should merge, drop, or be namespaced.
- Whether plugin-backed hypercomplex operations should share a formal result
  contract with ordinary complex operations.
- Whether a standalone Result Assembly Contract RFC should be opened after the
  mathematical semantics characterization tests exist.
- Whether a future Object Identity note should be part of that Result Assembly
  Contract or remain a section of this Mathematical Semantics RFC.
- Whether provenance should remain embedded in metadata/result assembly
  decisions or receive a small explicit subsection in a future contract.

### Can Remain Implementation-Defined For Now

These areas should be documented but do not require immediate maintainer
decisions.

- Exact history message wording.
- Exact title formatting for common ufuncs.
- Internal helper boundaries inside `NDMath`.
- The order of low-level validation checks when multiple errors could apply,
  except for user-facing semantic clarity cases such as Coord-vs-Dataset.

## Future RFC Topics

Topics needing maintainer decisions:

- Coord arithmetic contract;
- direct labels vs coordinate labels;
- metadata propagation by operation category;
- unit behavior policy for coordinate transforms and physical operations;
- mask policy for geometry-changing operations;
- coordinate compatibility policy for dataset arithmetic;
- Result Assembly Contract, after mathematical semantics is stabilized;
- scientific object identity categories for operation review;
- provenance categories for identity-changing and multi-source operations;
- `NDMath` result assembly responsibilities;
- NumPy-facing API support boundaries;
- processing/analysis semantic normalization;
- plugin-backed numeric backend contracts;
- future `NDLabelled` extraction after behavior is characterized.

## Recommendation

Do not refactor `NDMath` yet.

Recommended next step:

1. Add focused characterization tests for the behavior categories listed above.
2. Use those tests to separate intended semantics from historical accidents.
3. Convert the stable behavior into explicit maintainer contracts.
4. Keep result assembly in this RFC for now, then consider a dedicated Result
   Assembly Contract RFC once the behavior is characterized.
5. Treat object identity as part of that future contract discussion, not as a
   new implementation concept today.
6. Treat provenance as distinct from identity in that discussion: identity
   answers what the result is, provenance answers where it came from.
7. Only then consider small helper extractions, result assembly cleanup, or
   staged `NDLabelled` planning.

The conceptual stack now appears to be:

```text
Mathematical semantics
    ->
Result assembly
    ->
Scientific object identity
    +
Provenance
```

Provenance is not simply below identity in a strict hierarchy. It is an
orthogonal concern that result assembly must preserve, append, or synthesize
whether identity is preserved or changed.

The near-term architecture priority is not a class hierarchy redesign. It is a
clear operation semantics contract that separates identity, provenance,
scientific context, and structural validity.
