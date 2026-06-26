[Maintainers](../../README.md) · [RFCs](../INDEX.md)

# Coord Labels Portable Semantics

**Status:** Proposed Maintainer RFC

**Scope:** Semantic classification of `Coord.labels` for portable persistence
through the existing `NDDataset ↔ xarray.Dataset ↔ NetCDF` chain.

**Out of scope:** Code, PR, API changes, aliases, references, `Coord.meta`,
numeric same-dimension coordinates (already handled), `Project` persistence,
`Result` persistence, or backward-incompatible changes to the current label
model.

**Related documents:**

- `maintainers/rfcs/nddataset-xarray-mapping-specification.md`
- `maintainers/rfcs/xarray-backed-netcdf-persistence.md`
- the recent portable `CoordSet` enrichment work and same-dimension coordinate
  progress reviews

---

## Motivation

`Coord.labels` is the last major `CoordSet` feature without a portable
persistence story. Numeric default coordinates and numeric auxiliary same-
dimension coordinates are now covered. Labels are different: they are
semantically heterogeneous, dtype-flexible, often multi-row, and used in ways
that blur the line between coordinate values, annotations, metadata, and plot
decorations.

A clear portable semantics decision is needed before implementing, because:

- the current serialization silently drops labels — no warning, no error;
- several readers (OPUS, JCAMP, Omnic, SPC, CSV) populate labels with
  acquisition metadata that users may expect to survive round-trip;
- the xarray mapping RFC explicitly lists labels as an open question;
- earlier portable-persistence review recommended deferring labels to a
  dedicated RFC.

This RFC resolves that open question by classifying label usages and defining
which subset is portable.

---

## Current label model

### Inheritance

- `NDArray` defines `labels` as a trait (property + setter, `ndarray.py:1553`).
- `Coord` inherits directly (always 1D, labels sliced alongside data on `__getitem__`).
- `CoordSet.labels` is a read-only list delegating to each stored `Coord`.
- `NDDataset` has `_labels_allowed = False` — direct labels are disabled.
  Dataset-level labels live only inside the coordinates of `CoordSet`.

### Storage

`Coord.labels` is stored as a NumPy array of dtype `object`. Its shape is:

- `(N,)` for single-row labels — one label per coordinate point;
- `(M, N)` for multi-row labels — M rows of N labels.

There is no restriction on element types. Observed values include strings,
datetimes, `None`, numeric strings, and filenames.

### Construction

Labels are passed as a `labels=` keyword argument to the `Coord` constructor
or set later via the property setter.

### Display

Labels are displayed in `Coord.__repr__` via `str(self.labels.T).strip()` and
used as tick labels in 1D and 2D plots (`plot1d.py`, `plot2d.py`).

### Serialization (current)

- **xarray/NetCDF**: silently dropped — `to_xarray()` does not export labels;
  `from_xarray()` does not reconstruct them.
- **JSON utils**: special-cased — object-dtype label arrays are converted via
  `.tolist()` before encoding (`jsonutils.py:330`).
- **SCP/PSCP native**: labels are preserved through native serialization
  (trusted path, not portable).

---

## Observed usages

### Category 1: Sample / acquisition identification

Used as point annotations on the y-dimension (or x) to identify individual
acquisitions. Common patterns:

- acquisition dates and times (OPUS, JCAMP)
- filenames (OPUS, Omnic)
- sample names (CSV, download examples)
- channel names (Quadera)
- compound / species names (kinetic utilities, PLS)

These are **scientific metadata** — they identify which physical sample or
acquisition produced a given data point.

### Category 2: Plot labels (synthetic)

Synthetic labels generated for display:

- `f"#{i}"` iteration labels (`decorators.py`)
- auto-generated labels for plot legends

These are **display conveniences** — they have no scientific meaning outside
the session.

### Category 3: Multi-field annotations

Multi-row labels where each row carries a different kind of information:

- (acquisition date, title, filename) — OPUS reader
- (datetime, species name, replicate) — kinetic experiments

These mix semantics within a single `Coord.labels` array.

### Category 4: Target / reference names

Used in analysis contexts:

- PLS target variable names (`pls.py`)
- Component names in kinetic modeling

These are **structural model metadata** — they identify which output variable
or component a coordinate value corresponds to.

### Category 5: Coordinate-like numeric labels

Rare: labels used as quasi-coordinate values (e.g., channel indices stored as
labels rather than as coordinate data). Not recommended but present in legacy
code.

---

## Label categories

| Category | Scientific value | Structural role | Portability priority |
|---|---|---|---|
| 1. Acquisition identifiers | High | Annotation | Should be portable |
| 2. Synthetic plot labels | None | Display | Not portable |
| 3. Multi-field annotations | Medium | Annotation | Conditional |
| 4. Target / reference names | High | Model metadata | Should be portable |
| 5. Coordinate-like values | Medium | Structural (misplaced) | Out of scope |

---

## Portable semantics

### Guaranteed

A label is portable if it satisfies **all** of:

1. **1D string-only**: labels are a single row of strings (dtype `object`
   containing only `str` elements), one per coordinate point.
2. **Same length as the owning coordinate**: `len(labels) == len(coord)`.
3. **No mixed types**: all elements are `str` (or `None` representing missing).

This subset corresponds to the most common portable case: acquisition
identifiers, sample names, and target names that are already strings.

### Warning on non-exportable labels

Labels that exist but are not part of the guaranteed portable subset MUST NOT
disappear silently. When export encounters non-exportable labels, an explicit
warning MUST be emitted:

```text
PortableWarning:
Coord labels were not exported because they are not part of the
supported portable label subset.
```

The exact mechanism (Python `warnings.warn` vs structured logging) is an
implementation detail, but the requirement is behavioural: the user must be
notified that label information was dropped.

### Best effort

Multi-row string labels where all elements are strings and each row has the
same length. These could be represented as multiple 1D string coordinates in
xarray, each with a deterministic naming convention. However, the
interpretation of each row's semantics is lost without additional metadata.

### Not portable

- Object-dtype arrays containing non-string elements (numeric, mixed).
- Multi-row labels where rows have different semantic roles.
- Labels longer than the owning coordinate.
- Synthetic display-only labels (`f"#{i}"`).
- Labels used as structural coordinate values.

### Deferred

- **Datetime labels** — previously classified as not portable. Reclassified as
  deferred because datetime-valued labels (acquisition dates, timestamps) carry
  genuine scientific metadata and are common in reader output. A future phase
  could map datetime ↔ ISO 8601 string at export time if a real need is
  identified. For now, datetime labels trigger the non-exportable warning.

---

## xarray mapping options

### Option A: Labels as 1D string coordinate variable

Export single-row string labels as a non-dimension coordinate variable on the
owning dimension:

```python
xds.coords["{dim}_labels"] = xr.DataArray(
    labels_array,  # string array
    dims=(dim,),
    attrs={"scpy_coord_role": "label", "scpy_owner_dim": dim},
)
```

**Pros:** natural xarray representation; survives NetCDF as a string variable;
visible to external tools; reconstructible.

**Cons:** only works for 1D string labels; multi-row needs additional
convention; NetCDF string variable length limits may apply.

### Option B: Labels as JSON-encoded attr

Store labels as a JSON string in the coordinate's attrs:

```python
coord_attrs["scpy_labels"] = json.dumps(labels.tolist())
```

**Pros:** handles multi-row and mixed types; no new variable in xarray; no
NetCDF dtype issues; works with any shape.

**Cons:** invisible to external tools; opaque; no xarray indexing; fragile for
large label arrays.

### Option C: Labels as a separate data variable

Store labels as a dedicated data variable in the xarray Dataset:

```python
xds["{dim}_labels"] = xr.DataArray(
    labels_array,
    dims=(dim,) if labels.ndim == 1 else (dim, "label_row"),
    attrs={"scpy_role": "label", "scpy_owner_dim": dim},
)
```

**Pros:** handles multi-row cleanly with an extra dimension; visible to
external tools; reconstructible.

**Cons:** introduces a synthetic dimension for multi-row labels; not a
coordinate (does not participate in xarray indexing); more complex
reconstruction contract.

### Option D: No export (status quo)

Do not export labels. Document that labels are not portable and recommend
native SCP/PSCP for label preservation.

**Pros:** zero implementation cost; no backward compatibility risk; no xarray
mapping ambiguity.

**Cons:** silently drops user data; several readers populate labels with
scientifically meaningful metadata; users lose information on xarray/NetCDF
round-trip.

### Comparison

| Criterion | A (string coord) | B (JSON attr) | C (separate var) | D (none) |
|---|---|---|---|---|
| Scientific visibility | High | Low | High | None |
| External tool access | Yes | No | Yes | No |
| Multi-row support | No | Yes | Yes | N/A |
| Non-string support | No | Yes | No | N/A |
| Implementation cost | Medium | Low | Medium | Zero |
| Backward compat risk | Low | Low | Low | None |
| Reconstruction | Trivial | Trivial | Medium | N/A |

---

## NetCDF constraints

### String arrays

NetCDF3 (scipy engine) supports fixed-length strings. NetCDF4 supports
variable-length strings. Both can represent 1D string coordinate variables.

Practical limitations:

- very long strings may be truncated by the scipy engine;
- NetCDF3 does not support variable-length UTF-8 natively;
- Python object dtype is not supported — arrays must be explicitly converted to
  string type before writing.

### Multi-dimensional string arrays

xarray can represent multi-dimensional string arrays, but each string dimension
must be an explicit xarray dimension. For multi-row labels, this means:

```python
dims = (dim, f"{dim}_label_row")
```

This introduces a synthetic dimension that has no scientific meaning outside
SpectroChemPy.

### Object dtype

Object arrays cannot be written directly to NetCDF. They must be converted:

- strings → fixed or variable-length string arrays (xarray handles this);
- datetimes → numeric or string representation;
- mixed types → not representable without loss.

### scipy engine compatibility

The current NetCDF prototype defaults to the `scipy` engine (NetCDF3).
Variable-length strings and multi-dimensional string variables are possible
but more fragile than numeric arrays.

---

## Round-trip contract

### `NDDataset → xarray → NDDataset` (string labels)

```text
GUARANTEED:
    labels = ["a", "b", "c"] (1D, all str, len == N)
    → survives as string coordinate variable
    → reconstructed as np.array(["a", "b", "c"], dtype=object)
```

```text
BEST EFFORT:
    labels = [["a1", "b1"], ["a2", "b2"]] (2D, all str)
    → may survive as multi-dim string variable
    → reconstruction depends on explicit dimension handling
```

```text
NOT GUARANTEED:
    labels = ["a", 1, None, "b"]  # mixed types
    labels = ["a", "b"]  # len != N
```

```text
DEFERRED:
    labels = [datetime(2024, 1, 1), datetime(2024, 1, 2)]
    → not portable today; potential ISO 8601 string representation
```

### `NDDataset → NetCDF → NDDataset` (string labels)

Same contract as xarray, plus:

```text
LIMITATIONS:
    - scipy engine may truncate very long strings
    - multi-row labels require synthetic dimensions
    - non-string labels must be converted or rejected
```

---

## Recommendation

### Phase 2a: String-only 1D labels (Option A — recommended)

**Scope:**

- Export single-row string-only labels as 1D non-dimension coordinate
  variables with `scpy_coord_role = "label"` and `scpy_owner_dim = <dim>`.
- Import: detect `scpy_coord_role == "label"`, reconstruct labels as
  `np.array(..., dtype=object)`.
- Multi-row labels, non-string labels, datetime labels, and mixed types are
  **not exported** and MUST trigger a warning (e.g. `PortableWarning`).
- Labels must NOT disappear silently.

**Rationale:**

- Catches the most common portable case (acquisition identifiers, sample names).
- Maps cleanly to xarray/NetCDF string variables.
- External tools can read the labels.
- Low implementation risk.
- Clear migration path for Phase 2b.

**Naming convention:**

```text
{dim}_labels
e.g., "y_labels", "x_labels"
```

**Export only when** `labels.ndim == 1` and all elements are `str` or `None`.

### Phase 2b: Multi-row metadata encoding (Option B as fallback)

If multi-row string labels are common enough to justify the complexity:

- Export the primary label row as a 1D string coordinate (Phase 2a);
- Encode remaining rows as a JSON string in the coordinate attrs under
  `scpy_labels_extra`;
- Document that extra rows are best-effort metadata, not portable structural
  data.

**Rationale:** avoids the synthetic-dimension problem while preserving
ancillary information for SpectroChemPy round-trip.

### Deferred

- **Datetime labels** — deferred rather than excluded. Datetime-valued labels
  (acquisition dates, timestamps) carry scientific metadata and are common in
  reader output. A future phase could map `datetime ↔ ISO 8601 string` at
  export time if a real need is identified.
- Non-string labels (numeric, mixed).
- Multi-row labels as multi-dimensional coordinate variables (Option C).
- Label-based indexing guarantees through the portable path.

### Excluded

- Synthetic display labels (`f"#{i}"`).
- Labels used as coordinate values.
- Full backward compatibility with legacy label serialization edge cases.

---

## Open questions

1. Should a `scpy_label_rows` attr be emitted to record the original number of
   rows for multi-row labels that get partially exported?

2. Should the warning mechanism use Python `warnings.warn` with a dedicated
   `PortableWarning` subclass, or use the existing `warning_()` utility?

3. Should `from_xarray()` accept external xarray string coordinate variables
   (without `scpy_coord_role`) as label candidates, or require explicit markers?

4. Should the Phase 2a naming convention reserve `{dim}_labels` so that a
   future Phase 2b does not need to rename existing exports?

---

## Conclusion

| Question | Answer |
|---|---|---|
| Support labels now? | Yes, for 1D string-only labels (Phase 2a). |
| Scope? | Single-row `str` elements only. Multi-row and datetime deferred. |
| Representation? | 1D non-dimension coordinate variable with `scpy_coord_role`. |
| Limits? | No non-string, no multi-row as variables, no synthetic display labels. |
| Warning? | Yes — non-exportable labels trigger an explicit warning. |

**Recommended next step:** implement Phase 2a as a short follow-up PR after
the same-dimension numeric coordinate PR.
