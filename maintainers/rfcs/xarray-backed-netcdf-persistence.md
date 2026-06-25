# xarray-backed NetCDF Persistence

**Status:** Proposed Maintainer RFC

**Scope:** NetCDF persistence rules for the already-defined
`NDDataset ↔ xarray.Dataset` mapping.

**Out of scope:** New object-model design, `Project` persistence, `Result`
persistence, Zarr-specific policy, implementation, migration, or API work.

**Related documents:**

- `maintainers/rfcs/nddataset-xarray-mapping-specification.md`
- `maintainers/rfcs/trusted-and-portable-persistence.md`
- the earlier xarray / NetCDF prototype and architecture reviews

The key words MUST, SHOULD, and MAY express the intended future contract. They
do not change current behavior until separately implemented.

## Relation to the xarray RFC

The canonical portable carrier for `NDDataset` is `xarray.Dataset`.

NetCDF is a persistence backend for that carrier, not a replacement object
model. The persistence chain is therefore:

```text
NDDataset
    ↔ xarray.Dataset
    ↔ NetCDF
```

and not:

```text
NDDataset ↔ NetCDF
```

This distinction is mandatory because:

- xarray remains the canonical portable in-memory model;
- NetCDF backend constraints must not redefine SpectroChemPy semantics;
- backend-specific compromises belong at the persistence layer, not at the
  carrier-model layer.

## Goals

This RFC defines a NetCDF persistence profile for the SpectroChemPy xarray
mapping with the following goals:

- portable scientific persistence;
- archival-friendly storage of `NDDataset` content;
- interoperability with xarray and standard NetCDF tooling;
- compatibility with broader PyData workflows, including later Dask/Zarr usage.

## Non-goals

This RFC does not define:

- a new canonical representation for `NDDataset`;
- `Project` portable persistence;
- `ResultBase`, `AnalysisResult`, or `FitResult` persistence;
- full `CoordSet` fidelity beyond what the xarray RFC already allows;
- backend-specific chunking or compression policy;
- a provenance graph model;
- HyperComplex persistence guarantees.

## Primary variables

Each persisted NetCDF file MUST represent one logical SpectroChemPy primary
signal derived from the canonical xarray primary variable.

Required conventions:

- the Dataset MUST carry `scpy_primary_variable`;
- the referenced primary variable MUST exist;
- the primary variable MUST use the canonical dimension order already defined by
  the xarray RFC.

Recommended conventions:

- `scpy_format = "nddataset-xarray"`
- `scpy_version = 1`

NetCDF readers that do not understand `scpy_*` attrs may still read the file as
a normal xarray/NetCDF dataset, but SpectroChemPy reconstruction MUST rely on
these attrs when present.

## Coordinates

### Default coordinates

Default per-dimension coordinates MUST be written as NetCDF coordinate
variables.

Their persistence SHOULD preserve:

- coordinate values;
- dimension attachment;
- `units`;
- `scpy_title` when present;
- relevant `scpy_*` attrs required for reconstruction.

### Auxiliary coordinates

Auxiliary same-dimension coordinates SHOULD be written as non-dimension
coordinate variables when the xarray carrier includes them.

This is a best-effort portability layer:

- SpectroChemPy SHOULD preserve them when possible;
- third-party tools MAY ignore or drop some non-standard attrs;
- full `CoordSet` topology is not guaranteed by NetCDF itself.

### Guarantees

Guaranteed:

- default coordinates required for core scientific interpretation.

Best effort:

- auxiliary same-dimension coordinates;
- richer `CoordSet` metadata carried only in `scpy_*` attrs.

## Masks

### Options considered

Option A: explicit boolean mask variable

Option B: `NaN` / `_FillValue` only

Option C: hybrid approach

### Decision

The recommended NetCDF representation is Option C: hybrid, with the explicit
boolean mask variable as the canonical reconstruction source.

Concretely:

- SpectroChemPy SHOULD persist a dedicated boolean mask variable aligned with
  the primary variable dimensions;
- the Dataset SHOULD carry `scpy_mask_variable` naming that variable;
- `_FillValue` or `NaN` MAY also be emitted when useful for external-tool
  compatibility, but they are not the canonical source of truth.

### Rationale

Boolean masks are the most faithful reconstruction mechanism because:

- they work for integer, floating, and complex data;
- they preserve masked-data semantics without conflating them with ordinary
  missing-value conventions;
- they survive round-trip reconstruction more explicitly than `NaN` alone.

### Reconstruction

`NetCDF -> xarray -> NDDataset` reconstruction MUST prefer the explicit mask
variable when present.

If the explicit mask variable is absent, `_FillValue` or `NaN` MAY be used as a
fallback hint, but this fallback is best effort only.

## Complex data

### xarray versus NetCDF

The xarray RFC already decided that the canonical xarray carrier preserves
native NumPy complex dtype.

NetCDF persistence introduces a separate backend constraint:

- canonical xarray representation: native complex dtype;
- canonical NetCDF representation: split real/imag convention.

### Options considered

Option A: split real/imag variables

Option B: backend-specific opaque encodings

Option C: alternative non-standard conventions

### Decision

Option A is retained.

SpectroChemPy SHOULD persist complex-valued primary data as two aligned real
variables and MUST store the reconstruction metadata needed to rebuild the
logical complex signal.

Recommended attrs:

- `scpy_complex_representation = "split-real-imag"`
- `scpy_complex_real = <real variable name>`
- `scpy_complex_imag = <imag variable name>`
- `scpy_primary_variable = <logical primary signal name>`

Recommended variable naming:

- `<primary>__real`
- `<primary>__imag`

### Reconstruction

`NetCDF -> xarray -> NDDataset` reconstruction MUST:

- detect the split-real-imag convention;
- rebuild a native NumPy complex array in the xarray carrier;
- continue the normal xarray-based reconstruction path into `NDDataset`.

Opaque backend-specific encodings MUST NOT be the canonical SpectroChemPy
NetCDF contract.

## Units

Data and coordinate units MUST be persisted as textual `units` attrs.

This RFC recommends:

- prefer CF-compatible unit strings when feasible;
- preserve SpectroChemPy unit strings as written when exact CF normalization is
  not available or would change meaning.

Guaranteed:

- textual unit preservation when the NetCDF/xarray toolchain preserves attrs.

Best effort:

- interpretation by third-party readers that apply their own unit semantics;
- automatic reconstruction of a particular runtime unit engine outside
  SpectroChemPy.

Portable persistence is about stable scientific meaning, not about forcing
external readers to understand every SpectroChemPy unit expression identically.

## Metadata

NetCDF persistence SHOULD export portable scientific metadata, not arbitrary
Python runtime state.

### Export policy

Portable by default:

- JSON-compatible metadata only;
- `name`, `title`, `description`, `author`, `origin` when available;
- `history` as textual metadata;
- required `scpy_*` attrs.

Not portable by default:

- arbitrary Python objects;
- callables;
- implementation-private runtime helpers;
- non-JSON-compatible metadata values.

### Namespace policy

SpectroChemPy-specific persistence attrs SHOULD use the `scpy_*` namespace.

Typical examples:

- `scpy_format`
- `scpy_version`
- `scpy_primary_variable`
- `scpy_mask_variable`
- `scpy_complex_representation`
- `scpy_complex_real`
- `scpy_complex_imag`

### History

`history` SHOULD be written as textual metadata, not as a new provenance graph.

This RFC does not extend NetCDF persistence into workflow replay or structured
provenance.

## Round-trip guarantees

The relevant persistence path is:

```text
NDDataset
    -> xarray
    -> NetCDF
    -> xarray
    -> NDDataset
```

### Guaranteed

- primary numerical signal values, except for backend-imposed numeric precision
  changes;
- dimension names and order;
- default coordinate values;
- unit strings for the primary signal and default coordinates;
- explicit boolean mask reconstruction when the canonical mask variable is
  preserved;
- reconstruction of complex data through the split real/imag NetCDF convention;
- `scpy_*` attrs required by the persistence profile.

### Best effort

- auxiliary same-dimension coordinates;
- JSON-compatible metadata not required for core reconstruction;
- textual `history` preservation through external tools;
- compatibility through toolchains that rewrite attrs or variable encodings.

### Not guaranteed

- full `CoordSet` topology;
- aliases and reference identity;
- non-JSON-compatible metadata;
- HyperComplex dtype semantics;
- byte-for-byte file identity;
- identical behavior across all third-party NetCDF readers;
- preservation of NetCDF backend-specific encoding choices after arbitrary
  third-party rewrites.

## External tool compatibility

This profile is designed to remain usable outside SpectroChemPy.

### xarray

xarray compatibility is the primary target. SpectroChemPy-specific attrs SHOULD
be additive, not disruptive.

### Standard NetCDF readers

Readers that understand dimensions, variables, coordinates, and attrs SHOULD be
able to inspect the scientific content even if they ignore `scpy_*` attrs.

### Dask

Nothing in this RFC should prevent later lazy-loading or chunked workflows.
However, chunking policy is not defined here.

### Zarr conversion

Because the canonical carrier remains xarray, later NetCDF ↔ Zarr conversion
through xarray remains a valid downstream path. This RFC does not define Zarr
policy, but it should not block it.

### Third-party tools

Third-party tools may:

- ignore `scpy_*` attrs;
- preserve only standard attrs;
- flatten or rewrite some encodings.

That is acceptable. SpectroChemPy portability guarantees apply to the defined
profile, not to arbitrary third-party rewrites.

## Recommendations

This RFC recommends the following persistence profile:

1. Treat xarray as the canonical portable carrier and NetCDF as its backend.
2. Persist default coordinates as standard NetCDF coordinate variables.
3. Use a hybrid mask strategy, with an explicit boolean mask variable as the
   canonical reconstruction source.
4. Persist complex data in NetCDF via a split real/imag convention, while
   preserving native complex dtype in the xarray carrier.
5. Persist units as textual `units` attrs.
6. Restrict portable metadata to JSON-compatible values plus required
   scientific and `scpy_*` attrs.
7. Keep round-trip guarantees explicit and conservative.

## Open questions

- Should `_FillValue` emission be mandatory whenever an explicit mask variable
  is present, or only recommended?
- Should auxiliary coordinates be written as coordinates only, or may some of
  them be downgraded to ordinary variables for tool compatibility?
- Should future work define a stricter CF-aligned subset of allowed unit
  strings?
- Should later RFC work define a separate NetCDF profile for `Project`, or keep
  NetCDF persistence dataset-only?

Recommended next step: prototype NetCDF write/read on top of the validated
xarray carrier, without expanding object-model scope.
