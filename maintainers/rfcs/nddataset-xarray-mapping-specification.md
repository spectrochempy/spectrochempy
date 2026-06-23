# NDDataset ↔ xarray Dataset Mapping Specification

**Status:** Proposed Maintainer RFC

**Scope:** Canonical conceptual mapping between `NDDataset` and `xarray` for
future portable persistence and interchange work.

**Out of scope:** Code, API changes, SCP/PSCP changes, `Project` persistence,
result persistence, plugin persistence, backend selection details, or migration
mechanics.

**Related documents:**

- `maintainers/rfcs/scientific-object-model-and-persistence-boundaries.md`
- `maintainers/rfcs/trusted-and-portable-persistence.md`
- the recent portable xarray / NetCDF architecture review
- the recent CSDM native-persistence candidate review

The key words MUST, SHOULD, and MAY express the intended future contract. They
do not change current behavior until separately implemented.

## Motivation

SpectroChemPy now distinguishes two persistence roles:

- SCP/PSCP is trusted native persistence.
- xarray-backed portable persistence is the leading candidate for scientific
  interchange and archive-oriented storage.

This distinction matters because the future portable path must support modern
scientific interoperability with:

- xarray;
- NetCDF;
- Zarr;
- Dask;
- the broader PyData ecosystem.

At the same time, SpectroChemPy must not collapse its own scientific object
model into a thin wrapper around xarray internals. `NDDataset` remains the
primary SpectroChemPy scientific data object. xarray is the canonical portable
carrier, not the source of truth for SpectroChemPy semantics.

This RFC defines the mapping contract precisely enough that a future
implementation should not need to reopen core architectural decisions.

## Canonical representation

### Decision

The canonical portable representation of one `NDDataset` MUST be an
`xarray.Dataset`, not a bare `xarray.DataArray`.

### Rationale

`xarray.Dataset` is retained because it can represent:

- one primary data variable;
- auxiliary coordinate variables;
- mask variables;
- portable metadata at dataset and variable level;
- future extension points without changing the top-level carrier model.

A `DataArray` view MAY still be exposed as a convenience in a future API, but
it is not the canonical round-trip representation.

### Primary variable convention

An exported `xarray.Dataset` MUST contain exactly one primary data variable
representing the numerical signal of the `NDDataset`.

The canonical convention is:

- the dataset-level attribute `scpy_primary_variable` identifies the primary
  variable name;
- the primary variable stores the actual numerical data;
- global dataset attrs carry SpectroChemPy container-level metadata;
- per-variable attrs carry signal-level metadata when appropriate.

The primary variable name SHOULD be:

- `name` when it is stable and non-empty;
- otherwise `data`.

Consumers MUST NOT assume that the primary variable is always named `data`;
they MUST consult `scpy_primary_variable` when present.

### Canonical top-level conventions

The canonical exported `xarray.Dataset` SHOULD use:

- `attrs["scpy_format"] = "nddataset-xarray"`
- `attrs["scpy_version"] = 1`
- `attrs["scpy_primary_variable"] = <primary variable name>`

These markers define the SpectroChemPy mapping convention. They are distinct
from any NetCDF, Zarr, or backend-specific versioning.

## Dimensions

### Mapping rule

`NDDataset.dims` MUST map directly to the dimension tuple of the primary
variable.

Dimension order MUST be preserved exactly.

If `ds.dims == ("y", "x")`, then the primary variable in xarray MUST use
`("y", "x")` in that same order.

### Naming

Dimension names MUST be explicit strings.

The canonical exported names SHOULD be the SpectroChemPy dimension keys, not
human-readable titles. Titles are display metadata, not structural identifiers.

This avoids ambiguity during reconstruction and avoids coupling the structural
schema to presentation labels.

### Reconstruction

`xarray -> NDDataset` reconstruction MUST use the primary variable's dimension
tuple as the source of truth for `NDDataset.dims`.

Round-trip guarantee:

- `NDDataset -> xarray -> NDDataset` MUST preserve dimension count, names, and
  order.

No stronger cross-tool guarantee is made if a third-party tool renames or
reorders dimensions.

## Coordinates

### Default per-dimension coordinate

Each SpectroChemPy dimension SHOULD export one default coordinate as the
dimension coordinate in xarray.

This default coordinate is the coordinate that SpectroChemPy would consider the
active coordinate for that dimension.

### Coordinate payload

For each default coordinate, the mapping SHOULD preserve:

- coordinate values;
- coordinate dimensional attachment;
- units;
- title if present;
- selected coordinate attrs needed for reconstruction.

Coordinate metadata SHOULD live primarily in coordinate variable attrs.

Recommended attrs:

- `units`
- `scpy_title`
- `scpy_coord_role = "default"`

### Monotone and non-monotone coordinates

Both monotone and non-monotone coordinates MUST be representable.

The mapping MUST preserve actual coordinate values and MUST NOT infer a linear
or monotone representation unless that inference is lossless and explicitly
defined by a future implementation.

Coordinate values are canonical; inferred linear metadata is secondary.

### Coordinate identity

Coordinate titles, display labels, and aliases MUST NOT be used as the sole
structural identity of a coordinate during reconstruction.

The owning dimension and exported coordinate variable name are the structural
anchors.

## CoordSet mapping

`CoordSet` is the main richness gap between SpectroChemPy and xarray.

### Default coordinate

For each dimension, one coordinate MUST be designated as the default exported
dimension coordinate.

That coordinate provides the portable minimum representation.

### Auxiliary same-dimension coordinates

Additional coordinates attached to the same underlying dimension SHOULD be
exported as auxiliary coordinate variables on the `xarray.Dataset`.

Recommended convention:

- each auxiliary coordinate variable is 1D and indexed by the same dimension;
- each carries attrs describing its role and owning dimension.

Recommended attrs:

- `scpy_coord_role = "auxiliary"`
- `scpy_owner_dim = <dimension name>`
- `scpy_coord_key = <stable exported key>`

### CoordSet completeness

The `CoordSet` round-trip contract is intentionally partial.

Specifically:

- default same-dimension coordinate selection MUST round-trip;
- auxiliary coordinate values SHOULD round-trip when preserved by the backend;
- aliases, internal lookup conveniences, and implementation-specific grouping
  details are best effort only;
- reference-sharing identity is not guaranteed to round-trip as identity.

### Reference semantics

If multiple SpectroChemPy coordinates historically share values or references,
the portable mapping MAY reconstruct equivalent values without reconstructing
the original internal reference topology.

This means:

- scientific coordinate meaning SHOULD be preserved;
- Python object identity and reference-sharing MUST NOT be guaranteed.

### Overall decision

`CoordSet` round-trip is therefore:

- complete for the default portable coordinate layer;
- partial for richer same-dimension grouping semantics;
- best effort for aliases and internal reference topology.

## Units

### Data units

Primary signal units MUST be exported explicitly.

The recommended canonical location is:

- primary variable attr `units`

### Coordinate units

Coordinate units MUST be exported on the corresponding coordinate variable
attrs using the same `units` key.

### Semantics

This RFC does not require a future implementation to depend on `pint` inside
xarray or inside a backend. The portable contract is textual unit preservation,
not runtime unit-engine preservation.

Guarantees:

- unit strings MUST round-trip when preserved by the backend;
- scientific intent of units SHOULD round-trip;
- automatic runtime reconstitution of a specific unit engine is an
  implementation concern, not part of the portable schema contract.

### CF conventions

CF-compatible unit strings SHOULD be preferred when feasible, but this RFC does
not require full CF normalization.

If SpectroChemPy-specific unit spellings are used, they SHOULD be preserved as
written rather than silently coerced.

## Masks

### Canonical representation

Masks SHOULD be represented explicitly, not only by injecting `NaN` into the
primary data variable.

The recommended canonical mapping is:

- primary variable retains the numerical payload;
- an auxiliary boolean variable stores the SpectroChemPy mask;
- dataset attr `scpy_mask_variable` identifies that variable.

The mask variable SHOULD share the same dimensions as the primary variable.

### Why not NaN alone

`NaN`-only encoding is insufficient because:

- integer and complex payloads do not map cleanly to `NaN`;
- missing-data semantics and masked-data semantics are not always identical;
- explicit masks are easier to reconstruct faithfully.

### Reconstruction

`xarray -> NDDataset` reconstruction MUST prefer the explicit mask variable when
present.

`_FillValue`, backend-native missing-data markers, or `NaN` MAY be used as
fallback hints, but they are not the canonical source of truth.

## Complex and HyperComplex data

### Complex

`complex64` and `complex128` are in scope.

The complex-data decision MUST distinguish the xarray model from backend
constraints:

- canonical xarray representation: native NumPy complex dtype;
- canonical NetCDF representation: split real/imag convention.

This distinction is required because xarray can naturally carry complex NumPy
arrays, while NetCDF portability imposes the real/imag split at backend export
time rather than at the xarray carrier level.

Therefore:

- `NDDataset -> xarray.Dataset` SHOULD preserve native complex dtype;
- `xarray.Dataset -> NDDataset` SHOULD preserve native complex dtype when
  present;
- any future NetCDF mapping SHOULD define an explicit split real/imag
  convention without changing the canonical xarray carrier model.

The xarray mapping is the canonical portable carrier for SpectroChemPy, not the
source of truth for SpectroChemPy semantics, and backend-specific export
constraints MUST NOT unnecessarily degrade the in-memory xarray representation.

### HyperComplex

HyperComplex or quaternion-like payloads are out of scope for guaranteed
portable round-trip in this RFC.

Current conclusion:

- ordinary NumPy-backed dataset data fits the xarray model;
- HyperComplex-specific semantics do not currently justify a distinct canonical
  portable schema here;
- no guarantee is made for quaternion dtype round-trip through xarray/NetCDF.

Classification:

- complex data: supported by convention;
- HyperComplex data: partially supported at best, otherwise out of scope for
  guaranteed portable interchange.

## Metadata and attrs

### Export policy

Portable export SHOULD preserve scientific metadata, not arbitrary runtime
internals.

Only JSON-compatible metadata are portable by default.

The following SHOULD be exported when present:

- title;
- name;
- description;
- author;
- origin;
- date-like creation/update metadata when stable;
- history;
- serializable scientific metadata entries from `meta`.

The following are out of portable scope unless a later RFC defines them
explicitly:

- arbitrary Python objects in `meta`;
- callable objects;
- class instances without a JSON-compatible representation;
- backend-specific opaque objects.

The following MUST NOT be treated as portable schema guarantees:

- live runtime services;
- backend-specific caches;
- display-only transient state;
- Python object identities;
- implementation-private helper fields.

Portable persistence is not Python object persistence. Metadata that is not
JSON-compatible MAY remain valid in native trusted persistence, but it is not
guaranteed by this xarray portable mapping.

### Attr namespace

SpectroChemPy-specific attrs SHOULD use an explicit `scpy_` prefix.

This avoids collisions with:

- xarray conventions;
- CF conventions;
- third-party attrs.

Recommended dataset attrs include:

- `scpy_format`
- `scpy_version`
- `scpy_primary_variable`
- `scpy_mask_variable`
- `scpy_history_format`

### History

History SHOULD be exported explicitly as metadata.

The canonical first-step representation SHOULD be textual and lossless relative
to the current `NDDataset.history` field, not a new provenance graph.

This RFC does not redefine history into a portable workflow/provenance model.

## Round-trip contract

### `NDDataset -> xarray -> NDDataset`

Guaranteed:

- numerical data values, except for backend-imposed precision changes;
- native complex dtype preservation on the xarray path;
- dimension names and order;
- default per-dimension coordinate values;
- data units and default coordinate unit strings;
- explicit mask values when the mask variable is preserved;
- core exported metadata under the `scpy_` convention and selected scientific
  attrs;
- identification of the primary variable.

Best effort:

- auxiliary same-dimension coordinates;
- coordinate titles and secondary metadata;
- JSON-compatible `meta` entries;
- history formatting beyond textual content equivalence.

Not guaranteed:

- object identity;
- internal `CoordSet` grouping topology;
- alias tables;
- shared-reference identity between coordinates;
- non-JSON-compatible metadata objects;
- HyperComplex dtype semantics;
- backend-specific encoding details;
- byte-for-byte equality.

### `xarray -> NDDataset -> xarray`

Guaranteed only for xarray objects that already follow the SpectroChemPy
mapping convention defined in this RFC.

Specifically, the future importer MAY reject or partially import generic
xarray objects that lack:

- a clear primary variable;
- reconstructible dimension coordinates;
- compatible unit metadata;
- supported numeric dtypes;
- unambiguous mask conventions.

Therefore:

- convention-compliant xarray datasets SHOULD round-trip predictably;
- arbitrary third-party xarray datasets are import candidates, not guaranteed
  full-fidelity round-trips.

## Non-goals

This RFC does not define:

- `Project` portable persistence;
- `ResultBase`, `AnalysisResult`, or `FitResult` persistence;
- plugin-specific portable schemas;
- native SCP/PSCP evolution;
- a migration plan from existing files;
- a NetCDF-only API;
- a Zarr-only API;
- provenance graph design;
- a complete HyperComplex schema;
- backend-specific chunking, compression, or dask policy.

## Relation to CSDM

xarray is retained as the primary portable target because it is the strongest
fit for the broader PyData and scientific interoperability ecosystem:

- mature xarray data model;
- strong NetCDF and Zarr alignment;
- wide ecosystem integration;
- lower adoption risk for general-purpose portable persistence.

CSDM remains architecturally interesting and scientifically credible,
especially for spectroscopy-oriented interchange and domain-rich dataset
semantics.

However, based on the CSDM audit:

- CSDM is better positioned as a complementary specialized exchange format;
- it is not the preferred primary portable carrier for SpectroChemPy as a
  whole at this stage;
- it should not replace the xarray-centered mapping model defined here.

See the related CSDM position statement and the portable-persistence RFCs for
the tracked maintainer direction.

## Recommendations

This RFC recommends the following:

1. Canonical representation: `NDDataset` SHOULD map to an `xarray.Dataset`
   carrying one logical primary signal plus explicit auxiliary variables.
2. Compatibility target: the mapping SHOULD guarantee strong round-trip for the
   core numerical signal, dimensions, default coordinates, units, masks, and
   essential scientific metadata.
3. Rich-coordinate limit: `CoordSet` semantics beyond the default coordinate
   layer SHOULD be treated as partial or best-effort portable features, not as
   a reason to abandon the xarray model.
4. Complex strategy: xarray mapping SHOULD preserve native complex NumPy dtype,
   while future NetCDF export SHOULD use an explicit split real/imag
   convention.
5. HyperComplex boundary: HyperComplex portable round-trip SHOULD remain out of
   guaranteed scope until a real scientific need justifies a dedicated schema
   decision.
6. Metadata convention: SpectroChemPy-specific mapping metadata SHOULD use the
   `scpy_` namespace consistently.

## Open questions

- Should the primary variable default name always be `data`, or should stable
  dataset names be promoted when available?
- Should labels be exported as auxiliary coordinates by default, or remain
  metadata unless their semantics are clearly positional?
- Should future work define a stricter serialized subset of `meta`, or accept
  only plain JSON-compatible metadata?
- Should complex split variables use a naming convention, attrs only, or both?
- Should a future implementation expose both a strict importer and a permissive
  best-effort importer for generic xarray objects?
- Should a later RFC define a separate portable mapping for `Project`, or keep
  portable persistence dataset-only?

Implementation note:

- a substantial `to_xarray()` / `from_xarray()` subset now exists in the
  current codebase;
- remaining work is contract refinement and any future optional extension,
  not first-time prototype implementation.
