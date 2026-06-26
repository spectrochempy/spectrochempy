[Maintainers](../../README.md) · [Architecture](../INDEX.md)

# Portable Persistence Model

## Status

IMPLEMENTED — The portable surface described here reflects the currently
implemented xarray / NetCDF persistence for `NDDataset`. The Portable Metadata
Subset Contract is fully implemented for the maintained core subset.

## Overview

This note is the primary maintainer reference for the currently implemented
portable persistence surface of `NDDataset`.

Its purpose is practical:

```text
What currently survives portable xarray / NetCDF round-trips?
```

This note documents the implemented model. It does not repeat RFC debate or
campaign history.

The current portable path is:

- `NDDataset.to_xarray()`
- `NDDataset.from_xarray()`
- `NDDataset.to_netcdf()`
- `NDDataset.from_netcdf()`

NetCDF persistence is xarray-backed in the current architecture:

```text
NDDataset
  -> canonical xarray Dataset carrier
  -> NetCDF-safe xarray projection when needed
  -> NetCDF file or bytes
```

## Portable vs Native Persistence

SpectroChemPy maintains a strict distinction between:

- native trusted persistence;
- portable scientific persistence.

Portable persistence is the narrow, interoperable subset intended to survive
outside the native runtime model.

The current portable implementation is intentionally:

- textual where practical for provenance fields;
- JSON-compatible for extensible metadata payloads;
- explicit about structural carriers for masks, labels, and auxiliary
  coordinates;
- narrower than the full runtime richness of `CoordSet`, labels, provenance,
  and result containers.

Native persistence remains the richer path for runtime fidelity and
implementation-private state.

## Portable Scientific Identity

The current portable model preserves these dataset-level scientific identity
fields:

- `name`
- `title`
- `description`

Current carrier keys:

- `scpy_name`
- `scpy_title`
- `scpy_description`

The primary xarray data variable is also tracked explicitly through:

- `scpy_primary_variable`

This keeps the portable reconstruction of dataset identity independent from
third-party xarray variable renaming heuristics when the SpectroChemPy carrier
attrs are still present.

## Portable Structural Metadata

The current portable model preserves:

- dimension count;
- dimension names;
- dimension order;
- default per-dimension coordinates;
- coordinate values;
- coordinate units;
- coordinate titles;
- same-dimension auxiliary coordinates in the current narrow carrier model;
- dataset masks;
- complex numerical data;
- dataset data units.

### Default coordinates

Default coordinates are carried as xarray dimension coordinates.

Their portable semantics include:

- coordinate values;
- dimensional attachment;
- units when present;
- titles when present.

### Same-dimension auxiliary coordinates

Same-dimension auxiliary coordinates are preserved as non-dimension coordinate
variables with explicit ownership markers.

Current carrier attrs include:

- `scpy_coord_role`
- `scpy_owner_dim`
- `scpy_default`
- `scpy_title`

This preserves alternate numeric support values, but not the full runtime
topology of `CoordSet`.

### Masks

Masks are preserved through an explicit mask variable referenced by:

- `scpy_mask_variable`

In the NetCDF-safe carrier, boolean masks are temporarily encoded as `int8`
with a restoration marker, then restored to boolean on import.

### Complex data

Complex arrays are preserved in NetCDF through the current split real/imag
convention using explicit carrier attrs, then reconstructed on import.

## Portable Labels

The currently implemented portable label subset is:

- one-dimensional;
- textual;
- coordinate-local;
- optionally nullable.

Supported current behavior:

- simple textual coordinate labels survive xarray and NetCDF round-trips;
- nullable textual labels survive through an explicit none-mask carrier attr.

Current label carrier conventions include:

- non-dimension label variables such as `{dim}_labels`;
- `scpy_coord_role="label"`;
- `scpy_owner_dim`;
- `scpy_label_none_mask` when needed.

Not part of the current portable implementation:

- label-only coordinates;
- multi-row labels;
- non-string labels;
- richer categorical or mixed-runtime label structures.

## Portable Provenance

The current portable model preserves these dataset-level provenance fields:

- `author`
- `origin`
- `created`
- `modified`
- `acquisition_date`
- `history`

Current carrier keys:

- `scpy_author`
- `scpy_origin`
- `scpy_created`
- `scpy_modified`
- `scpy_acquisition_date`
- `scpy_history`

### Timestamp fields

`created`, `modified`, and `acquisition_date` are carried as stable textual
attrs using:

- `datetime.isoformat(sep=" ", timespec="seconds")`

The importer restores them with `datetime.fromisoformat()` when the attrs are
present and parseable.

Current behavior:

- timezone-aware values preserve their offset textually;
- timezone-naive values remain naive;
- missing attrs keep the existing fallback behavior.

### History

Portable history is preserved as textual event content, not as a richer native
event object model.

Current xarray carrier form:

- list of history strings in `scpy_history`

Current NetCDF-safe form:

- JSON-encoded form of that same list

On import, the portable history text is rebuilt into the internal runtime
history structure so multi-entry round-trips survive.

## Portable Meta Payloads

The current portable model preserves:

- JSON-compatible `Meta`;
- nested JSON-compatible `Meta`;
- reader-specific metadata when it already lives in JSON-compatible `Meta`;
- vendor-specific metadata when it already lives in JSON-compatible `Meta`;
- plugin-specific metadata when it already lives in JSON-compatible `Meta`.

Current carrier attrs:

- `scpy_meta`
- `scpy_skipped_meta_keys`

Behavior:

- JSON-compatible payloads survive;
- nested dict/list content survives;
- tuples normalize to lists;
- NumPy scalar values normalize to Python scalar values;
- non-JSON-compatible payloads are skipped rather than coerced.

The portable implementation does not currently introduce dedicated typed
mapping for reader-specific or vendor-specific metadata beyond this
JSON-compatible `Meta` channel.

## Runtime-only Concepts

The following remain runtime-only or native-only in the current maintained
model:

- `filename`;
- label-only coordinates;
- multi-row labels;
- non-string labels without stable textual encoding;
- full `CoordSet` grouping/reference topology;
- exact auxiliary coordinate naming identity;
- internal storage details and convenience aliases;
- arbitrary Python objects in `Meta`;
- parser-only runtime state;
- result-object persistence.

These are not all architectural defects.

Some are intentionally out of scope for portable persistence, while others are
possible future extensions rather than current contract failures.

## Result Export and Persistence Boundary

`ResultBase`, `AnalysisResult`, and `FitResult` are not part of the implemented
portable persistence surface.

For the 0.11 architecture, the supported bridge is dataset export:

```text
Result
  -> named NDDataset outputs
  -> xarray / NetCDF or dataset-only Project persistence
```

This is an export, not Result round-trip persistence. Loading the exported
datasets does not reconstruct the original Result type, parameter record, or
diagnostic grouping.

If a future portable Result format is proposed, it would require its own
versioned manifest because Result outputs may:

- have unrelated dimensions;
- mix datasets, arrays, and scalar diagnostics;
- be produced by optional plugins;
- include output datasets whose dtype has narrower portability guarantees than
  ordinary core numerical arrays.

Results are currently runtime objects. No structured Result persistence work is
currently committed. If it is pursued,
the design would first need decisions about:

- live-view, cached-view, or fit-time snapshot semantics;
- a restricted parameter and diagnostic value domain;
- stable Result type identifiers and schema versions;
- provenance and input-summary rules;
- unknown-plugin/provider behavior.

Typed Project membership remains deferred independently. The current
architecture does not require either structured Result persistence or broader
Project membership. Dataset persistence remains the established persistence
model.

### Native persistence clarification

Current version-2 `.scp` and `.pscp` writes use safe raw-base64 numerical
payloads and explicit format/version markers. Default loading rejects
pickle-backed historical payloads unless the user explicitly opts into trusted
legacy loading with `allow_unsafe_legacy=True`.

Older architecture discussion that classifies every SCP/PSCP write as
pickle-backed trusted persistence describes the historical format, not the
current safe-by-default writer and loader.

## Current Limitations

The main current limitations of the portable implementation are:

- `filename` is still not preserved;
- label-only coordinates are not yet supported by the portable mapping;
- auxiliary same-dimension coordinate values survive, but stable auxiliary
  coordinate naming is not part of the current maintained portable contract;
- richer `CoordSet` topology and reference identity do not survive;
- only the current textual label subset is portable;
- only JSON-compatible `Meta` payloads survive;
- result containers are outside the portable persistence surface.

For maintainers, the practical reading order is:

1. this note for the implemented portable model;
2. `metadata-and-support-model.md` for the runtime ownership and metadata
   taxonomy behind that model;
3. `reader-normalization-architecture.md` for how imported information should
   enter the runtime model before portable projection;
4. the portable-persistence RFC cluster only when design history or future
   extension decisions are needed.
