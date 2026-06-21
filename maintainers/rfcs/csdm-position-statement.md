# Current Position on CSDM

**Status:** Accepted maintainer position

**Scope:** Architectural role of the Core Scientific Dataset Model (CSDM) in
SpectroChemPy.

**Related issue:**
[#1153](https://github.com/spectrochempy/spectrochempy/issues/1153)

**Supporting context:** This position is informed by the recent maintainer
evaluation of CSDM interoperability and native-persistence candidates.

## Decision

SpectroChemPy considers CSDM a credible optional scientific exchange format,
particularly for spectroscopy and NMR-oriented interoperability.

CSDM is not the native persistence format for SpectroChemPy and is not the
primary portable persistence layer.

The current architecture remains:

```text
Native persistence:
    NDDataset / Project <-> SCP / PSCP

Primary portable layer:
    NDDataset <-> xarray.Dataset <-> NetCDF

Optional specialized exchange:
    NDDataset <-> CSDM
```

## CSDM as an Optional Exchange Format

CSDM has useful scientific semantics for multidimensional data, physical
units, complex and componented signals, correlated measurements, and sparse
sampling. These concepts make it a plausible exchange format for communities
and tools that already understand CSDM.

Future CSDM support may therefore be considered as an optional import/export
adapter for `NDDataset`. Such support should:

- remain an optional dependency;
- preserve standard CSDM meaning wherever possible;
- define a deliberately bounded scientific mapping;
- report unsupported or lossy SpectroChemPy features explicitly;
- be validated with files produced by independent CSDM tools.

No CSDM implementation is committed by this statement.

## CSDM Is Not Native Persistence

SCP/PSCP remains the SpectroChemPy-owned native persistence path. It is
responsible for reconstructing SpectroChemPy-specific objects and relationships
that do not naturally belong to a portable dataset exchange model.

CSDM does not naturally represent the complete SpectroChemPy runtime model,
including:

- rich `CoordSet` topology, aliases, and reference identity;
- explicit mask semantics without an additional convention;
- recursive heterogeneous `Project` contents;
- typed analysis Result objects;
- plugin-owned or HyperComplex semantics.

Using CSDM for exact native reconstruction would require a substantial
SpectroChemPy-specific schema inside CSDM application metadata. This would
weaken interoperability while duplicating the native persistence contract.

SpectroChemPy therefore does not plan to replace SCP/PSCP with CSDM.

## CSDM Is Not the Primary Portable Layer

The canonical portable carrier for one `NDDataset` is `xarray.Dataset`, with
NetCDF as the current persistence backend.

This layer already supports numerical data, dimensions, default and auxiliary
same-dimension coordinates, portable labels, units, masks, JSON-compatible
metadata, and complex data. It also connects SpectroChemPy to the broader
PyData and NetCDF ecosystems.

CSDM complements this architecture where its specialized scientific semantics
are useful. It does not replace xarray or NetCDF, and portable NetCDF
persistence should not be routed through CSDM.

## Conditions for Reconsideration

The position may be revisited if concrete demand appears, for example:

- users need routine exchange with a CSDM-based tool or data repository;
- an external community provides representative compatibility files;
- CSDM adoption and multi-tool support materially increase;
- a plugin, especially an NMR-oriented plugin, demonstrates a clear use case;
- CSDM evolves first-class support for currently awkward SpectroChemPy
  semantics.

Any future implementation proposal should begin with a narrow mapping RFC and
an isolated prototype. It should not reopen native persistence or replace the
established xarray/NetCDF portable path without new evidence.

## Current Position

- **Recommended:** CSDM as an optional specialized exchange format.
- **Not recommended:** CSDM as native SpectroChemPy persistence.
- **Not recommended:** CSDM as the primary portable layer.
- **Revisit:** only when real interoperability demand or significant ecosystem
  change provides a concrete reason.
