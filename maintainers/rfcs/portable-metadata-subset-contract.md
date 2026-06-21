# Portable Metadata Subset Contract

## Status

Proposed Maintainer RFC.

This document is conceptual and normative in intent.

It defines the maintainer-level contract for one question:

```text
What should survive portability?
```

It builds on:

- [`metadata-taxonomy-contract.md`](metadata-taxonomy-contract.md)
- [`label-semantics-contract.md`](label-semantics-contract.md)
- [`provenance-and-history-contract.md`](provenance-and-history-contract.md)
- [`reader-metadata-normalization-contract.md`](reader-metadata-normalization-contract.md)
- [`scientific-object-model-and-persistence-boundaries.md`](scientific-object-model-and-persistence-boundaries.md)
- [`trusted-and-portable-persistence.md`](trusted-and-portable-persistence.md)
- [`nddataset-xarray-mapping-specification.md`](nddataset-xarray-mapping-specification.md)
- [`xarray-backed-netcdf-persistence.md`](xarray-backed-netcdf-persistence.md)

It must not redesign:

- dimensions;
- coordinates;
- labels;
- metadata taxonomy;
- provenance.

It does not define implementation details, migration sequencing, or API work.

The key words MUST, SHOULD, MAY, and MUST NOT are to be interpreted as
normative requirements for maintainers and future contributors.

## 1. Purpose

The recent architecture campaign established stable contracts for:

- dimensions;
- coordinates and `CoordSet`;
- metadata taxonomy;
- labels;
- reader normalization;
- provenance and history.

The architecture conformance audit then identified one major gap:

```text
portable persistence currently preserves much less
than the architecture now suggests
```

The project therefore needs a narrow but explicit contract for which metadata,
provenance, label, and support-related semantics belong to portable
persistence.

This RFC defines that contract for:

- xarray export/import;
- NetCDF export/import;
- future portable formats that aim to preserve the same maintained subset.

Inside scope:

- the current portable surface;
- portability principles;
- portable status of typed metadata fields;
- portable status of structural support metadata;
- portable label subset;
- portable provenance subset;
- portable `Meta` subset;
- result-object portability boundaries;
- conceptual mapping guidance;
- implementation touch points.

Outside scope:

- code changes;
- migration plans;
- native SCP/PSCP fidelity;
- full provenance graphs;
- workflow reproducibility systems;
- plugin runtime state;
- arbitrary Python object persistence.

## 2. Part 1 — Current Portable Surface

This section describes current behavior, not intended behavior.

### 2.1 Preserved today

Current xarray / NetCDF round-trips preserve:

- dimension count, names, and order;
- default per-dimension coordinates;
- same-dimension auxiliary coordinates in the current narrow carrier model;
- data units;
- coordinate units;
- dataset `name`;
- dataset `title`;
- masks;
- JSON-compatible `Meta`;
- a narrow string-label subset on coordinates;
- complex numerical data through the split real/imag NetCDF convention.

### 2.2 Partially preserved today

Current portable persistence partially preserves:

- `CoordSet` richness:
  - default coordinate survives;
  - auxiliary same-dimension coordinates usually survive;
  - richer grouping and reference topology does not fully survive;
- labels:
  - only 1D string-only coordinate labels are currently portable;
  - `None` values are supported through explicit mask metadata;
  - richer label forms are not portable;
- metadata:
  - only JSON-compatible `Meta` payloads survive;
  - non-JSON-compatible payloads are skipped rather than normalized.

### 2.3 Not preserved today

Current portable persistence does not preserve, as a maintained round-trip
contract:

- `description`;
- `author`;
- `origin`;
- `filename`;
- `created`;
- `modified`;
- `acquisition_date`;
- `history`;
- multi-row labels;
- non-string labels;
- richer `CoordSet.references` semantics;
- project-level persistence;
- result-object persistence.

### 2.4 Current diagnosis

The current portable surface is coherent but narrower than the current
maintainer architecture.

This RFC does not treat that narrowness as a bug by itself.

The architectural problem is not that portable persistence is narrow. The
problem is that the boundary is not yet stated precisely enough to guide
future implementation and review.

## 3. Part 2 — Portability Principles

Portable persistence is not native persistence.

The maintained distinction is:

```text
native persistence
    preserves trusted runtime richness

portable persistence
    preserves a stable, safe, interoperable scientific subset
```

### 3.1 What makes metadata portable

Metadata is portable when it satisfies most of the following:

- it carries stable scientific or provenance meaning outside one Python
  process;
- it can be represented without executable Python state;
- it has reproducibility, interpretation, or interoperability value;
- it can be expressed through stable typed fields, portable attrs, coordinate
  variables, or other schema-bound structures;
- it does not depend on SpectroChemPy-internal object identity or runtime
  convenience.

### 3.2 What is not required for portability

Portable persistence does not need to preserve:

- Python object identity;
- live references between runtime objects;
- every convenience alias;
- every internal grouping detail;
- every implementation-private or display-private helper;
- arbitrary reader or plugin payloads.

### 3.3 Runtime-first rule

The SpectroChemPy runtime model remains the source of truth.

Portable formats must adapt to the SpectroChemPy contract as far as their
schemas allow. Portable constraints MUST NOT redefine the runtime model.

### 3.4 Practical rule

Portable persistence SHOULD preserve:

- scientific interpretation;
- essential support geometry;
- meaningful provenance needed to understand source and lineage;
- metadata that remains useful across tools and languages.

Portable persistence SHOULD avoid:

- trusted-runtime richness that cannot be represented safely or stably;
- arbitrary Python-specific payloads;
- metadata whose meaning depends on private SpectroChemPy internals.

## 4. Part 3 — Scientific Identity Metadata

This section evaluates dataset-level scientific identity metadata.

### 4.1 `name`

Portable status: `SHOULD preserve`

Rationale:

- `name` is useful for human identity, variable naming, and reconstruction;
- it is not always scientifically fundamental, so portability should not fail
  if a backend or third-party tool rewrites it;
- the xarray mapping already uses it as part of the primary-variable
  convention.

### 4.2 `title`

Portable status: `MUST preserve`

Rationale:

- `title` often expresses the primary scientific meaning of the signal;
- losing it weakens interpretation even when the numerical data survives;
- it is textual and portable.

### 4.3 `description`

Portable status: `SHOULD preserve`

Rationale:

- `description` often carries meaningful scientific narrative or acquisition
  context;
- it is useful across tools;
- it is not structurally required to interpret dimensions or coordinates, so
  it is not elevated to MUST.

### 4.4 Scientific descriptors in `Meta`

Portable status: `SHOULD preserve`

Rationale:

- some scientific descriptors do not have stable typed fields;
- when JSON-compatible, these descriptors remain valuable scientific context;
- portability should preserve them as part of the portable `Meta` subset.

### 4.5 Scientific identity summary

Portable persistence MUST preserve enough scientific identity metadata that a
portable consumer can still answer:

- what is this signal;
- what do these axes mean;
- what core scientific descriptors remain attached.

## 5. Part 4 — Structural Metadata

Structural metadata is the most fundamental portable layer.

### 5.1 Dimensions

Portable status: `MUST preserve`

Includes:

- dimension count;
- dimension names;
- dimension order.

Rationale:

- dimensions are the structural schema keys of the dataset;
- without them, coordinate attachment and signal meaning degrade immediately.

### 5.2 Default coordinates

Portable status: `MUST preserve`

Includes:

- coordinate values;
- coordinate dimensional attachment;
- coordinate units;
- coordinate titles when present.

Rationale:

- default coordinates are the minimum support geometry needed for scientific
  interpretation.

### 5.3 Same-dimension auxiliary coordinates

Portable status: `SHOULD preserve`

Rationale:

- they are scientifically useful and already part of the maintained `CoordSet`
  model;
- they matter for support richness and alternate axis descriptions;
- some portable carriers can preserve them cleanly;
- they are not always necessary for minimal scientific interpretation.

### 5.4 `CoordSet` grouping and references

Portable status: `MAY preserve`

Includes:

- default-coordinate selection;
- auxiliary ownership markers;
- selected same-dimension grouping markers.

Portable status for runtime-only richness: `RUNTIME ONLY`

Includes:

- Python identity of grouped coordinates;
- full internal storage topology;
- exact reference-sharing identity;
- internal aliases and convenience lookup surfaces.

Rationale:

- the scientific meaning of alternate coordinates matters;
- the exact runtime topology does not define portable scientific meaning.

### 5.5 Masks

Portable status: `MUST preserve`

Rationale:

- masks are part of the structural and interpretive state of the signal;
- they affect which values are scientifically valid;
- treating them as optional would weaken round-trip fidelity materially.

### 5.6 Coordinate-local scientific titles and units

Portable status:

- coordinate units: `MUST preserve`
- coordinate titles: `SHOULD preserve`

Rationale:

- units are essential support semantics;
- titles strongly aid axis interpretation but are not as structurally central
  as dimensions and values.

### 5.7 Runtime-only structure

The following structural concepts are runtime-only unless a future RFC states
otherwise:

- implementation-private storage details;
- exact `CoordSet._storage` layout;
- exact alias naming conveniences;
- presentation-only structural helpers;
- transient parser or reconstruction state.

## 6. Part 5 — Labels

This section defines the portable label subset.

### 6.1 String labels

Portable status: `SHOULD preserve`

Conditions:

- labels are attached to coordinates;
- labels are one-dimensional along one structural dimension;
- labels are text values or `None`;
- labels serve as identifiers, categories, annotations, or support-local
  names.

Rationale:

- these are the most interoperable and stable label forms;
- they map naturally to portable coordinate-like structures.

### 6.2 Categorical labels

Portable status: `SHOULD preserve`

When categorical labels can be represented as simple textual labels, they fall
inside the same portable subset as string labels.

Portable persistence does not need a separate categorical object model for
this RFC.

### 6.3 Label-only coordinates

Portable status: `MAY preserve`

Rationale:

- label-only support is a real runtime pattern;
- some carriers can represent it cleanly as support-local coordinate content;
- interoperability is less predictable than for numeric default coordinates.

Portable persistence SHOULD preserve label-only coordinates when the carrier
can still represent the dimension and its support meaning unambiguously.

### 6.4 Multi-row labels

Portable status: `RUNTIME ONLY`

Rationale:

- multi-row labels are part of current runtime richness;
- they do not yet have a sufficiently clear, interoperable portable schema in
  the maintained model.

### 6.5 Non-string labels

Portable status: `RUNTIME ONLY`

Includes:

- arbitrary Python objects;
- mixed-type label arrays;
- non-textual structures without a stable portable representation.

Rationale:

- they are not robustly portable across tools and languages.

### 6.6 Portable label subset summary

The portable label subset is:

- support-local;
- one-dimensional;
- textual;
- optionally nullable;
- attached to a coordinate or coordinate-like support variable.

Everything richer remains runtime-first unless promoted by a future RFC.

## 7. Part 6 — Provenance

This section evaluates typed provenance fields.

### 7.1 `author`

Portable status: `SHOULD preserve`

Rationale:

- authorship is useful source context;
- it is textual and interoperable;
- it is not structurally required for scientific interpretation.

### 7.2 `origin`

Portable status: `SHOULD preserve`

Rationale:

- origin communicates source format, system, or acquisition provenance;
- it helps portable consumers interpret lineage;
- it is especially useful when data passed through readers or imports.

### 7.3 `filename`

Portable status: `MAY preserve`

Rationale:

- filename can be useful lineage context;
- it is often environment-specific and may become misleading after copying,
  export, or archival relocation;
- it should not be required for portability.

### 7.4 `created`

Portable status: `SHOULD preserve`

Rationale:

- creation time often contributes meaningfully to provenance;
- it is portable if represented as stable text/time data.

### 7.5 `modified`

Portable status: `MAY preserve`

Rationale:

- modification time may be useful provenance;
- it is more volatile and tool-dependent than created/source time;
- not every portable carrier or workflow preserves it reliably.

### 7.6 `acquisition_date`

Portable status: `MAY preserve`

Rationale:

- acquisition time may represent provenance or support geometry depending on
  semantics;
- if it is dataset-level source lineage, it may be preserved as provenance;
- if it is axis-local support time, it belongs in coordinates instead.

This field therefore cannot be a universal MUST in the portable subset.

### 7.7 `history`

Portable status: `SHOULD preserve`

Rationale:

- history is the explicit event trail of the object;
- it often carries important transformation and import provenance;
- textual history is portable enough to preserve meaningfully;
- portable persistence is not required to preserve every structured internal
  nuance of history formatting, only the textual event content.

### 7.8 Provenance summary

Portable persistence SHOULD preserve a useful textual provenance subset, but
it is not required to become a complete provenance-graph system.

## 8. Part 7 — Meta Payloads

### 8.1 JSON-compatible `Meta`

Portable status: `SHOULD preserve`

Rationale:

- JSON-compatible `Meta` carries useful scientific or technical context;
- it is already the main extensible metadata reservoir;
- it can be carried without importing arbitrary Python runtime state.

### 8.2 Reader-specific metadata

Portable status: `MAY preserve`

Rationale:

- some reader payloads are useful beyond the original runtime;
- others are too tool-specific or too noisy;
- portable persistence should preserve them only when they remain useful and
  JSON-compatible.

### 8.3 Vendor metadata

Portable status: `MAY preserve`

Rationale:

- vendor metadata may have scientific or reproducibility value;
- not all vendor payloads deserve portable-contract status;
- when preserved, they should remain clearly non-core metadata rather than
  being mistaken for normalized typed fields.

### 8.4 Plugin metadata

Portable status: `MAY preserve`

Rationale:

- plugin metadata may be valuable, but portability must not depend on
  plugin-specific runtime objects;
- only portable, JSON-compatible plugin payloads should survive by default.

### 8.5 MUST NOT require

Portable persistence MUST NOT require support for:

- arbitrary Python objects;
- callables;
- live estimator or plugin instances;
- backend handles;
- non-JSON-compatible payloads;
- implementation-private parser state.

### 8.6 Portable `Meta` subset summary

The portable `Meta` subset is:

- JSON-compatible;
- scientifically or technically useful;
- non-executable;
- not required to encode full native runtime richness.

## 9. Part 8 — Result Objects

This RFC does not redesign result objects.

It sets the current portability boundary only.

### 9.1 Result provenance

Portable status: `RUNTIME ONLY`

at the result-object container level.

Rationale:

- result objects do not yet have an accepted standalone persistence contract;
- output datasets may themselves be portably persisted through the dataset
  contract;
- the result container is not yet a portable scientific object.

### 9.2 Estimator identity

Portable status: `RUNTIME ONLY`

at the result-object container level.

Rationale:

- estimator identity may be scientifically useful, but it belongs to a future
  result-object persistence decision, not to the current dataset portability
  contract.

### 9.3 Diagnostics

Portable status: `RUNTIME ONLY`

at the result-object container level.

Rationale:

- diagnostics may survive indirectly when exported as datasets or plain
  values through another explicit contract;
- this RFC does not promote result containers into portable persistent
  objects.

## 10. Part 9 — Mapping Strategy

This section is conceptual, not implementation-level.

### 10.1 Natural mapping layers

The natural portable mapping is:

- dimensions:
  - dimension names and variable dimensions;
- default coordinates:
  - coordinate variables;
- auxiliary same-dimension coordinates:
  - auxiliary coordinate variables;
- core typed metadata:
  - dataset attrs and variable attrs;
- portable labels:
  - coordinate-aligned textual support variables;
- masks:
  - explicit mask variables or equivalent maintained mask projection;
- portable `Meta`:
  - structured JSON-compatible attrs.

### 10.2 Mapping rule

Portable constructs should be chosen according to semantic role:

- structural support semantics map to dimensions and coordinate variables;
- scientific identity and provenance map to attrs or comparable typed
  metadata fields;
- portable label subset maps to support-local textual variables;
- reconstruction metadata maps to explicit format markers rather than private
  Python conventions.

### 10.3 Mapping boundary

Portable mapping MUST preserve scientific meaning first.

It MUST NOT attempt to encode every runtime convenience as if it were a
portable scientific contract.

## 11. Part 10 — Explicit Non-Goals

This RFC does not define:

- full native SCP fidelity;
- complete provenance graphs;
- workflow reproducibility systems;
- plugin runtime state persistence;
- arbitrary Python object persistence;
- full result-object persistence;
- exact runtime object identity restoration;
- a complete archival format beyond the maintained portable subset.

## 12. Part 11 — Final Contract

### 12.1 Normative summary

Portable persistence MUST preserve the structural scientific core of one
`NDDataset`.

Portable persistence SHOULD preserve the principal scientific identity,
provenance, portable labels, and portable `Meta` that remain useful outside
the native trusted runtime.

Portable persistence MAY preserve additional JSON-compatible, semantically
useful context when it does not compromise safety, clarity, or
interoperability.

Portable persistence MUST NOT require native-runtime richness, arbitrary
Python objects, or exact reconstruction of internal runtime topology.

### 12.2 Contract table

| Category | Portable Status |
| -------- | --------------- |
| Dimension count, names, order | MUST preserve |
| Default coordinates | MUST preserve |
| Data units | MUST preserve |
| Coordinate units | MUST preserve |
| Masks | MUST preserve |
| Dataset title | MUST preserve |
| Coordinate values | MUST preserve |
| Coordinate dimensional attachment | MUST preserve |
| Dataset name | SHOULD preserve |
| Dataset description | SHOULD preserve |
| Scientific descriptors in JSON-compatible `Meta` | SHOULD preserve |
| Auxiliary same-dimension coordinates | SHOULD preserve |
| Coordinate titles | SHOULD preserve |
| String or textual categorical coordinate labels | SHOULD preserve |
| Author | SHOULD preserve |
| Origin | SHOULD preserve |
| Created | SHOULD preserve |
| History | SHOULD preserve |
| JSON-compatible `Meta` | SHOULD preserve |
| Filename | MAY preserve |
| Modified | MAY preserve |
| Acquisition date as dataset-level provenance | MAY preserve |
| Label-only coordinates | MAY preserve |
| Reader-specific JSON-compatible payloads | MAY preserve |
| Vendor-specific JSON-compatible payloads | MAY preserve |
| Plugin-specific JSON-compatible payloads | MAY preserve |
| Full `CoordSet` grouping/reference topology | RUNTIME ONLY |
| Multi-row labels | RUNTIME ONLY |
| Non-string labels without stable textual encoding | RUNTIME ONLY |
| Arbitrary Python objects in `Meta` | RUNTIME ONLY |
| Runtime-only parser state | RUNTIME ONLY |
| Result-object estimator identity | RUNTIME ONLY |
| Result-object diagnostics | RUNTIME ONLY |
| Result-object provenance container state | RUNTIME ONLY |

## 13. Part 12 — Implementation Impact

This RFC implies likely future work in:

- xarray mapping:
  - typed metadata coverage;
  - portable labels;
  - portable `Meta` policy;
- NetCDF mapping:
  - attr-level provenance coverage;
  - portable metadata subset enforcement;
- reader normalization:
  - clearer distinction between runtime richness and portable subset;
- portable tests:
  - explicit coverage for the preserved subset;
  - explicit non-goal coverage for runtime-only features.

This RFC does not prescribe an implementation sequence.

## 14. Part 13 — Candidate Follow-Up Work

### Documentation only

- Update architecture notes once the portable subset is implemented and
  stabilized.
- Synchronize the xarray and NetCDF RFCs with any final naming or status
  changes that follow acceptance of this contract.

### Tests only

- Add portable round-trip tests for newly required typed provenance fields.
- Add explicit negative tests for runtime-only label and `Meta` features.
- Add contract tests covering SHOULD/MAY categories where implementation
  commits to preserving them.

### Small implementation alignment

- Extend xarray export/import to preserve the accepted typed metadata subset.
- Extend NetCDF mapping to preserve the accepted textual provenance subset.
- Align portable warnings and skips with the accepted subset rather than the
  current ad hoc behavior.

### Future RFCs

- Result Provenance Contract, if result-object persistence becomes a real
  design target.
- Portable Label Extensions RFC, if multi-row or richer categorical labels
  later require a maintained portable schema.
- Portable Metadata Normalization follow-up, if `Meta` classification needs a
  stricter rule than JSON-compatibility plus semantic usefulness.

## 15. Part 14 — Promotion Candidates

### Future architecture notes

- Portable persistence profile for metadata and provenance.
- Maintainer guidance on how to classify runtime-only versus portable
  metadata during reader or persistence work.

### Future implementation campaigns

- xarray metadata/provenance alignment campaign;
- NetCDF metadata/provenance alignment campaign;
- portable round-trip contract test campaign.

### Historical-only material

- current implementation quirks that motivated this RFC should remain in local
  audits once the portable subset is accepted and implemented.

## 16. Conclusion

The maintained portable boundary should be neither maximal nor vague.

It should preserve:

- the structural scientific core;
- the main scientific identity fields;
- a useful textual provenance subset;
- a narrow but meaningful label subset;
- JSON-compatible metadata that remains valuable outside the native runtime.

It should not attempt to preserve:

- full native SCP richness;
- arbitrary Python state;
- internal runtime topology;
- every current convenience surface.

This gives SpectroChemPy a stable target for future portable persistence work
without collapsing the runtime-first architecture into a lowest-common-denominator
interchange model.
