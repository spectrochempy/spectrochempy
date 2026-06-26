[Maintainers](../../README.md) · [RFCs](../INDEX.md)

# Reader Metadata Normalization Contract

## Status

Accepted Maintainer RFC.

This document is conceptual and normative in intent.

It defines the maintainer-level contract for one question:

```text
When a reader imports metadata,
where should it go?
```

It builds on:

- [`metadata-taxonomy-contract.md`](metadata-taxonomy-contract.md)
- [`label-semantics-contract.md`](label-semantics-contract.md)
- [`coordinate-and-coordset-semantics.md`](coordinate-and-coordset-semantics.md)
- [`dimensional-semantics-contract.md`](dimensional-semantics-contract.md)

It must not reopen those decisions.

It does not define implementation details, migration sequencing, or API work.

The key words MUST, SHOULD, and MAY are to be interpreted as normative
requirements for maintainers and future contributors.

## 1. Purpose

SpectroChemPy readers import metadata from many different ecosystems:

- OMNIC;
- OPUS;
- JCAMP;
- CSV-derived conventions;
- MATLAB exports;
- NetCDF / xarray mappings;
- other vendor or ad hoc formats.

Imported metadata arrives with highly variable structure, quality, and
semantic stability.

Some imported fields already become:

- typed dataset fields;
- coordinates;
- labels;
- provenance fields;
- `Meta` payloads.

Other fields remain reader-local or format-specific.

This RFC defines the normalization contract maintainers should use when
deciding where imported metadata belongs in the SpectroChemPy runtime model.

Inside scope:

- current reader metadata behavior;
- available normalization destinations;
- principles for typed-field versus label versus provenance versus `Meta`
  placement;
- consistency goals across readers;
- interaction with trusted and portable persistence;
- philosophy for vendor-specific payload retention.

Outside scope:

- parser implementation;
- file-format support decisions;
- persistence redesign;
- metadata taxonomy redesign;
- coordinate redesign;
- dimensional redesign;
- migration plans.

## 2. Part 1 — Current Landscape

This section describes current behavior, not intended behavior.

### 2.1 Typed-field mappings already exist

Current readers already normalize some imported metadata into explicit runtime
fields.

Observed examples include:

- `dataset.name` from filenames or embedded acquisition names;
- `dataset.title` from signal semantics such as absorbance, transmittance, or
  detector signal;
- `dataset.units` from vendor intensity units;
- `dataset.filename` from the source path;
- `dataset.description` from imported comments or descriptive text;
- `dataset.history` from import provenance or vendor processing history;
- `dataset.origin` in some readers, such as OPUS and OMNIC-family imports;
- `dataset.author` in the MATLAB reader.

This means readers already act as semantic normalizers, not merely byte
decoders.

### 2.2 Provenance mappings are partly normalized

Current readers often map source provenance into typed dataset fields.

Observed current provenance patterns include:

- `filename` set from the opened path;
- `origin` set to reader or format identity such as `omnic`, `tga`, or
  `opus-*`;
- `history` set to import messages or vendor processing text;
- timestamps represented as support coordinates;
- occasional author/date import in the MATLAB reader.

However, provenance handling is not fully consistent across readers:

- some readers use `origin` explicitly, others leave it unset;
- some imported dates become coordinate values plus labels rather than typed
  provenance fields;
- some vendor processing history is promoted into `history`, others remain in
  comments or are dropped.

### 2.3 Coordinate mappings are a major normalization path

Many readers normalize imported support information directly into coordinates.

Observed examples include:

- numeric x-axis values with titles and units from OMNIC, OPUS, and JCAMP;
- time or timestamp y-axes for multi-acquisition data;
- label-only coordinates in MATLAB or Quadera cases;
- coordinate-local titles such as `acquisition timestamp (GMT)`, `elapsed
  time`, `Time`, `wavenumbers`, or `spectrum index`.

This confirms that support metadata frequently belongs in `Coord` and
`CoordSet`, not in generic dataset-level metadata.

### 2.4 Label mappings are common and semantically mixed

Current readers frequently import strings into coordinate labels.

Observed examples include:

- acquisition dates and titles in JCAMP;
- acquisition dates and filenames in OMNIC;
- acquisition dates, sample names, and filenames in OPUS;
- acquisition date and sample name in OMNIC-exported CSV;
- channel names in Quadera;
- MATLAB-imported categorical axis labels.

These labels mix several semantic roles:

- support-local identifiers;
- provenance-adjacent details;
- scientific categorization;
- display aids.

This is why the Label Semantics Contract is a prerequisite for this RFC.

### 2.5 Raw metadata retention already exists

Some readers already retain format-specific payloads in `Meta`.

Observed examples include:

- OPUS parameter blocks stored in `dataset.meta`;
- OMNIC acquisition or interferogram properties stored in `dataset.meta`;
- format-specific technical values such as laser frequency, collection length,
  optical velocity, and interferogram markers.

This demonstrates that not every imported field is forced into a typed core
field.

### 2.6 Reader-specific behavior remains substantial

Current readers still differ meaningfully in normalization style.

Examples:

- OPUS stores a large parameter payload in read-only `Meta`;
- OMNIC elevates comments into `description` and some acquisition properties
  into `Meta`;
- JCAMP uses labels and inferred units/titles but relatively little extra
  metadata retention;
- MATLAB imports author/date plus coordinate labels when available, but with a
  different shape and completeness profile from vendor readers;
- CSV import behavior depends strongly on the caller-provided `origin`
  convention;
- Quadera uses label-only support for channel identity.

This diversity is partly appropriate and partly a source of fragmentation.

## 3. Part 2 — Metadata Destinations

Using the Metadata Taxonomy Contract, reader-imported metadata has a finite set
of valid destinations.

### 3.1 Typed scientific identity fields

These fields describe what the imported scientific object is.

Typical destinations:

- `title`
- `description`
- `units`
- stable scientific names or descriptors

Reader metadata SHOULD land here when it expresses a stable,
cross-reader-meaningful scientific identity already represented in the runtime
object model.

### 3.2 Structural metadata

Structural metadata describes support organization and object topology.

Typical destinations:

- dimensions;
- coordinates;
- coordinate titles and units;
- support attachment through `CoordSet`.

Reader metadata SHOULD land here when it determines:

- axis values;
- axis units;
- axis titles;
- support grouping;
- structural coordinate identity.

### 3.3 Provenance metadata

Provenance metadata describes where the imported object came from.

Typical destinations:

- `filename`
- `origin`
- `author`
- `created`
- `modified`
- `history`

Reader metadata SHOULD land here when it records source, authorship, import
lineage, or vendor processing lineage rather than scientific support content.

### 3.4 Labels

Labels are a distinct destination, not merely a fallback.

Typical label destinations include:

- sample identifiers;
- acquisition identifiers;
- categorical descriptors;
- support-local names that vary point-by-point along one dimension.

Reader metadata SHOULD become labels when it identifies or annotates support
positions on a dimension and fits the Label Semantics Contract better than a
typed provenance field or generic `Meta`.

### 3.5 `Meta` payloads

`Meta` is the main extensible destination for imported metadata that is useful
but not part of the normalized core contract.

Typical cases:

- vendor-specific payloads;
- imported metadata with unclear or mixed semantics;
- technical acquisition properties without a stable typed core field;
- extension payloads retained for user access or later normalization.

### 3.6 Runtime-only reader state

Some import-time values SHOULD remain runtime-only and not enter the final
scientific object at all.

Examples:

- parser state;
- temporary import helpers;
- reconstruction intermediates;
- offsets, sentinels, and decoding-control variables with no scientific value.

These are implementation state, not imported metadata.

## 4. Part 3 — Normalization Philosophy

This section defines general principles before any format-specific policy.

### 4.1 What deserves a typed field?

Reader metadata SHOULD become a typed field when all of the following are true:

- its meaning is stable across multiple readers or scientific contexts;
- the runtime object model already has an appropriate field;
- using that field improves cross-reader consistency;
- the field expresses primary scientific identity, provenance, or structure
  rather than vendor-specific detail.

Examples:

- signal units;
- axis units and titles;
- source filename;
- imported processing history;
- stable scientific title or description.

### 4.2 What should remain in `Meta`?

Reader metadata SHOULD remain in `Meta` when one or more of the following are
true:

- its meaning is format-specific;
- its scientific importance is real but its normalized home is not yet agreed;
- promoting it to a typed field would overfit the core model to one vendor;
- it is useful to preserve for inspection, debugging, or advanced workflows.

`Meta` is therefore the right place for retained but not-yet-core metadata.

### 4.3 What should become labels?

Reader-imported strings SHOULD become labels when they are:

- support-local;
- pointwise or axis-aligned;
- primarily identifying, annotating, or categorizing support positions;
- more meaningful as coordinate-attached content than as dataset-wide
  provenance or `Meta`.

Typical examples include:

- sample names;
- acquisition identifiers;
- target names;
- channel names;
- categorical observation tags.

### 4.4 What should become provenance?

Reader metadata SHOULD become provenance when it answers:

```text
Where did this object come from?
Who produced it?
When was it created or processed?
```

Typical examples:

- file path or logical source filename;
- reader/format origin;
- vendor processing history text;
- author or acquisition date when it describes source lineage rather than
  support-local observation identity.

### 4.5 Prefer normalization by meaning, not by storage convenience

Maintainers SHOULD normalize imported metadata according to semantic meaning,
not according to the easiest available field.

This means:

- not every string belongs in labels;
- not every unmapped value belongs in `description`;
- not every technical value deserves a typed field;
- not every imported detail should survive import.

## 5. Part 4 — Reader Consistency

Current reader behavior shows both acceptable diversity and harmful
fragmentation.

### 5.1 Acceptable diversity

Diversity is acceptable when it reflects genuine source differences.

Examples:

- OMNIC and OPUS expose different technical acquisition payloads;
- MATLAB imports may contain ad hoc axis labels with less standardization than
  vendor spectroscopy formats;
- Quadera channel names are naturally label-like rather than numeric support;
- some formats provide rich vendor history while others do not.

This kind of diversity does not require forced homogenization.

### 5.2 Harmful fragmentation

Fragmentation is harmful when equivalent semantics are mapped inconsistently.

Observed current tensions include:

- acquisition times sometimes appear as coordinate values, sometimes only as
  labels, and sometimes not as typed provenance at all;
- filenames can appear as `filename`, as coordinate labels, and inside
  `description`;
- vendor history text is sometimes placed in `history`, sometimes folded into
  `description`, and sometimes effectively discarded;
- origin handling is explicit in some readers and implicit in others;
- similar support-local identifiers are normalized differently depending on
  format.

### 5.3 Maintained position on consistency

Reader consistency SHOULD mean:

- equivalent semantics map to the same conceptual destination whenever
  practical;
- reader-specific richness may still be preserved in `Meta`;
- the core runtime contract should not vary arbitrarily by format.

Consistency is a semantic goal, not a demand that every reader produce the same
surface shape regardless of source reality.

## 6. Part 5 — Portable Persistence Interaction

Normalization affects three layers:

```text
runtime model
native persistence
portable persistence
```

### 6.1 Runtime model

Normalization primarily defines the runtime scientific object.

Reader output SHOULD first be correct as a SpectroChemPy object, even before
considering portable export.

### 6.2 Native persistence

Trusted native persistence SHOULD preserve the normalized runtime result,
including richer `Meta` payloads and label semantics, subject to the native
format's existing scope.

This is the persistence layer that preserves SpectroChemPy richness, not the
one that forces portable simplification.

### 6.3 Portable persistence

Portable persistence SHOULD preserve only the normalized metadata subset that
belongs to the portable contract.

That includes:

- portable typed fields;
- portable coordinate structure;
- portable labels when they fit the accepted portable-label subset;
- JSON-compatible metadata explicitly admitted by the portable mapping.

Portable persistence MUST NOT be treated as the reason to avoid useful runtime
normalization.

### 6.4 What should never become part of the portable contract

The following SHOULD generally remain outside the portable contract unless a
later RFC says otherwise:

- parser-internal state;
- vendor-specific technical payloads with no stable cross-format meaning;
- arbitrary imported blobs preserved only for inspection;
- label forms outside the portable-label subset;
- runtime-only reconstruction helpers.

## 7. Part 6 — Vendor-Specific Metadata

This section answers:

```text
How much vendor-specific information
should SpectroChemPy normalize?
```

### 7.1 Normalize shared semantics aggressively enough

SpectroChemPy SHOULD normalize vendor metadata when it expresses a shared
scientific or provenance concept already meaningful in the runtime model.

Examples:

- axis values, titles, and units;
- signal units and titles;
- source filename;
- import history;
- stable sample/acquisition identifiers when they are support-local.

### 7.2 Do not force vendor uniqueness into core fields

SpectroChemPy SHOULD NOT create or overload core typed fields merely to host
every vendor-specific acquisition property.

Examples of fields that often belong better in `Meta`:

- vendor instrument tuning details;
- format-specific internal flags;
- technical acquisition parameters without stable cross-reader meaning;
- ambiguous imported descriptors whose semantics are not yet maintained as part
  of the core model.

### 7.3 Preserve useful vendor detail without promoting it prematurely

The preferred pattern is:

```text
normalize shared meaning
retain useful remainder in Meta
discard pure parser state
```

This avoids both data loss and overfitting the core object model to one vendor.

## 8. Part 7 — Relationship to Labels

This section explicitly depends on the Label Semantics Contract.

### 8.1 When imported strings should become labels

Imported strings SHOULD become labels when they are:

- attached to support positions;
- pointwise along one dimension;
- identifiers, categories, or annotations for those positions;
- better understood as support-local content than as dataset-level provenance.

Examples:

- sample names;
- acquisition names repeated per spectrum;
- channel names;
- categorical observation classes.

### 8.2 When imported strings should become provenance

Imported strings SHOULD become provenance when they identify:

- source files;
- processing origin;
- authorship;
- acquisition or processing lineage.

If the same source detail appears once as dataset lineage and once as
point-by-point support identity, the two roles SHOULD remain conceptually
distinct even when the current runtime representation uses both.

### 8.3 When imported strings should remain metadata

Imported strings SHOULD remain metadata when they are:

- vendor comments;
- technical notes;
- descriptive payloads without clear support-local or provenance semantics;
- too format-specific for immediate normalization.

### 8.4 Maintained caution

Because labels can carry provenance-adjacent content, maintainers SHOULD avoid
using labels as a generic catch-all for imported strings.

Labels are not a dumping ground for any string a reader happens to find.

## 9. Part 8 — Explicit Non-Goals

This RFC does not define:

- parser implementation details;
- file-format support priorities;
- the full provenance contract;
- the full label indexing contract;
- the portable encoding of every metadata category;
- a redesign of `Meta`;
- a redesign of dimensions, coordinates, or labels.

## 10. Part 9 — Final Contract

Maintainers SHOULD apply the following rules when normalizing reader metadata:

1. Normalize by semantic meaning, not by import convenience.
2. Use typed fields for stable shared concepts already owned by the runtime
   object model.
3. Use coordinates and `CoordSet` for structural support information.
4. Use labels for support-local identifiers, annotations, and categories that
   fit the Label Semantics Contract.
5. Use provenance fields for source, lineage, authorship, timestamps, and
   import history.
6. Use `Meta` for useful vendor-specific or not-yet-normalized payloads.
7. Discard parser-internal state instead of persisting it as metadata.
8. Preserve reader-specific richness where useful, but do not let vendor
   payloads redefine the core object model.
9. Treat portable persistence as a projection of the normalized runtime model,
   not as the primary driver of normalization.

## 11. Promotion Candidates

### Future architecture notes

Implemented as:

- [`maintainers/architecture/reader-normalization-architecture.md`](../architecture/reader-normalization-architecture.md)
  documents the reader normalization contract across the maintained spectroscopy
  formats.

Remaining future work:

- A focused architecture note on reader-to-runtime mapping patterns for common
  vendor ecosystems.

### Future RFCs

- `provenance-and-history-contract.md`
- `portable-metadata-subset-contract.md`
- `reader-label-and-support-normalization-guidelines.md`

### Historical-only material

- per-reader implementation quirks that matter only to parser maintenance;
- temporary characterization details for incomplete vendor formats;
- migration notes for future normalization cleanup work.
