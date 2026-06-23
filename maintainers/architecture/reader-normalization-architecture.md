# Reader Normalization Architecture

## Status

IMPLEMENTED — This architecture is authoritative for the maintained reader
normalization contract. The Reader Alignment campaign is COMPLETED for the
maintained wave-1 and wave-2 core readers: OMNIC, OPUS, JCAMP, LabSpec, CSV,
TopSpin, WiRE, Quadera, SOC, MATLAB/DSO, and SPC.

## Overview

This note is the primary maintainer reference for how imported reader
information should be normalized into the SpectroChemPy runtime model.

Its purpose is to answer one practical maintainer question:

```text
When a reader imports information,
where should it go?
```

The maintained rule is to normalize by semantic meaning, not by parser
convenience and not by format-specific habit.

Future reader contributors should consult this note first, then the underlying
RFCs if they need decision history.

For what the current portable xarray / NetCDF layer actually preserves after
runtime normalization, see
[`portable-persistence-model.md`](portable-persistence-model.md).

## Normalization Targets

Imported information has a small set of valid destinations:

- typed scientific identity fields;
- structural support state through coordinates and `CoordSet`;
- labels;
- provenance/history;
- `Meta` payloads;
- runtime-only parser state.

## Typed Fields

Use typed fields when the imported meaning is already part of the maintained
runtime object model and is stable across readers.

Typical typed-field destinations include:

- `name`
- `title`
- `description`
- `units`
- `filename`
- `origin`
- `author`
- `created`
- `modified`
- `history`

Use typed fields for shared concepts, not to host every vendor-specific
property.

## Coordinates

Use coordinates when the imported information describes support geometry or
axis-local support semantics.

Typical coordinate destinations include:

- axis values;
- axis units;
- axis titles;
- elapsed time support;
- timestamp support when timestamps locate observations along an axis;
- label-only support in cases where the source expresses a categorical or
  identifier axis rather than a numeric one.

Coordinates describe where data lives along a structural dimension.

Do not place support geometry into generic metadata when it belongs on a
coordinate.

## Labels

Use labels when imported information is support-local and acts as:

- an identifier;
- an annotation;
- a category;
- a display-facing pointwise name.

Typical examples:

- sample identifiers;
- acquisition identifiers;
- channel names;
- target names;
- categorical observation tags.

Labels are not a fallback for arbitrary imported strings.

If the imported string is actually source lineage, it belongs in provenance.
If it is support geometry, it belongs in coordinates.
If it is an unnormalized payload, it belongs in `Meta`.

## Provenance

Use provenance fields when imported information describes source and lineage
context.

Typical provenance destinations include:

- `filename`
- `origin`
- `author`
- `created`
- `modified`
- `history`

Examples of provenance-oriented imports:

- source file path or logical source name;
- reader/format origin;
- import log entries;
- vendor processing history;
- acquisition/session time when it describes source lineage rather than support
  geometry.

One source may legitimately provide both support time and provenance time. In
that case, preserve the conceptual distinction.

## `Meta`

Use `Meta` for useful imported payloads that do not belong to the normalized
core contract.

Typical `Meta` content includes:

- vendor-specific technical parameters;
- reader-specific payloads;
- imported values with real user value but no stable typed destination;
- extension or plugin metadata.

The preferred rule is:

```text
normalize shared meaning
retain useful remainder in Meta
```

Do not overfit the core object model to a single vendor by turning every useful
imported field into a typed attribute.

## Runtime-Only Parser State

Some information should not be retained at all after import.

Examples:

- offsets;
- sentinels;
- temporary decoder state;
- reconstruction helpers with no scientific or user-facing meaning;
- parser bookkeeping.

These are implementation details, not imported scientific metadata.

## Reader Consistency

Reader consistency means mapping equivalent semantics to the same conceptual
destination whenever practical.

Consistency does not mean that every reader must emit identical surfaces.
Different formats may legitimately expose different richness or different
support structure.

Acceptable diversity:

- vendor-specific technical payloads in `Meta`;
- different support structures driven by the source format;
- different levels of detail in imported comments or acquisition metadata.

Harmful fragmentation:

- equivalent source lineage sometimes stored as provenance and sometimes only
  as labels;
- support time and provenance time collapsed without distinction;
- vendor history text placed inconsistently between `history`,
  `description`, and untracked payloads;
- similar pointwise identifiers mapped inconsistently across readers.

## Temporal Information

Imported time is one of the main normalization boundaries.

Use provenance when time answers:

```text
when was this object acquired, created, imported, or processed
as a source event?
```

Use coordinates when time answers:

```text
where along the support axis does this observation live?
```

Use labels only when imported temporal values are best understood as support
annotations or identifiers rather than primary coordinate values.

## Runtime vs Persistence

Reader normalization targets the runtime model first.

### Runtime

Readers should construct the best SpectroChemPy runtime object they can from
the source meaning.

### Native persistence

Trusted native persistence can preserve richer runtime normalization, including:

- coordinate labels;
- vendor-specific `Meta` payloads;
- detailed provenance/history context.

### Portable persistence

Portable persistence preserves only the maintained portable subset:

- portable typed fields;
- portable coordinate structure;
- accepted portable label subset;
- JSON-compatible metadata.

Portable export is a projection of normalized runtime state. It should not be
the primary driver of normalization decisions.

## Maintainer Guidance

- Normalize by meaning, not by parser convenience.
- Use typed fields for shared concepts already owned by the core model.
- Use coordinates for support geometry and axis-local support semantics.
- Use labels for support-local identifiers, annotations, and categories.
- Use provenance fields for source, lineage, authorship, timestamps, and
  import/process logs.
- Use `Meta` for useful vendor-specific or not-yet-normalized payloads.
- Discard runtime-only parser state.
- When in doubt, preserve useful reader-specific richness in `Meta` rather than
  inventing a new typed field.

## Normalized Reader Model

The implemented reader model now uses the following maintained semantic
destinations:

- `description`: object-level descriptive text, sample notes, and free-form
  descriptive comments when they are not clearly support-local or
  event-oriented.
- `author`: source/operator identity when the reader can recover a meaningful
  person or user value.
- `origin`: reader or source-format identity using the maintained vocabulary or
  preserved source value where the contract explicitly allows it.
- `acquisition_date`: dataset/session provenance time when a reliable source
  timestamp is available.
- `history`: explicit import events, reader-driven reorder events, and vendor
  processing history when the source clearly provides an event-like trail that
  can be extracted safely.
- coordinates: support geometry, axis-local titles/units, elapsed time, and
  timestamp axes that locate observations structurally.
- labels: pointwise identifiers, acquisition names, channel names, and other
  support-local annotations or categories.
- `Meta`: useful vendor-specific technical payloads and reader-specific
  information that does not belong to a normalized typed field.

This is the maintained “semantic destination” model for contributors adding or
updating readers.

## Canonical Origin Vocabulary

The maintained origin values for the covered readers are:

| Reader | `origin` value |
| ------ | -------------- |
| OMNIC | `omnic` |
| OPUS | `opus-<type>` (e.g. `opus-AB`, `opus-SM`) |
| JCAMP-DX | Preserved `##ORIGIN`; deterministic sorted `"; "` join for multi-origin |
| LabSpec | `labspec` |
| TopSpin | `topspin` |
| CSV generic | caller-specified |
| CSV OMNIC | `omnic` |
| WiRE | application/version per reader convention |
| Quadera | `quadera` |
| SOC | `soc` |
| MATLAB generic | `matlab` |
| DSO | `dso` |
| SPC | `thermo galactic` |

## Dual-Time Rule

Imported time has two distinct roles that must be preserved separately:

- **Provenance time** (`acquisition_date`): records the acquisition session,
  dataset creation, or source lineage event at the dataset level.
- **Support time** (coordinate values): locates individual observations along
  the support axis.

Both may coexist in the same imported dataset. The reader must not collapse
one into the other. Removing or overwriting time coordinates when setting
`acquisition_date` is a normalization error.

Example:
- `dataset.acquisition_date` = `2024-01-02 09:00:00` (session start)
- `dataset.y.coord` = `[0.0, 2.5, 5.0, ...]` seconds (elapsed observation time)

## History Policy

The maintained history policy for reader normalization is:

1. **Import events**: every reader SHOULD add one explicit import event when it
   successfully constructs a dataset from an external source.
2. **Vendor processing history**: if the source provides meaningful processing
   history and the reader can extract it safely, it SHOULD be preserved in
   `history` without overwriting the import event.
3. **Reader-driven reordering**: if the reader explicitly reorders observations
   (e.g. sorting by acquisition date), it SHOULD append a distinct history
   entry describing the reorder.
4. **Merge/stack events**: when the importer combines multiple datasets, it
   SHOULD append a merge or stack event.
5. **Wording**: no global wording standard is required. Meaningful event content
   is the requirement, not identical text.

## Reader Alignment Status

Reader normalization alignment is COMPLETED for the maintained reader-alignment
campaign scope.

### Wave 1

| Reader | `acquisition_date` | `origin` | `history` | Semantic tests |
| ------ | ------------------ | -------- | --------- | -------------- |
| OMNIC | Set from parsed timestamps | `omnic` for all variants | Import + vendor history | CHARACTERIZED |
| OPUS | Set from parameter timestamp | `opus-<type>` | Import event | CHARACTERIZED |
| JCAMP | Set from earliest LONGDATE/TIME | Preserved `##ORIGIN`; deterministic multi-origin | Import + sort event | CHARACTERIZED |
| LabSpec | Set from acquisition start | `labspec` | Import event | CHARACTERIZED |
| CSV | OMNIC date from filename | caller-specified or `omnic` | Import event | CHARACTERIZED |
| TopSpin | Set from vendor DATE | `topspin` | Import event | CHARACTERIZED |

### Wave 2

| Reader | `acquisition_date` | `origin` | `history` | Semantic tests |
| ------ | ------------------ | -------- | --------- | -------------- |
| WiRE | Set from first ORGN/session timestamp | application/version per reader convention | Import event | CHARACTERIZED |
| Quadera | Set from first acquisition timestamp | `quadera` | Import event | CHARACTERIZED |
| SOC | Inherited OMNIC-style acquisition provenance | `soc` | SOC-specific import event appended without losing inherited provenance | CHARACTERIZED |
| MATLAB/DSO | DSO acquisition provenance promoted when available | `matlab` / `dso` | Import event; DSO vendor history preserved | CHARACTERIZED |
| SPC | Set from parsed header/session timestamp | `thermo galactic` | Import event | CHARACTERIZED |

### Campaign outcome

Major outcomes of the completed campaign:

- semantic characterization baselines now exist for the targeted readers;
- `acquisition_date` is normalized across both campaign waves where reliable
  source/session timestamps exist;
- `origin` normalization and canonical vocabulary are documented and
  implemented for the covered readers;
- history consistency is established around explicit import events, with vendor
  processing history preserved where the source clearly exposes it;
- author normalization is implemented where meaningful source/operator identity
  is available;
- the contributor guide for adding readers now documents the maintained
  semantic destinations and the dual-time rule.

Carroucell remains intentionally outside the completed reader-alignment
campaign scope. Its temperature representation and related plugin semantics are
future enhancement topics, not unfinished core reader-alignment work.
