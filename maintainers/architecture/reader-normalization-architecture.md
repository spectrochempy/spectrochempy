# Reader Normalization Architecture

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
