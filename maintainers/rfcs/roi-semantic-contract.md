[Maintainers](../../README.md) · [RFCs](../INDEX.md)

# roi — Semantic Contract RFC

**Status:** Accepted maintainer decision record

**Implementation status:** `NDDataset.roi` has been assessed as orphaned
historical state and should be removed from the public runtime data model.
Legacy serialized `roi` / `_roi` entries should remain load-compatible and be
ignored on read.

## Purpose

This document records the focused audit of `NDDataset.roi` and the resulting
maintainer decision.

It does not redesign interactive display, plotting, or future selection APIs.

## Summary

`roi` was historical structural state carried by `NDArray` / `Coord` /
`NDDataset`, but no current global semantic contract was visible.

Observed behavior before removal was inconsistent:

- preserved through shape operations;
- preserved through reductions;
- preserved through indexing and slicing;
- recomputed by some processing wrappers;
- preserved unchanged by others;
- absent from active user-facing documentation except for historical notes.

The audit conclusion is that `roi` behaved like orphaned interactive-selection
state rather than stable scientific metadata.

## Definition

Before removal, ROI was defined in core array infrastructure:

- `_roi = tr.List(allow_none=True)` on `NDArray`
- default value derived lazily from `limits`
- property getter/setter on `NDArray`
- included in `_attributes_()` copy propagation on `NDArray`, `Coord`, and
  `NDDataset`
- excluded from equality comparisons
- unit-converted by generic unit-conversion helpers

This made ROI pervasive in runtime copying even though no coherent semantic
policy existed for it.

## Production reads and writes

The audit found no active production feature that clearly owned ROI semantics.

- No meaningful production workflow writes ROI as part of analysis, IO, or
  plotting logic.
- Remaining production behavior was mostly implicit propagation through
  copy/constructor/unit-conversion helpers.
- Processing wrappers exposed inconsistent emergent behavior: some paths
  recomputed ROI from the data range, others preserved it unchanged.
- The only plotting references found were commented-out `plotly.py` snippets,
  indicating dead historical experimentation rather than active runtime use.

In other words, ROI survived in the object model without a live production
owner.

## Test usage

Most non-core ROI usage was in characterization tests added during the
mathematical-semantics campaign.  Those tests were valuable because they
exposed stale propagation behavior, but they did not establish ROI as a
supported user-facing feature.

Characterized historical behavior included:

- preservation through shape operations;
- preservation through reductions;
- preservation through indexing/slicing;
- propagation from the last dataset in concatenation;
- recomputation vs preservation divergence across processing wrapper families.

These tests documented drift and inconsistency; they did not justify keeping
ROI as part of the public runtime model.

## Documentation usage

Documentation references were limited and historical:

- maintainer RFC and audit documents discussed ROI as stale structural state;
- a historical whatsnew note (`v0.2.0`) mentioned addition of ROI;
- no current user guide, tutorial, or example established ROI as an active
  supported feature.

This strongly suggests ROI was not a living public feature.

## Serialization impact

Because ROI participated in generic attribute propagation, old native
serialized datasets and projects may contain `roi` / `_roi` payloads.

Removing ROI from the runtime model therefore requires compatibility handling:

- old files should continue to load successfully;
- legacy serialized ROI entries should be ignored on read;
- loaders should not restore obsolete ROI state onto live objects.

This mirrors the compatibility posture adopted for removed `modeldata`.

## Semantic classification

Using the mathematical-semantics taxonomy, ROI is best classified as:

- interactive selection state; and
- orphaned historical state.

It is not well-described as scientific context or provenance, and it was too
inconsistently handled to count as reliable structural metadata.

## Maintainer decision

Maintainership chose **removal** rather than semantic rehabilitation.

### Rationale

1. No active production owner or global contract was found.
2. Characterization showed inconsistent and frequently stale propagation.
3. Documentation did not support ROI as a current public feature.
4. Any future interactive ROI workflow should use explicit interface/view
   state rather than hidden dataset state.

### Compatibility posture

- Remove ROI from live `NDArray` / `Coord` / `NDDataset` objects.
- Keep reading old serialized files, but ignore obsolete `roi` / `_roi`
  fields on load.

## Final conclusion

ROI appears historical but still serialized; remove with compatibility
handling.
