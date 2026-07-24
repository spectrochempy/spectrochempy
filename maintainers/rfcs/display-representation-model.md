[Maintainer Docs](../README.md) · [RFC Index](INDEX.md)

# Display / Representation Model

## Status

Superseded RFC.

This historical RFC defined the representation model contract before
implementation. The authoritative maintained reference now lives in the display
architecture note.

## Superseded By

[`display-architecture.md`](../architecture/display-architecture.md)

## Related Issue

#843 — Display / Representation Architecture

## Original Context

The display layer was fragmented: Coord, CoordSet, and NDDataset shared a
`_cstr()` → `convert_to_html()` pipeline with sentinel-marker parsing, while
Project had a completely separate path. There was no single semantic source of
truth for what each object should communicate.

## Key Decisions Reached

| Question | Decision |
|----------|----------|
| `str(obj) == repr(obj)` for array-like types? | **Yes.** Compact display remains the default terminal representation. |
| Project `__repr__` with name? | **Yes.** |
| Project metadata visible in display? | **Yes.** In detailed/HTML display only. |
| Project join common pipeline? | **Yes.** Through `_cstr()` and the semantic HTML model. |
| HTML pipeline separated from `_cstr()`? | **Yes.** HTML now uses `_repr_sections()` → `_render_sections()`. |
| Shared representation contract document? | **Yes.** This RFC (superseded by `display-architecture.md`). |
| CoordSet compact-repr showing titles? | **Yes.** Coordinate names/titles provide useful identification. |
| `_repr_html_` heading differ from `__str__()`? | **Yes.** Notebook headings are richer (name/title). |

## Four-Level Model

The representation model distinguishes four levels, all preserved in the final
implementation:

* `repr(obj)` → compact object fingerprint
* `str(obj)` → compact human-readable terminal representation
* `pstr(obj)` → detailed inspection representation
* `_repr_html_()` → rich notebook representation (semantic, not sentinel-based)
