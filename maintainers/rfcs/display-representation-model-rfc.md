# Display / Representation Model RFC

**Status:** Completed — superseded by [`maintainers/architecture/display-architecture.md`](../architecture/display-architecture.md)
**Issue:** #843 — Display / Representation Architecture

This RFC defined the representation model contract before implementation
began. All major decisions were implemented. The authoritative reference
for the final architecture is `maintainers/architecture/display-architecture.md`.

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
