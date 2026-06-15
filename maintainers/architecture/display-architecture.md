# Display Architecture

## Background

Before the Display Architecture work (Issue #843), SpectroChemPy's
presentation layer was fragmented.  Each object produced its own HTML
independently, and the shared `convert_to_html()` function relied on
parsing sentinel markers (`\0`, `\0\0`, `\0\0\0`) from `_cstr()` output
to reconstruct HTML structure.  This coupling meant that improving
terminal output could inadvertently break HTML rendering, and that
adding a new display feature required understanding two layers at once.

The goal was to separate concerns: terminal output remains textual;
HTML output is built from an explicit structured model.

---

## Current Architecture

SpectroChemPy now has two independent rendering layers:

### Terminal Representation

```
__repr__()          → short one-line identifier
__str__()           → human-readable detail
_cstr()             → full annotated text (used by __str__ for most objects)
```

* Purpose: terminal interaction, debugging, logging.
* All core objects produce terminal output through `_cstr()`.
* No object uses HTML helpers for terminal output.

### HTML Representation

```
_repr_sections()    → list[DisplaySection]
                         ↓
_render_sections()  → HTML string (body)
                         ↓
_html_heading()     → heading string
                         ↓
                    → <details><summary>heading</summary>body</details>
                    → wrapped in <div class="scp-output">
```

* Purpose: notebook display, documentation rendering.
* HTML rendering no longer depends on sentinel parsing for Coord,
  CoordSet, NDDataset, or Project.
* `_html_heading()` provides type-specific summaries with stable
  identity (no internal UUIDs in headings).

---

## Semantic Display Model

HTML content is built from two dataclasses in `src/spectrochempy/utils/print.py`:

| Class | Key fields | Purpose |
|-------|------------|---------|
| `DisplayItem` | `kind` (`field`, `data`, `label`, `block`), `value`, optional `key` | A single piece of display content |
| `DisplaySection` | `role` (`summary`, `data`, `dimension`), `title`, `items: list[DisplayItem]` | Groups related items under a role |

Role behaviour:

* **`summary`** — items rendered inline under the heading (no collapsible toggle).
* **`data`** — primary data, wrapped in `<details>`.
* **`dimension`** — coordinate dimension info, wrapped in `<details>`.

Item kind behaviour:

* **`field`** — key-value pair rendered as a labeled row.
* **`data`** / **`label`** — array content with type-appropriate formatting.
* **`block`** — opaque content (e.g. a project hierarchy line) rendered as `<div>`.

### Example

```
heading:  NDDataset [spectrum]

  name        spectrum        (inline field)
  author      lab             (inline field)
  ───[Data]───                (collapsible)
  [1.  2.]   float32 | size: 2
```

Backed by:

```python
[
    DisplaySection("summary", "Summary", [
        DisplayItem("field", "spectrum", "name"),
        DisplayItem("field", "lab", "author"),
    ]),
    DisplaySection("data", "Data", [
        DisplayItem("data", "[1.  2.]"),
        DisplayItem("field", "float32 | size: 2"),
    ]),
]
```

---

## Object Participation

| Object        | HTML Path           | `_repr_sections()` | `_repr_html_()` via |
|---------------|---------------------|--------------------|----------------------|
| Coord         | Semantic            | coord.py           | `_render_sections()` |
| CoordSet      | Semantic            | coordset.py        | `_render_sections()` |
| NDDataset     | Semantic            | nddataset.py       | `_render_sections()` |
| Project       | Semantic            | project.py         | `_render_sections()` |
| NDArray       | Sentinel            | —                  | `convert_to_html()`  |
| NDComplexArray| Sentinel (inherited)| —                  | `convert_to_html()`  |

Objects on the semantic path go through `_repr_sections()` →
`_render_sections()`.  Objects on the sentinel path still use
`_cstr()` → `convert_to_html()`.

All objects retain `_cstr()` for terminal output regardless of their
HTML path.

---

## Design Decisions

These decisions were reached during the RFC and migration work:

1.  **`str(obj) == repr(obj)` for array-like objects.**  NDDataset,
    Coord, and NDArray share a common `__str__` / `__repr__` convention.
    Project has distinct short (`__repr__`) and detailed (`__str__`)
    representations.

2.  **Project participates in common heading conventions.**
    `Project._repr_html_()` uses `_html_heading()` and the same outer
    wrapper structure as other semantic objects.

3.  **Project metadata is visible.**  Project HTML shows name (always),
    author (if set), and description (if set) as inline summary fields,
    matching the NDDataset convention.

4.  **HTML is separated from `_cstr()`.**  No semantic-path object
    calls `_cstr()` during HTML rendering.  The sentinel path is
    preserved for objects not yet migrated.

5.  **Coord identity is exposed through `name:title` headings.**  A
    Coord named `x` with title `wavenumbers` renders as
    `Coord [x:wavenumbers]`.

6.  **CoordSet identity is exposed through coordinate `name:title`
    headings.**  A CoordSet containing `x:wavenumbers` and
    `y:acquisition timestamp` renders those pairs in its heading.

7.  **Same-dim CoordSets use `Coord` terminology, not `Dimension`.
    Synthetic child coordinates of a shared dimension are presented as
    `Coord` (e.g., `Coord [_1]`) since they are coordinates of a shared
    dimension, not dimensions themselves.

---

## Lessons Learned

* Semantic rendering can be introduced incrementally — each object was
  migrated independently without breaking the others.
* A flat `DisplaySection` × `DisplayItem` model was sufficient for all
  four migrated objects; the renderer did not require hierarchy.
* `Coord` items are reused by `CoordSet`, and `CoordSet` sections are
  reused by `NDDataset`, avoiding duplicated formatting across layers.

---

## Open Questions

These are recorded for future reference.  They are not active roadmap
items.

* **Coord identity:**  The `name:title` heading convention works well
  but the distinction between Coord identity and dimension identity may
  need revisiting if richer dimension semantics are added.

* **Summary information density:**  The current summary shows a fixed
  set of fields (name, author, created, description).  Future objects
  may need configurable or context-dependent summary content.

* **NDArray / NDComplexArray migration:**  NDArray and NDComplexArray
  still use the sentinel path.  They can be migrated to the semantic
  model if their display requirements justify the effort.

* **Hypercomplex display semantics:**  NDComplexArray display uses
  `R[` / `I[` notation.  A future hypercomplex type would need similar
  conventions for its components (RR/RI/IR/II).

---

## Related Documents

| Document | Purpose |
|----------|---------|
| [`architecture/display-architecture-audit.md`](architecture/display-architecture-audit.md) | Original fragmentation analysis (archived) |
| [`src/spectrochempy/utils/print.py`](../src/spectrochempy/utils/print.py) | `DisplayItem`, `DisplaySection`, `_render_sections()`, `_html_heading()` |
