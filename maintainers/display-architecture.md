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

The core model consists of two simple dataclasses:

### DisplayItem

Represents a single piece of display content.

| Attribute | Type   | Purpose                            |
|-----------|--------|------------------------------------|
| `kind`    | `str`  | Content type: `field`, `data`, `label`, `block` |
| `value`   | `str`  | Display value                      |
| `key`     | `str`  | Optional label key for field items |

Kind semantics:

* **`field`** — key-value pair (e.g. `name: myproj`).  Rendered inline
  when under a `summary` section, or in a table when under other roles.
* **`data`** — numeric array content.  Rendered with type-appropriate
  formatting.
* **`label`** — coordinate label.  Rendered with distinct styling.
* **`block`** — opaque content block (e.g. a project hierarchy line).
  Rendered as `<div>value</div>`.

### DisplaySection

Groups related `DisplayItem` objects under a role.

| Attribute | Type              | Purpose                          |
|-----------|-------------------|----------------------------------|
| `role`    | `str`             | Section type: `summary`, `data`, `dimension` |
| `title`   | `str`             | Section heading                  |
| `items`   | `list[DisplayItem]` | Contained items                |

Role semantics:

* **`summary`** — key-value metadata displayed inline under the object
  heading (no collapsible wrapper).
* **`data`** — primary data content.  Wrapped in a collapsible
  `<details>` toggle.
* **`dimension`** — coordinate dimension information.  Wrapped in a
  collapsible `<details>` toggle.

### Example

For `NDDataset([1.0, 2.0], name="spectrum", author="lab")`:

```
heading:  NDDataset [spectrum]

  name        spectrum        (inline field)
  author      lab             (inline field)
  ───[Data]───                (collapsible)
  [1.  2.]   float32 | size: 2
```

The semantic model reifies this structure as:

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

* **Semantic rendering can be introduced incrementally.**  Each object
  was migrated independently without breaking the others.

* **The renderer (`_render_sections()`) did not require hierarchical
  extensions.**  A flat list of `DisplaySection` × `DisplayItem` was
  sufficient for all four migrated objects.

* **Coord semantic items could be reused by CoordSet.**  CoordSet
  `_repr_sections()` delegates to child `Coord._repr_sections()` for
  simple (single-coordinate) dimensions, avoiding duplicated formatting.

* **CoordSet semantic sections could be reused by NDDataset.**
  NDDataset `_repr_sections()` delegates to
  `CoordSet._repr_sections()` for coordinate dimensions, keeping
  dimension rendering consistent between CoordSet and NDDataset.

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
| [`audit/~display-architecture-current-state.md`](../audit/~display-architecture-current-state.md) | State after Phase D (NDDataset migration) |
| [`audit/~project-semantic-display-migration.md`](../audit/~project-semantic-display-migration.md) | Project migration notes |
| [`audit/~semantic-display-phase-a.md`](../audit/~semantic-display-phase-a.md) | Phase A — model validation |
| [`audit/~semantic-display-phase-b.md`](../audit/~semantic-display-phase-b.md) | Phase B — Coord migration |
| [`audit/~semantic-display-phase-c.md`](../audit/~semantic-display-phase-c.md) | Phase C — CoordSet migration |
| [`audit/~semantic-display-phase-d.md`](../audit/~semantic-display-phase-d.md) | Phase D — NDDataset migration |
| [`audit/~display-html-architecture-audit.md`](../audit/~display-html-architecture-audit.md) | Initial fragmentation analysis |
| [`audit/~display-representation-model-rfc.md`](../audit/~display-representation-model-rfc.md) | Original RFC for the semantic model |
| [`src/spectrochempy/utils/print.py`](../src/spectrochempy/utils/print.py) | `DisplayItem`, `DisplaySection`, `_render_sections()`, `_html_heading()` |
