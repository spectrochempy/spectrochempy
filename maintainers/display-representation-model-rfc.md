# Display / Representation Model RFC

**Status:** Draft for discussion
**Issue:** #843 — Display / Representation Architecture
**Precedes:** PR #1116 (display characterization tests)
**Audit context:** `audit/~display-architecture-audit.md`, `audit/~display-representation-characterization-audit.md`

---

## Context

SpectroChemPy currently has a functional but fragmented display layer.

- `Coord`, `CoordSet`, and `NDDataset` share a common pipeline built on
  `_cstr()` and `convert_to_html()`, but compact text (`__repr__`) and detailed
  text (`_cstr`) are assembled independently.
- `Project` does not participate in the shared pipeline. It has no custom
  `__repr__`, uses a standalone `__str__` / `_repr_html_` path, and does not
  define `_cstr`.
- HTML output is derived by parsing `_cstr()` output through regex-based
  sentinel markers (`\0`, `\0\0`, `\0\0\0`) rather than from a semantic display
  model.
- PR #1116 (merged) added characterization tests for all four object types,
  freezing current observable behavior without changing source code.

The architecture audit identified that the display layer lacks a single semantic
source of truth for what each object should communicate to users. This RFC is
the next step: defining the intended representation model before any
implementation work.

---

## Current Baseline

These are **observed current behaviors** from PR #1116 characterization tests.
They are documented here as context for design decisions, not as constraints.

### Entry points

| Entry point | Coord | CoordSet | NDDataset | Project |
|---|---|---|---|---|
| `__repr__` | `Coord: [dtype] (size: N)` | `CoordSet: [name:title, ...]` | `NDDataset: [dtype] (size: N)` | default Python `object.__repr__` |
| `__str__` | same as `__repr__` | same as `__repr__` | same as `__repr__` | tree listing |
| `_repr_html_()` | `convert_to_html(self)` | `convert_to_html(self)` | `convert_to_html(self)` | `__str__` → space/br replacement |
| `_cstr()` / `pstr()` | shape + title + values + labels | delegated Coord display | full metadata + data + coords | not defined |

### Current information hierarchy

**Compact (`__repr__`):**
- type name, dtype, shape/size, units (when present)
- CoordSet additionally shows coordinate names and titles
- metadata (name, author, description, history) is hidden

**Detailed (`pstr` / `_cstr`):**
- NDDataset: name, author, created, modified, description, history, data, shape,
  dimensions/coordinates
- Coord: shape, title, values, labels
- CoordSet: dimension lines, delegate to Coord._cstr, aliases, nested sets
- Project: not defined

**HTML:**
- Coord, CoordSet, NDDataset: collapsible `<details>` sections wrapping
  `_cstr()` output, with CSS class-based formatting derived from sentinel
  markers
- Project: flat `<br/>` / `&nbsp;` transformation of `__str__()` output

---

## Problem Statement

The current display system works, but it has no explicit contract for what each
representation level should communicate.

Decisions about which information belongs in compact text versus detailed
display are encoded in implementation code rather than in a shared semantic
model. This makes the following activities harder than they should be:

- **Reviewing display changes** — there is no reference document to check
  whether a change violates the intended information hierarchy.
- **Adding new object types** — there is no contract dictating what `__repr__`,
  `__str__`, and `_repr_html_` should do for a new type.
- **Integrating Project** — Project is the only core type without a custom
  `__repr__` or a `_cstr` method.
- **Assessing HTML correctness** — the current HTML path relies on parsing
  text output with sentinel markers; there is no semantic HTML model.

The goal of this RFC is to answer: what should each representation level
communicate, for each core type, before we consider how to implement it?

---

## Representation Levels

SpectroChemPy objects are displayed through four entry points, each with a
different audience and context.

### Compact representation: `repr(obj)`

**Role:** Identify the object in one line. Used by the Python runtime in lists,
error messages, logging, and terminal interactive sessions.

**Current contract (implicit):**
- Must fit on one line.
- Must identify object type.
- Must communicate the essential identity: dtype, shape/size, units.
- Should not show metadata, history, or coordinate detail.

**Proposed contract:**
- Same as current, but formalised.
- Compact repr is the *object's fingerprint* — enough to recognise what the
  object is, not enough to inspect it.
- Coord: type + dtype + shape + units.
- CoordSet: type + list of coordinate names.
- NDDataset: type + dtype + shape.
- Project: type + name (adding a custom `__repr__`).

### User text representation: `str(obj)`

**Role:** Human-readable text for terminal use. Currently identical to
`__repr__` for Coord, CoordSet, and NDDataset, but different for Project.

**Open question:** Should `str(obj)` remain an alias for `repr(obj)` for
array-like objects, or should it provide a richer view?

**Options:**
1. Keep `str(obj) == repr(obj)` for Coord, CoordSet, NDDataset (current
   behavior, minimal surprise).
2. Make `str(obj)` equivalent to `pstr(obj)` (compact in terminal, detailed in
   notebook is already the HTML pattern, but terminal `str` is not the same as
   notebook `repr`).

**Proposed contract (conservative option 1):**
- `str(obj) == repr(obj)` for Coord, CoordSet, NDDataset.
- `str(obj)` remains the tree listing for Project (current behavior).

Rationale: changing `str(obj)` semantics risks surprising users who rely on
`print(obj)` giving the same output as `obj` in a REPL.

### Detailed text representation: `pstr(obj)`

**Role:** Full inspection of the object. Called explicitly by the user.

**Current contract (implicit):**
- Shows all metadata (name, author, dates, description, history for NDDataset).
- Shows coordinates and dimension information.
- Shows data values.
- Uses sentinel markers for terminal color formatting.

**Proposed contract:**
- Same information hierarchy as current, formalised.
- `pstr(obj)` is the *inspection* level — it should show everything a user
  needs to understand the object's full state.
- The sentinel-marker approach is an implementation detail. The semantic
  contract is: detailed display includes metadata, coordinates, and data.

### Notebook representation: `_repr_html_()`

**Role:** Rich display in Jupyter notebooks. The primary visual interface for
many users.

**Current contract (implicit):**
- Shows object type summary derived from `__str__()` plus object name.
- Wraps detailed content in collapsible sections.
- Parses `_cstr()` output to produce CSS-classed HTML.
- Project uses a separate, simpler path.

**Proposed contract:**
- Must produce valid HTML without raising.
- Must display the same semantic information as `pstr(obj)`, but in HTML form.
- Should use collapsible sections for metadata, data, and dimensions.
- Should eventually be governed by the same semantic model as other levels.
- The current sentinel-marker implementation can be replaced as long as the
  semantic content is preserved.

---

## Object Identity Model

Each object communicates its identity through display output. The following
attributes form the identity model:

### Type identity

Always visible at all representation levels. The type name (Coord, CoordSet,
NDDataset, Project) is the first token in `__repr__` and `__str__`, and is
visible in `_repr_html_`.

### Name

- **Coord:** Currently hidden in compact repr. Visible in detailed repr via
  `_cstr()` (as part of dimension listing in CoordSet context).
- **CoordSet:** Currently hidden in compact repr. The CoordSet's own name is
  not shown; only the names of contained coordinates appear.
- **NDDataset:** Currently hidden in compact repr. Visible in detailed repr.
- **Project:** Currently shown in `__str__` and `_repr_html_`.

**Proposed:** Name is an *identity attribute* that belongs in detailed display
for all types. Compact repr may optionally include the name for Project, but
should remain compact for Coord, CoordSet, and NDDataset.

### Title

- **Coord:** Currently hidden in compact repr. Visible in detailed repr.
- **CoordSet:** Currently shown in compact repr (as `name:title` for each
  coordinate). Visible in detailed repr.
- **NDDataset:** Not a primary attribute (name serves the role).
- **Project:** Not applicable.

**Proposed:** Title is a *descriptive attribute* that belongs in detailed
display. The current CoordSet behavior of showing titles in compact repr is a
pragmatic choice (compact coordinate identification) that the RFC should
preserve unless maintainers decide otherwise.

### Dimensions

- **Coord:** Shape is visible in compact and detailed repr.
- **CoordSet:** Dimension grouping is visible in compact repr (through
  coordinate names) and detailed repr (through expanded sections).
- **NDDataset:** Shape is visible in compact repr. Dimensions and coordinates
  are visible only in detailed repr.
- **Project:** Not applicable.

**Proposed:** Shape/dimensions are *structural attributes* that belong in
compact repr for array-like types. Full coordinate detail belongs in detailed
repr only.

### Units

- **Coord:** Visible in compact repr when present.
- **NDDataset:** Visible in compact repr when present.
- **CoordSet:** Units are visible through contained coordinate repr.
- **Project:** Not applicable.

**Proposed:** Units are *essential identity attributes* that must always be
visible for array-like objects. Units belong in compact repr.

### Metadata

- **NDDataset:** Name, author, created, modified, description, history are
  visible only in detailed repr.
- **Project:** Currently hidden (author, description not shown in `__str__`).
- **Coord / CoordSet:** Metadata is limited; the existing attributes (name,
  title) are handled above.

**Proposed:** Metadata (author, dates, description, history) belongs in
detailed display only. Compact repr should not show metadata.

### Hierarchy

- **Project:** Hierarchy is the primary content of `__str__` and `_repr_html_`.
- **CoordSet:** Nested CoordSets appear in detailed repr.
- **NDDataset:** Coordinate hierarchy appears in detailed repr.

**Proposed:** Hierarchy is a *detailed-display concern* for data objects, and
a *primary-display concern* for Project (where tree structure is the main
content).

---

## Proposed Representation Contract

### Coord

| Level | Content |
|---|---|
| `__repr__` | `Coord: [dtype] [units] (size: N)` |
| `__str__` | same as `__repr__` |
| `_repr_html_` | collapsible sections: shape, title, values, labels |
| `pstr` / `_cstr` | shape, title, values, labels |

Title and name appear only in detailed/HTML display. Units always appear.

### CoordSet

| Level | Content |
|---|---|
| `__repr__` | `CoordSet: [name1:title1, name2:title2, ...]` |
| `__str__` | same as `__repr__` |
| `_repr_html_` | collapsible sections per dimension, delegated to Coord |
| `pstr` / `_cstr` | dimension lines + delegated Coord display + aliases + nests |

The CoordSet's own name is hidden in compact repr. Coordinate names and titles
appear in compact repr for identification. Aliases and nesting appear only in
detailed display.

### NDDataset

| Level | Content |
|---|---|
| `__repr__` | `NDDataset: [dtype] [units] (shape: ...)` |
| `__str__` | same as `__repr__` |
| `_repr_html_` | collapsible sections: metadata, data, dimensions/coordinates |
| `pstr` / `_cstr` | name, author, created, modified, description, history, data, shape, dimensions/coordinates |

Metadata, history, and coordinate detail appear only in detailed/HTML display.
Compact repr shows type, dtype, units, and shape.

### Project

| Level | Content |
|---|---|
| `__repr__` | `Project: <name>` (new — currently missing) |
| `__str__` | tree listing with type indicators (current behavior) |
| `_repr_html_` | tree listing as collapsible HTML (current behavior, may be upgraded) |
| `pstr` / `_cstr` | could be added if decided (see Open Questions) |

Project is primarily a container. Its identity is its name and its contents.
Metadata (author, description) is currently hidden; this RFC does not propose
showing it unless maintainers decide otherwise.

---

## Current Behavior to Preserve

The following behaviors are considered stable user-facing contracts that should
be preserved unless the RFC explicitly decides to change them:

1. **`__repr__` identifies the object type** — every representation starts with
   the class name.
2. **`__repr__` fits on one line** — compact repr is a single-line summary.
3. **`__str__` matches `__repr__` for array-like objects** — Coord, CoordSet,
   NDDataset currently have `__str__ == __repr__`. Changing this would be
   surprising.
4. **Units are always visible in compact repr** — for array-like types with
   units, the unit string appears in `__repr__`.
5. **Metadata is hidden in compact repr** — name, author, dates, description,
   history belong in detailed display only.
6. **`_repr_html_()` never raises** — all types produce valid HTML output even
   when empty or None-valued.
7. **`pstr()` shows the full object state** — detailed display is the
   inspection level.
8. **Project `__str__` shows the tree hierarchy** — Project's primary display
   mode is container-oriented.
9. **Project shows type indicators** — sub-projects, datasets, and scripts are
   visually distinguished in tree output.
10. **Empty objects display safely** — no display method raises on empty data.

---

## Current Behavior That May Change

These behaviors are currently observed but may be changed intentionally after
the RFC is agreed:

1. **`_repr_html_` depends on sentinel markers parsed from `_cstr()` output.**
   This is an implementation detail that may be replaced with a semantic HTML
   model. The semantic content must be preserved.

2. **`_repr_html_` summary line uses `obj.__str__() + [obj.name]` for
   Coord/CoordSet/NDDataset.** The format `Coord: [dtype] (size: 3)[name]` is
   not necessarily the ideal HTML heading. This may change as long as the type,
   dtype, shape, units, and name still appear.

3. **Project has no custom `__repr__`.** The RFC proposes adding one. This is a
   pure improvement with no backward-compatibility risk.

4. **Project uses flat string replacement for `_repr_html_`.** The HTML output
   may be upgraded to use the common pipeline or CSS classes. Current tests
   should still pass if the semantic content (project name, hierarchy, type
   indicators) is preserved.

5. **Compact Coord repr does not show title or name.** This is current behavior
   and is proposed as the contract, but it is a design choice that could be
   revisited.

6. **Compact CoordSet repr shows coordinate titles.** This is current behavior
   and is proposed as the contract, but the RFC should confirm this is
   intentional.

7. **`_repr_html_` uses `convert_to_html()` which re-parses `_cstr()` output.**
   This is the most fragile part of the current architecture. It may be
   replaced with a direct semantic-to-HTML path. The observable output (type
   heading, collapsible sections, metadata, data, dimensions) must be
   preserved.

8. **Project `_repr_html_` does not use collapsible sections.** This may be
   upgraded to match the Coord/CoordSet/NDDataset pattern once Project
   participates in the common model.

---

## Open Questions

### 1. Should `str(obj)` differ from `repr(obj)` for array-like types?

Current behavior: `str(obj) == repr(obj)` for Coord, CoordSet, NDDataset.

Option A (conservative): Keep them equal. `print(obj)` and `obj` at the REPL
give the same output.

Option B (expressive): Make `str(obj)` equivalent to `pstr(obj)` for
interactive use. This would make `print(obj)` show the detailed view, but
introduce a difference between `print(obj)` and `obj` at the REPL.

**Recommendation:** Option A. The characterisation tests (PR #1116) already
assert `str(obj) == repr(obj)` for Coord, CoordSet, and NDDataset. Changing
this would require updating those tests and would risk user-facing surprise.

### 2. Should Project implement `_cstr()` and join the common display pipeline?

If yes, Project would get:
- a single source of truth for detailed display
- standard-collapsible-section HTML (via `convert_to_html()`)
- consistent metadata display rules
- `pstr(project)` support

If no, Project maintains its separate path but the decision should be explicit.

**Recommendation:** Defer. The immediate priority is agreeing the semantic
contract. Project integration can be a later PR.

### 3. Should Project metadata (author, description) appear in `__str__` / `_repr_html_`?

Current behavior: hidden. This RFC does not propose showing it. If maintainers
want Project metadata visible, that requires a separate decision.

### 4. Should the HTML pipeline be separated from `_cstr()` output parsing?

The current sentinel-marker approach works but is fragile. A future
implementation could produce HTML directly from object attributes rather than
re-parsing text. This would be a pure implementation change if the semantic
output is preserved.

### 5. Should there be a shared `__repr__` contract for all core types?

The proposed contract gives each type specific rules, but the general pattern
is: `TypeName: [essential attributes...]`. This could be formalised into a
shared guideline.

### 6. What is the minimal useful identity for a Coord?

The current answer is: type + dtype + shape + units. This seems sufficient.
The RFC proposes keeping it.

### 7. How should `_repr_html_` handle objects with no name?

Current behavior: `convert_to_html()` uses `obj.name` in the `__str__()` line
of the HTML summary. If name is None, it shows `[None]`. This is technically
correct but visually noisy. Consider showing `[unnamed]` or nothing.

### 8. Should `_repr_html_` show a different heading than `__str__()`?

Currently: `_repr_html_` heading = `__str__()` output + `[name]`. This means
the HTML heading for NDDataset is the compact repr (type, dtype, shape)
followed by the name in brackets. This may be a reasonable pattern, but it
should be explicitly confirmed.

---

## Candidate Implementation Path

The implementation path is described at the PR level. No implementation work is
being done in this RFC. The following is a suggested sequence for future PRs:

### Phase 1 (complete — PR #1116)

Characterisation tests for current display behaviour of Coord, CoordSet,
NDDataset, and Project.

### Phase 2 (this RFC)

Agree the representation model contract. Define what each level should show for
each type. Decide which current behaviors are stable and which may change.

### Phase 3: Contract tests

Add or update tests that assert the agreed contract, not just the current
implementation. These tests would replace the "current observed behavior" tests
from Phase 1 with "intended contract" tests.

### Phase 4: Project integration (if decided)

Give Project a custom `__repr__`. Optionally implement `Project._cstr()` and
migrate Project `_repr_html_` to the common pipeline.

### Phase 5: Implementation cleanup

Replace the sentinel-marker HTML parsing with a direct semantic-to-HTML path,
if the RFC concludes this is desirable. This phase must preserve the observable
output.

A shared DisplayMixin is not assumed in any phase. It may emerge as a useful
abstraction during Phase 5, but the decision to introduce it should be made
when there is concrete implementation experience.

---

## Recommended Next PR

**PR 3: Add representation contract tests.**

After the RFC is agreed, the next PR should update the display characterisation
tests to assert the agreed contract rather than just the current observed
behavior. Specifically:

- Replace "current behavior" test classes (e.g., `TestCoordCurrentBehavior`)
  with "representation contract" test classes (e.g.,
  `TestCoordRepresentationContract`).
- Add a `Project.__repr__` test that asserts the agreed format.
- Add explicit tests for the information hierarchy: which attributes appear at
  which level.
- Preserve the safety-net tests (empty objects, None values, no-raise
  guarantees).

This PR would make the contract explicit in code, not just in this RFC. After
that, implementation changes (phases 4 and 5) can proceed with confidence that
the contract is protected.

---

## Summary of Design Decisions Reached

*To be filled in after RFC review.*

| Question | Decision |
|---|---|
| Should `str(obj) == repr(obj)` for array-like types? | |
| Should Project implement `_repr__` with name? | |
| Should Project metadata be visible in display? | |
| Should Project join the common `_cstr()` pipeline? | |
| Should the HTML pipeline be separated from `_cstr()` parsing? | |
| Should there be a shared representation contract document? | |
| Is the CoordSet compact-repr showing titles acceptable? | |
| Should `_repr_html_` heading differ from `__str__()`? | |
