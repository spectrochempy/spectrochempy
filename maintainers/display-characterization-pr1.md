# Display Characterization PR1 - Maintainer Notes

**Issue:** #843 - Display / Representation Architecture
**PR Type:** Characterization and safety-net PR
**Date:** 2026-06-11
**Status:** Ready for review

## Scope

PR1 characterizes current user-visible display behavior for:

- `Coord`
- `CoordSet`
- `NDDataset`
- `Project`

The tests focus on semantic information content in:

- `repr(obj)`
- `str(obj)`
- `_repr_html_()`
- detailed text output through `pstr(obj)`

No source display behavior is changed in this PR.

## Observed Invariants

Current display output provides enough information to identify the main object
types:

- `Coord` repr output includes object type, dtype, and shape or size
  information.
- `CoordSet` repr output includes object type and coordinate names.
- `NDDataset` repr output includes object type, dtype, and shape or size
  information.
- `Project` string output includes the project name and contained object names.

Current HTML display returns non-empty output for all characterized classes.
The tests intentionally assert only broad content, such as object type or
project name, rather than exact markup.

Detailed display currently includes additional semantic content when present:

- `Coord` titles.
- `CoordSet` coordinate titles and aliases.
- `NDDataset` names, metadata, history, and coordinate information.

Empty objects and objects with `None` values display without raising.

## Current Observed Behavior

The current tests keep a small number of visible quirks because they are useful
reference points for later display work:

- compact `Coord` repr output does not show title or name values;
- `Coord`, `CoordSet`, and `NDDataset` string output currently matches repr
  output;
- compact `CoordSet` repr output does not show the `CoordSet` object's own
  name;
- compact `NDDataset` repr output does not show metadata values;
- `Project` repr currently shows object identity and differs from `str`;
- empty projects show an empty indicator;
- project children show visible type indicators for sub-projects, datasets, and
  scripts;
- project string output currently does not show project metadata.

These are observations of today's behavior, not future design constraints.

## Deliberately Out Of Scope

PR1 does not test internal display mechanisms, including:

- private display helpers;
- HTML conversion helpers;
- inheritance details;
- exact markup structure;
- exact whitespace, indentation, or CSS classes;
- private state used to format text or HTML.

The tests also avoid exhaustive coverage of every metadata field or nested
display case. They cover representative behavior only.

## Validation

Targeted validation command:

```bash
conda run -n scpy-core python -m pytest tests/test_core/test_display_characterization.py
```

If `scpy-core` is unavailable, fall back to the project virtual environment or
system Python with `PYTHONPATH=src`.
