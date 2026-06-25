# CoordSet Storage Architecture

## Status

Completed architecture note.

This document summarizes the final state of the `CoordSet` storage redesign.
It replaces the earlier tracked completion note that lived under
`maintainers/completed/`; the current maintained location is
`maintainers/architecture/coordset-storage-architecture.md`.

## Context

The redesign replaced the historical runtime model of `CoordSet` while
preserving public behavior, public APIs, copy/deepcopy behavior, and
serialization compatibility.

Historically, `CoordSet` relied on:

```python
_coords
_default
_is_same_dim
_parent_dim
```

The central problem was that `_coords` was not only a storage container.
Through the `traitlets.List` declaration and the `_coords_validate()` hook, it
also carried hidden responsibilities:

- runtime storage;
- validation;
- copy handling;
- nested `CoordSet` initialization;
- same-dimension setup;
- compatibility aliases such as `_1`, `_2`, and `_3`;
- parent-dimension propagation;
- sorting;
- empty-state coercion.

The redesign moved those responsibilities into explicit helpers and a
group-based internal architecture.

## Final Architecture

The final internal model separates three concerns:

```text
group projection
    semantic model

_storage
    runtime container

helpers
    explicit normalization and synchronization responsibilities
```

`_storage` is now the runtime coordinate container. It is a plain Python list.
The trait-based `_coords` storage model must not be reintroduced as runtime
state.

The group model is the semantic reference used by lookup, lifecycle operations,
assignment, deletion, append, and same-dimension mutation paths.

Public behavior remains centered on the existing public API:

```python
coords
names
titles
labels
units
sizes
default
default_index
is_same_dim
```

Serialization keeps the historical public state shape:

```text
coords
references
is_same_dim
default_index
name
```

## Migration Summary

The redesign was completed incrementally.

The foundation phase introduced public contract tests, extracted
read/write/delete logic, introduced the private group model, and validated copy
and serialization preservation.

The lifecycle phase migrated dimension dropping, reduction, slicing, reshaping,
replacement, concatenation, and interpolation to the group-projection pattern:

```text
legacy CoordSet
    ->
group projection
    ->
operation-specific reasoning
    ->
runtime reconstruction
```

The mutation phase migrated assignment, deletion, same-dimension mutation, and
append mutation. This removed the final active dependency on direct legacy
`_coords` mutation for normal `CoordSet` operations.

The storage switch replaced trait-based `_coords` storage with `_storage` and
removed `_coords_validate()`. Its former responsibilities were redistributed
explicitly:

- copy handling moved to insertion paths;
- nested `CoordSet` setup moved to finalization helpers;
- name validation moved to construction and finalization;
- sorting is handled explicitly where legacy behavior requires it;
- empty `CoordSet` behavior is explicit.

## Important Invariants

### Empty CoordSet

An empty `CoordSet` is a valid first-class state.

It must satisfy:

```python
len(cs) == 0
cs.is_empty is True
cs.names == []
cs.titles == []
cs.labels == []
cs.units == []
cs.sizes == []
cs.default is None
cs.default_index is None
repr(cs) == "CoordSet: []"
```

The old implicit state `_coords is None` must not be reintroduced.

### Same-Dimension CoordSets

`is_same_dim` remains explicit metadata.

It must not be inferred solely from group structure because single-entry cases
are ambiguous:

```python
CoordSet(x=coord)
CoordSet([coord])
```

These can produce equivalent group structure while preserving different
historical `is_same_dim` intent.

### Compatibility Aliases

Compatibility aliases such as `_1`, `_2`, and `_3` must remain stable. Deletion
must not renumber aliases.

For example, after deleting `_2`, aliases should move from:

```text
_1, _2, _3
```

to:

```text
_1, _3
```

### Defaults

Default selection is normalized through group metadata and synchronization
helpers.

For empty `CoordSet` objects:

```python
default is None
default_index is None
```

For deletion of the selected default, the default should move to the first
remaining entry where appropriate.

### Nested CoordSet Ordering

Nested `CoordSet` finalization preserves the effective legacy behavior.

In the old model, sorting could occur indirectly through `coord.copy()` and the
constructor path used by `_coords_validate()`. In the new model, this behavior
is explicit in the child finalization path.

## Follow-Up Items

### `_parent_dim` After Same-Dim Mutation

Direct same-dim mutation may not always restore child `_parent_dim` after
replacement or append. This appears to predate the storage switch and should be
treated as a targeted future bugfix.

Suggested characterization tests:

```python
cs.x["_1"] = Coord(...)
assert all(c._parent_dim == "x" for c in cs.x.coords)

cs.x["_3"] = Coord(...)
assert all(c._parent_dim == "x" for c in cs.x.coords)
```

A dataset-level lookup test after same-dim mutation would also be useful.

### Rename `_legacy_coordset_from_lifecycle_groups`

The helper name still contains `legacy`, although it is now part of the normal
group-to-runtime reconstruction path.

Possible future names:

```python
_coordset_from_groups
_runtime_coordset_from_groups
_reconstruct_coordset_from_groups
```

### Final Dead-Code Audit

Some fallback branches in `_resolve_set()` and `_resolve_delete()` were retained
as safety nets after the storage switch.

Do not remove them without proving that they are unreachable or redundant.

## Audit Trail

Detailed campaign history and intermediate checkpoints remain in the local
maintainer audit trail. The tracked references for maintainers are this final
architecture note and the curated architecture indexes under `maintainers/`.

### Broader Traitlets Audit

The main issue was not Traitlets itself, but business logic hidden inside
Traitlets validation hooks.

Warning signs for future audits:

```text
@validate methods that copy or reconstruct objects
@observe methods that mutate multiple fields
validators that normalize metadata
setters that silently alter object topology
```

The lesson from `CoordSet` is that validation hooks should not become hidden
semantic engines.

## Current Status

Remaining work is limited to optional follow-ups and targeted bugfixes rather
than core architecture migration.
