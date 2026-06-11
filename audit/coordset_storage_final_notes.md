# CoordSet Storage Redesign — Final Architecture Summary

## Context

The CoordSet storage redesign was a multi-PR refactoring intended to replace the historical runtime model of `CoordSet` while preserving public behavior, public APIs, copy/deepcopy behavior, and serialization compatibility.

Historically, `CoordSet` relied on:

```python
_coords
_default
_is_same_dim
_parent_dim
```

The central problem was that `_coords` was not only a storage container. Through the `traitlets.List` declaration and the `_coords_validate()` hook, it also carried hidden responsibilities:

* runtime storage;
* validation;
* copy handling;
* nested `CoordSet` initialization;
* same-dimension setup;
* compatibility aliases (`_1`, `_2`, ...);
* parent-dimension propagation;
* sorting;
* empty-state coercion.

This made the class fragile because important semantic behavior was hidden inside an implicit validator rather than expressed in explicit helpers.

The redesign progressively moved CoordSet behavior toward a group-based internal architecture.

## Final architecture

After the redesign, the internal model is separated more clearly:

```text
group projection
    semantic model

_storage
    runtime container

helpers
    explicit normalization / synchronization responsibilities
```

The previous trait-based `_coords` runtime storage has been replaced by `_storage`, a plain Python list.

The group model is now the semantic reference used by lookup, lifecycle operations, assignment, deletion, append, and same-dimension mutation paths.

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

## Migration summary

The redesign was completed incrementally.

### Foundation phase

The early PRs introduced contract tests, extracted read/write/delete logic, introduced the private group model, and validated preservation of copy and serialization behavior.

### Lifecycle phase

Lifecycle operations were migrated to the group-projection pattern:

```text
legacy CoordSet
    ->
group projection
    ->
operation-specific reasoning
    ->
legacy/runtime reconstruction
```

This covered operations such as dimension dropping, reduction, slicing, reshaping, replacement, concatenation, and interpolation.

### Mutation phase

Mutation paths were then migrated:

* assignment mutation;
* deletion mutation;
* same-dim mutation;
* append mutation.

This was the critical phase that removed the final active dependency on direct legacy `_coords` mutation for normal CoordSet operations.

### Storage switch phase

The runtime storage switch replaced the trait-based `_coords` storage with `_storage`.

The `_coords_validate()` validator was removed and its former responsibilities were redistributed explicitly:

* copy handling moved to insertion paths;
* nested `CoordSet` setup moved to a dedicated finalization helper;
* name validation moved to construction/finalization;
* sorting is handled explicitly where legacy behavior required it;
* empty `CoordSet` behavior was made explicit.

## Important invariants

### Empty CoordSet

An empty `CoordSet` is now a valid first-class state.

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

The old implicit state:

```python
_coords is None
```

must not be reintroduced.

### Same-dimension CoordSets

`is_same_dim` remains explicit metadata.

It must not be inferred solely from group structure because single-entry cases are ambiguous:

```python
CoordSet(x=coord)
CoordSet([coord])
```

can produce equivalent group structure while preserving different historical `is_same_dim` intent.

### Aliases

Compatibility aliases such as `_1`, `_2`, `_3` must remain stable.

Deletion must not renumber aliases.

For example:

```text
_1, _2, _3
```

after deleting `_2` should become:

```text
_1, _3
```

### Defaults

Default selection is normalized through group metadata and synchronization helpers.

For empty `CoordSet` objects:

```python
default is None
default_index is None
```

For deletion of the selected default, the default should move to the first remaining entry where appropriate.

### Nested CoordSet ordering

Nested `CoordSet` finalization preserves the effective legacy behavior.

In the old model, sorting could occur indirectly through `coord.copy()` and the constructor path used by `_coords_validate()`.

In the new model, this behavior is explicit in the child finalization path.

## Points to revisit later

### 1. `_parent_dim` after same-dim mutation

An independent audit noted that direct same-dim mutation may not always restore child `_parent_dim` after replacement or append.

This appears to predate the storage switch and should be treated as a future targeted bugfix, not as part of the storage redesign.

Suggested tests:

```python
cs.x["_1"] = Coord(...)
assert all(c._parent_dim == "x" for c in cs.x.coords)

cs.x["_3"] = Coord(...)
assert all(c._parent_dim == "x" for c in cs.x.coords)
```

A dataset-level lookup test after same-dim mutation would also be useful.

### 2. Rename `_legacy_coordset_from_lifecycle_groups`

The helper name still contains `legacy`, although it is now part of the normal group-to-runtime reconstruction path.

This is not urgent because the helper has many callers, but it should eventually be renamed to better reflect its current role.

Possible future names:

```python
_coordset_from_groups
_runtime_coordset_from_groups
_reconstruct_coordset_from_groups
```

### 3. Final dead-code audit

Some fallback branches in `_resolve_set()` and `_resolve_delete()` were intentionally retained as safety nets after the storage switch.

A future cleanup pass could re-evaluate whether they are still needed after more release-cycle exposure.

Do not remove them without proving that they are unreachable or redundant.

### 4. Broader Traitlets audit

The main issue was not Traitlets itself, but business logic hidden inside Traitlets validation hooks.

A future audit could look for other long or side-effect-heavy validators/observers in the codebase.

Particular warning signs:

```text
@validate methods that copy or reconstruct objects
@observe methods that mutate multiple fields
validators that normalize metadata
setters that silently alter object topology
```

The lesson from `CoordSet` is that validation hooks should not become hidden semantic engines.

### 5. Display / math metadata parallels

This redesign followed the same architectural pattern identified in other audits:

```text
multiple construction paths
    ->
semantic fragmentation
```

Similar risks may exist in:

* mathematical metadata propagation;
* display / representation architecture;
* object serialization boundaries.

Future refactorings should first identify competing sources of truth before modifying behavior.

## Current status

Remaining work is limited to optional follow-ups and targeted bugfixes rather than core architecture migration.
