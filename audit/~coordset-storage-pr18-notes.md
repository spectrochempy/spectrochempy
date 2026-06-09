# PR18 — `_interpolate_dim` migration

## Goal

Migrate `CoordSet._interpolate_dim()` to the lifecycle adapter pattern
(projection → lifecycle reasoning → legacy reconstruction) while preserving
all public behavior.

## Behavior Characterization

The `_interpolate_dim` operation is invoked by
`spectrochempy.interpolate()` during coordinate interpolation.  The original
implementation (before PR18):

1. Copied the entire coordset: `result = self.copy()`
2. If `dim not in result.names`: appended a new coord from `target_coord` copy
   (labels cleared, name set to dim)
3. Otherwise, located `coord_idx = result.names.index(dim)` and retrieved
   `old_container = result._coords[coord_idx]`
4. **Simple coord case** (not a CoordSet): replaced with `target_coord.copy()`,
   labels cleared, name set to dim
5. **Same-dim multi-coord case** (CoordSet): applied `interpolate_secondary`
   callable to each sub-coord, created new CoordSet with reversed children
   (`[::-1]`), preserved `_default` and `_is_same_dim`
6. Returned the modified copy

### Group-level behaviors

| Case | Legacy action | Lifecycle equivalent |
|------|--------------|---------------------|
| Dim not found | Append new coord at end of `_coords` | Append new group for dim |
| Simple coord | Replace coord at position | Replace single-entry coord in group |
| Same-dim multi-coord | Apply `interpolate_secondary` to each `_coords[i]`, reverse, reconstruct CoordSet | Apply `interpolate_secondary` to each entry's coord, update entries |

### Secondary interpolation semantics

The `interpolate_secondary` callable (defined in the outer `interpolate()`
function):
1. Copies the coord
2. Interpolates its numeric data from the old grid to the new grid
   (using `interp1d` with the old primary coordinate values as x-axis)
3. Clears labels
4. Returns the new coord

For same-dim multi-coord groups, `interpolate_secondary` is applied to ALL
entries (including the primary/default entry).  The `target_coord` parameter
is only used for simple coord replacement (non-same-dim) or when appending
a new dimension.

## Helpers Introduced

### `_interpolate_lifecycle_groups` (static method)

Located at `coordset.py:1108-1146`.

Process:
1. Locates the target group by matching `dim` against non-reference group dims
2. **Dim not found**: creates a new group from `target_coord.copy()` (labels
   cleared) via `_coordset_to_groups` and appends it to the group tuple
3. **Simple coord** (single entry, no aliases): replaces the entry's coord
   with `target_coord.copy()` (labels cleared)
4. **Same-dim multi-coord** (multiple entries or aliases): applies
   `interpolate_secondary` to each entry's coord via
   `replace(entry, coord=interpolate_secondary(entry.coord))`

### Wrapper in `_interpolate_dim`

Located at `coordset.py:1094-1117`.

```python
groups = self._lookup_groups()
groups = self._interpolate_lifecycle_groups(
    groups, dim, target_coord, interpolate_secondary,
)
return self._legacy_coordset_from_lifecycle_groups(groups)
```

This follows the exact same pattern as all other lifecycle-migrated methods.

## Group Metadata Consumed

- `_DimensionCoordinates.dim` — matched against the interpolation target
- `_DimensionCoordinates.entries[i].coord` — per-entry coord is replaced
  with either `target_coord.copy()` (simple coord) or
  `interpolate_secondary(entry.coord)` (same-dim multi-coord)
- `_DimensionCoordinates.reference` — reference groups pass through unchanged

## Reconstruction Strategy

Uses the existing `_legacy_coordset_from_lifecycle_groups` which delegates
to `_groups_to_coordset`.  No new reconstruction code was added.

`_groups_to_coordset` handles:
- Single-entry groups (no aliases) → simple Coord, name set from group dim
- Multi-entry or aliased groups → nested CoordSet with `_is_same_dim=True`,
  `_default` from `group.default_id`, aliases from `group.aliases`,
  `_parent_dim` set from group dim

## Same-Dimension Handling

Same-dimension CoordSets (multi-coordinate axis) are handled transparently:
- `_coordset_group_to_dimension` creates a multi-entry `_DimensionCoordinates`
- `_interpolate_lifecycle_groups` applies `interpolate_secondary` to each entry
- `_groups_to_coordset` reconstructs the multi-entry CoordSet with
  `_is_same_dim=True`, preserving `_default`, aliases, and `_parent_dim`

## Metadata Propagation

| Metadata | Path | Status |
|----------|------|--------|
| Coord name | Set by `_groups_to_coordset` from group dim | ✅ |
| Coord labels | Cleared by `interpolate_secondary` or explicit `_labels = None` | ✅ |
| Coord data | Preserved through `interpolate_secondary` or `target_coord.copy()` | ✅ |
| `_is_same_dim` | Set by `_groups_to_coordset` | ✅ |
| `_default` | Preserved via `group.default_id` match | ✅ |
| `_parent_dim` | Set by `_groups_to_coordset` via `nested._set_parent_dim(group.dim)` | ✅ |
| Aliases | Preserved via `group.aliases` and `_set_names` | ✅ |
| References | Groups with `reference` set are passed through unchanged | ✅ |
| Titles/Units | Preserved through `coord.copy(keepname=True)` | ✅ |

## Preservation Guarantees

| Behavior | Status |
|----------|--------|
| Simple coord replacement (data, labels, name) | ✅ preserved |
| Same-dim multi-coord reconstruction | ✅ preserved |
| Secondary coord interpolation | ✅ preserved |
| Empty coords (no crash) | ✅ preserved |
| Dim-not-found append | ✅ preserved |
| Non-first default index | ✅ preserved |
| `_is_same_dim` flag | ✅ preserved |
| Alias names | ✅ preserved (via group metadata round-trip) |
| Reference coordinates | ✅ preserved (groups with reference are skipped) |
| Title/unit propagation | ✅ preserved |
| `_parent_dim` consistency | ✅ preserved |

## Tests Run and Results

### Interpolation tests: **26 passed**

```bash
conda run -n scpy python -m pytest \
  tests/test_processing/test_interpolation/test_interpolate.py -v
```

All existing tests pass without modification.

### CoordSet tests: **126 passed**

```bash
conda run -n scpy python -m pytest \
  tests/test_core/test_dataset/test_coordset.py \
  tests/test_core/test_dataset/test_coordset_behavior.py -v
```

No regression in CoordSet behavior.

## Key Design Decisions

1. **Why `interpolate_secondary` in the lifecycle helper, not the wrapper?**
   The secondary interpolation operates on projected group entries
   (per-sub-coord), so it belongs in the lifecycle helper.  Simple coord
   replacement and dim-not-found appending also happen at the group level.

2. **Why not also concatenate per-sub-coord data?** N/A — interpolation
   does not concatenate data.

3. **Why use `_coordset_to_groups` to create the new group for missing dims?**
   This avoids importing `_CoordinateEntry` or `_DimensionCoordinates`
   directly, following the same pattern as `_replace_lifecycle_coord`.

## Lifecycle Migration Completeness

All six dimension-manipulation methods have now been migrated to the
lifecycle adapter pattern:

| Method | PR | Status |
|--------|----|--------|
| `_drop_dims()` | PR12 | ✅ |
| `_reduce_dim()` | PR13 | ✅ |
| `_slice_dims()` | PR14 | ✅ |
| `_reshape_dims()` | PR15 | ✅ |
| `_replace_dim()` | PR16 | ✅ |
| `_concatenate_dim()` | PR17 | ✅ |
| `_interpolate_dim()` | PR18 | ✅ |

## Recommendation for Post-Lifecycle Architecture Phase

With PR18, the lifecycle migration phase is **complete**. All dimension
manipulation operations now follow the established pattern:

```
legacy CoordSet
  → group projection (_lookup_groups)
  → lifecycle reasoning (_*_lifecycle_groups)
  → legacy reconstruction (_legacy_coordset_from_lifecycle_groups)
```

The next architectural phase should be a **dedicated post-lifecycle audit**
to:

1. Assess whether group projections can be made persistent (replacing
   `_coords` as the runtime source of truth).
2. Evaluate the cost/benefit of removing legacy runtime storage
   (`_coords`, `_default`, `_is_same_dim`, `_parent_dim`) in favor of
   group-only storage.
3. Plan serialization migration to match the new storage model.
4. Identify remaining dead code paths (e.g., the `_data` attribute on
   CoordSet).
5. Assess mutation-layer changes needed for write/delete operations on
   the new storage model.
