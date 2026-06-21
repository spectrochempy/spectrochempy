# Project Copy Semantics

**Status:** Proposed maintainer decision record

**Related issue:** `#1164`

**Prerequisites:** The core invariants RFC has been fully implemented
(PRs #1230–#1234). Copy semantics was explicitly deferred there and is the
subject of this RFC.

**Prerequisite audit:** `audit/~project-copy-semantics-audit.md`

## Motivation

The copy semantics of `Project` were deferred during the core invariants work
because:

1. The stale-parent, duplicate-policy, cycle-protection, and key/name-identity
   invariants were higher priority.
2. The copy behavior was known to be incomplete/ambiguous, but not blocking the
   core ownership model.
3. It was unclear whether `copy()` should be shallow, deep, or something in
   between.

The audit (`audit/~project-copy-semantics-audit.md`) found that:

- The current behavior is **not the result of an explicit design decision**.
  It is a side effect of how Traitlets type validators interact with
  `Project.__copy__`.
- **Datasets are duplicated** (because `NDDatasetType.validate` calls
  `NDDataset(value)`, creating a new object).
- **Subprojects are shared** (because `This()` validator returns the
  reference as-is).
- The copy's subprojects still have `parent` pointing to the **original**
  project — a stale-parent bug analogous to the one fixed by PR #1231.
- `copy.deepcopy(project)` has no custom `__deepcopy__` and its behavior is
  unpredictable.
- `Meta.__copy__` internally uses `deepcopy` even for a shallow Project copy.

This RFC defines the intended copy model so that all future fixes and tests
converge on a coherent design.

## Current Behavior

### Mechanism

`Project` defines only `__copy__` (no `__deepcopy__`):

```python
def __copy__(self):
    new = Project()
    for item in self._attributes_():   # name, meta, parent, datasets, projects
        item = "_" + item
        data = getattr(self, item)
        setattr(new, item, cpy.copy(data))
    return new
```

The divergent behavior comes from Traitlets validation of `setattr`:

| Trait | Type | Effect |
|-------|------|--------|
| `_datasets` | `tr.Dict(NDDatasetType())` | `NDDatasetType.validate` → `NDDataset(value)` → **new object** |
| `_projects` | `tr.Dict(tr.This())` | `This.validate` → value **as-is** |
| `_parent` | `tr.This()` | Value **as-is** (same reference) |
| `_meta` | `tr.Instance(Meta)` | `Meta.__copy__` → **deep-copies** `_data` |

### Result matrix

| Component | Current treatment | Parent after copy |
|-----------|-------------------|-------------------|
| Dataset children | **Duplicated** via `NDDataset(value)` | `None` |
| Subproject children | **Shared** (same object) | Original project (**stale**) |
| Parent of self | **Shared** (same reference) | Original parent |
| Metadata | **Deep-copied** | Independent |

### Invariant violations

- **Single-parent ownership** is violated for subprojects: the copy's child
  reports `parent = original_parent`, not `parent = copy`.
- **Key/name identity** is preserved by accident (duplicated datasets keep
  their name; shared subprojects keep theirs).

### Existing tests

- `TestCopy`: verifies name and datasets_names surface properties.
- `TestProjectCopyCharacterization`: documents the asymmetric behavior
  (datasets duplicated, subprojects shared) but does not test ownership
  invariants after copy.

No tests exist for:
- `copy.deepcopy` behavior
- Ownership invariants after copy
- Mixed structures (datasets + subprojects)
- Nested subprojects

## Copy Model

One model is considered. The current accidental behavior is listed for
reference but is rejected.

| Component | Current (accidental) | **Recommended: Recursive detached** |
|-----------|----------------------|-------------------------------------|
| Datasets | Duplicated (via Traitlets) | **Recursively deep-copied** |
| Subprojects | Shared (stale parent) | **Recursively deep-copied** |
| Parent of self | Shared (same reference) | **`None`** |
| Metadata | Deep-copied (already) | **Deep-copied** (unchanged) |
| `copy == deepcopy` | No `__deepcopy__` | **`copy(x) == deepcopy(x)`** |

### Recommended model: recursive detached copy

`project.copy()` returns a new project that is a **fully independent,
recursively copied** subtree. Every child — dataset, subproject,
sub-subproject, and so on — is a new object. The copy shares nothing with
the original.

```
copy(x) == deepcopy(x)
```

| Component | Treatment |
|-----------|-----------|
| Datasets | Deep-copied via `cpy.deepcopy()` or their own `copy(deep=True)` |
| Subprojects | Recursively deep-copied — each nested project is itself copied with the same model |
| Parent | Always `None` — the copy is a standalone root project, regardless of whether the original was a root or a subproject |
| Metadata | Deep-copied via `Meta.__copy__` (already deep — keep as-is) |

**Key properties:**

1. **Full structural isolation.** Mutating the original after copy never
   affects the copy, and vice versa.
2. **Ownership invariants preserved.** Every child in the copy has exactly
   one parent (the copy or its recursive parent inside the copy). No stale
   references, no aliasing.
3. **Root detachment.** The copy is always a root project (`parent is None`).
   Re-attaching it elsewhere is an explicit `add_project` call by the user.
4. **Unified shallow/deep.** `copy(x) == deepcopy(x)` — both produce the
   same fully independent result. `__deepcopy__` is added explicitly and
   delegates to the same internal method as `__copy__`.

### Non-recommended model

**Current accidental behavior (datasets duplicated via Traitlets side effect,
subprojects shared with stale parent)** is rejected. It is not an intentional
design, it violates ownership invariants, and it is confusing.

## Subproject Semantics

Subprojects are **recursively deep-copied**. The copy of a project copies
its subprojects, which in turn copy their subprojects, and so on. Every
project node in the tree is a new, independent instance.

- Ownership: each node's `parent` points to its correct parent inside the
  copy (or `None` for the root of the copy).
- Mutations: modifying any node in the original tree has no effect on the
  copy, and vice versa.
- Depth: unbounded — the entire tree is copied. There is no non-recursive
  special case.

### Why recursive?

Project is an **ownership hierarchy**, not a flat dict. Copying the
container without copying the children is equivalent to copying a directory
without copying its contents. The ownership invariants (single-parent,
acyclic, key/name identity) apply to every node in the tree. A non-recursive
copy would either:

- Share children (stale-parent violation), or
- Copy children but leave their children shared (inconsistent depth).

Recursive copy is the only model that preserves all invariants at every depth
without special cases.

## Parent Semantics

`copy.parent is None` unconditionally.

| Original | Copy's parent |
|----------|---------------|
| Root project (`parent is None`) | `None` |
| Subproject (`parent is proj`) | `None` |

The copy is a standalone project. If the user wants it re-attached, they
call `parent_project.add_project(copied)` explicitly.

This avoids the current stale-parent bug (the copy's subprojects currently
point to the original parent) and gives the cleanest invariant:

> A copied Project is always a root Project until explicitly inserted into
> a parent.

## deepcopy Semantics

`copy.deepcopy(project) == copy.copy(project)`.

`Project` has no current `__deepcopy__`. Python's default behavior for
objects that define `__copy__` but not `__deepcopy__` is unpredictable with
traitlets `HasTraits`. The RFC treats this as a bug.

The recommended model defines both `__copy__` and `__deepcopy__` and makes
them delegate to the same internal method. Rationale:

- The Project container does not own deep resources that require distinct
  shallow/deep treatment. Its children are already fully copied.
- Python convention is that `deepcopy` deeply copies **contained elements**
  while `copy` may share them. For Project, all elements are copied anyway;
  there is no semantic distinction to express.
- The simplest contract is equality: `copy(x) == deepcopy(x)`.

## Metadata Semantics

`Meta.__copy__` already uses `copy.deepcopy(self._data)` internally. This
is correct for the recommended model and is kept unchanged.

Metadata is fully independent after copy — no aliasing of metadata content
between original and copy.

## Alternatives Considered

### Shallow copy (Model A)

`project.copy()` returns a new project with shared children. Discussed in
the audit as Option A.

**Rejected.** Shared mutable children create aliasing surprises and fail to
preserve ownership invariants. Project is an ownership container, not a
`dict`-like view.

### Non-recursive detached copy (Model B from audit)

`project.copy()` copies direct children but does not recurse into
subprojects.

**Rejected.** Incomplete isolation. A non-recursive copy creates an
inconsistent depth model: the copy's subprojects share their own children
with the original. This is effectively the same stale-parent problem, one
level down.

### Copy-on-write (COW)

Children are shared until one of them is modified.

**Rejected for this RFC.** COW is substantially more complex than the current
problem warrants. It can be revisited if profiling shows that copy
performance is a bottleneck in real workflows.

### Serialization-based copy

Copy by save-to-buffer then load.

**Rejected.** Fragile, slow, mixes persistence with copy semantics.

### keepname parameter

A `keepname` parameter like `NDDataset.copy(keepname=...)`.

**Deferred.** Can be added later if user feedback justifies it. Does not
affect the core model.

## Recommendation

### Recommended model: recursive detached copy

```
copy(project) == deepcopy(project)
```

| Component | Treatment |
|-----------|-----------|
| Datasets | Recursively deep-copied |
| Subprojects | Recursively deep-copied |
| Parent | `None` (standalone root) |
| Meta | Deep-copied (already, via `Meta.__copy__`) |
| `__deepcopy__` | Explicit, delegates to `__copy__` |

**Rationale:**

1. **Ownership correctness.** Every child has exactly one parent (the copy
   or its recursive parent inside the copy). No stale references, no
   aliasing.
2. **Full isolation.** Mutating original or copy never affects the other.
   This is what users in scientific Python expect from `.copy()` (numpy,
   xarray, pandas).
3. **Simplicity of contract.** `copy(x) == deepcopy(x)` is the easiest rule
   to document, test, and reason about. No special cases for shallow vs.
   deep.
4. **Tree model consistency.** Project is an ownership hierarchy; recursive
   copy is the only model that preserves invariants at every depth.

### Implementation notes (for future PR, not implemented here)

The implementation must **bypass** the Traitlets validation trap that
currently creates the accidental asymmetry. Instead of relying on `setattr`
to trigger type validators, the copy should:

1. Create a new `Project()`.
2. Copy scalar attributes (`_name`, `_meta`) directly.
3. Set `_parent` to `None` directly.
4. For `_datasets`: iterate the original dict and deep-copy each value,
   then attach via `new.add_dataset(cpy.deepcopy(ds), name=key)` or
   equivalent.
5. For `_projects`: iterate the original dict and recursively copy each
   value (same model — the subproject's own `__copy__` is called), then
   attach via `new.add_project(copied, name=key)`.

This ensures the copy path is explicit, recursive, and not dependent on
Traitlets validation side effects.

### What this RFC does NOT decide

- `copy(keepname=...)` parameter — deferred as optional UX improvement.
- Serialization implications — copy and persistence are separate concerns.
- Result object membership in Project — deferred to the Result campaign.

## Follow-Up Work

### Required before implementation

1. **Adopt this RFC** — accept or amend the recommended model.
2. **Implementation PR** — rewrite `Project.__copy__` and add
   `__deepcopy__` following the recursive detached model.
3. **Update characterization tests** — replace
   `TestProjectCopyCharacterization` with tests that verify the new behavior.
4. **Add ownership tests** — verify that after copy:
   - Every child's `parent` is coherent inside the copy (`None` for root
     of the copy, the correct parent project for nested nodes).
   - Mutating the copy's children does not affect the original.
   - `copy.deepcopy(project)` produces the same result as
     `copy.copy(project)`.
   - The copy is a valid project: no duplicates, acyclic, key/name
     identity preserved.

### Optional follow-up

5. **Add `keepname` parameter** — if user feedback justifies it, as a
   convenience option.
6. **Performance benchmarking** — verify that recursive deep copy is not
   prohibitive for typical project sizes.

```
Recommended implementation path: recursive detached copy (copy == deepcopy).
```
