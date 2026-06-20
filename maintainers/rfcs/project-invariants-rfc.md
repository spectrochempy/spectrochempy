# Project Invariants and Ownership Semantics

**Status:** Proposed maintainer decision record

**Related issue:** `#1164`

## Motivation

The Project architecture audit and the invariants characterization reached the
same conclusion: `Project` remains useful, but several historical behaviors are
still implicit, inconsistent, or plainly wrong.

This RFC does not redesign `Project`. It only defines the minimum invariants
that make its current role coherent and safe to maintain.

The goal is to answer one question:

> What are the fundamental invariants of `Project`?

## Ownership

`Project` is a lightweight hierarchical owner of child objects.

The ownership contract is:

1. A child has at most one parent.
2. A child cannot exist in several `Project` instances at the same time.
3. If `child.parent is project`, that parent must contain the child.
4. A child must never retain an obsolete parent reference after replacement,
   removal, or move.
5. Moving a child between projects is an ownership transfer, not a shared
   reference.

This applies to both datasets and subprojects.

The intended model is therefore strict single-parent ownership, not shared
membership and not graph-style attachment.

## Cycle Policy

`Project` is an acyclic hierarchy.

The following are forbidden:

* self-insertion;
* insertion of an ancestor as a descendant;
* any indirect cycle across multiple subprojects.

Any operation that would create a cycle must fail.

This is a correctness requirement, not an optional UX choice. Cycles break the
tree model and invalidate display, traversal, and persistence assumptions.

## Duplicate Policy

Current behavior is inconsistent:

* datasets auto-rename;
* subprojects silently overwrite.

This RFC chooses a single future policy:

> Duplicate child keys must raise an explicit error for both datasets and
> subprojects.

### Why not auto-rename everywhere?

Auto-renaming is convenient, but it silently mutates child identity and makes
membership less explicit.

### Why not keep overwrite for subprojects?

Silent overwrite is not acceptable for an ownership container because it can
discard membership while leaving stale parent state behind.

### Chosen direction

Explicit failure is the smallest and clearest invariant:

* no silent rename;
* no silent overwrite;
* no accidental identity mutation;
* no ambiguity about which child is attached.

## Key and Name Identity

Within a `Project`, the membership key is the authoritative identifier of a
child entry.

The invariant is:

> After insertion, replacement, or load, `project_key == child.name`.

This means:

* a child stored under key `"sample"` is the child whose `name` is `"sample"`;
* `Project` is not expected to maintain divergent key/name identities;
* implementations may synchronize either side, but the observable contract is
  equality while the child is attached.

This RFC does not define the technical mechanism for synchronization. It only
defines the semantic contract.

## Deferred Topics

Two topics are intentionally deferred.

### Copy

Current `copy()` behavior is shallow and shares children whose parent pointers
still refer to the original project. This RFC records that the current
behavior needs clarification, but it does not redefine `copy()`.

### Root name

Current save/load behavior may overwrite the root project name from the
filename. This RFC records the question but does not decide it here.

Both topics should be handled separately after the core ownership invariants
are enforced.

## Minimal Project Model

`Project` remains:

> a lightweight hierarchical dataset container

More precisely, it is an optional typed hierarchy for:

* `NDDataset` leaves;
* nested `Project` nodes;
* project metadata;
* native project persistence.

`Project` is not:

* a workspace;
* a workflow engine;
* a provenance graph;
* a generic object store.

This keeps the future scope aligned with issue `#1164` and with the Project
architecture audit: preserve the useful hierarchy, avoid reopening broad
product-design questions.

## Follow-Up Work

Recommended follow-up PR sequence

1. Add focused invariant tests for replacement, moves, duplicate handling, and
   cycle rejection.
2. Fix stale parent handling so replacement, removal, and moves cannot leave
   obsolete parent references.
3. Align duplicate handling to explicit error behavior for both datasets and
   subprojects.
4. Add cycle protection at all public attachment boundaries.
5. Enforce or restore the `key == child.name` invariant consistently across
   insert, replace, move, and load paths.
6. Revisit deferred topics (`copy()`, root-name semantics) in separate, narrow
   follow-up RFCs or PRs if still needed.

## Conclusion

The core `Project` contract is intentionally narrow:

* single-parent ownership;
* acyclic hierarchy;
* explicit duplicate rejection;
* stable key/name identity;
* lightweight hierarchical dataset-container scope.

These invariants are sufficient to stabilize `Project` without redesigning it.
