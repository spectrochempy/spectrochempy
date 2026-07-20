[Maintainer Docs](../README.md) · [Architecture Index](INDEX.md)

# Optimize and FitResult Architecture

## Status

Implemented architecture note.

This note is the authoritative maintainer reference for the
current architectural boundary between `Optimize` and `FitResult` after the
completed 2026-07 FitResult enrichment campaign.

## Purpose

This note answers one maintainer-facing question:

```text
What stays on Optimize,
and what becomes FitResult?
```

The boundary matters because curve fitting mixes two different concerns:

- numerical solver execution
- scientific interpretation of a completed fit

The campaign established a clean split. This note records it as current
architecture.

For the scientific contract of `FitResult` itself, see
[`fitresult-scientific-contract.md`](fitresult-scientific-contract.md).

## Core Split

The maintained architecture is:

```text
Optimize
    │
    ├── fit configuration
    ├── script / model definition
    ├── runtime fitting logic
    ├── raw solver artifacts
    │       └── jacobian
    │
    └── FitResult
            │
            ├── outputs
            ├── diagnostics
            └── uncertainty interpretation
```

This is the key decision of the campaign.

`Optimize` is the numerical engine and orchestration surface.
`FitResult` is the grouped scientific result surface.

## What Optimize Owns

`Optimize` owns:

- fitting configuration (`method`, `max_iter`, `autobase`, etc.)
- script parsing and validation
- parameter initialization and update flow
- solver dispatch
- runtime bookkeeping
- backend-specific raw solver artifacts

Most importantly, `Optimize` owns the **raw Jacobian artifact** when a backend
provides one naturally.

That Jacobian is solver state, not yet a scientific report.

### Raw Solver Artifacts

The current maintained raw artifact is:

- `opt.jacobian`

Availability reflects backend capability rather than scientific desirability.

It is available only for least-squares-backed methods that naturally provide
the required information.

It is absent (`None`) for cases such as:

- `simplex`
- `basinhopping`
- dry fits
- backends with no stable Jacobian path

This is intentional. The architecture does not fabricate or silently
reconstruct missing solver artifacts.

## What FitResult Owns

`FitResult` owns the scientific interpretation built from a completed fit.

This includes:

- output datasets (`fitted`, `residuals`, `components`)
- residual-based fit-quality diagnostics
- solver summary fields (`success`, `status`, `message`)
- uncertainty objects derived from the retained Jacobian

The important rule is:

```text
Optimize keeps raw numerical state
FitResult exposes scientific meaning
```

## Why the Jacobian Stays on Optimize

The Jacobian is a numerical artifact of the solve process.

It is:

- backend-dependent
- format-sensitive
- not always available
- useful mainly as an intermediate for later scientific interpretation

If exposed as a regular `FitResult` field, it would blur the distinction
between:

- raw solver internals
- stable user-facing scientific results

The campaign therefore kept the Jacobian on `Optimize`, while allowing
`FitResult` to expose the first stable scientific objects derived from it:

- covariance
- variance
- standard errors
- correlation
- confidence intervals

## Why Covariance Moves to FitResult

Covariance is no longer just solver state.

It is already a scientific interpretation:

- it depends on the retained Jacobian
- it depends on residual variance estimation
- it depends on the degrees-of-freedom convention

That interpretation is exactly the kind of grouped scientific result that
belongs on `FitResult`.

The same logic applies to:

- `variance`
- `stderr`
- `correlation`
- `confidence_intervals`

## Diagnostics Policy

`FitResult.diagnostics` is the grouped surface for scalar and summary result
information.

This includes two categories that are easy to confuse:

1. residual-based fit-quality summaries
2. normalized solver-summary information

Examples:

- `rss`
- `rmse`
- `r_squared`
- `degrees_of_freedom`
- `aic`
- `bic`
- `success`
- `status`
- `message`

The architecture explicitly allows solver summaries in `diagnostics` once they
have been normalized into a stable contract.

That does **not** mean the full backend result belongs there.

## Backend Diversity

The current public `Optimize.method` surface includes:

- `least_squares`
- `leastsq`
- `simplex`
- `basinhopping`

These methods do not all expose the same numerical internals.

The architecture therefore avoids pretending that every backend supports the
same post-fit scientific machinery.

Current consequence:

- residual-based diagnostics are broadly available
- covariance-style uncertainty is only available on the least-squares path

This is an honest capability boundary, not an API defect.

## Dry-Fit Policy

Dry fits are treated conservatively.

The maintained dry-fit behavior is:

- `FitResult` remains accessible
- solver success is `False`
- solver status is `None`
- solver message is `""`
- uncertainty surfaces remain unavailable

This preserves the usefulness of dry validation while avoiding false numerical
claims.

## Parameter Counting Boundary

The campaign also settled an important counting rule:

```text
the number of fitted parameters is defined by the internal parameter
representation actually optimized
```

not by a looser script-level intuition.

This rule belongs conceptually to the `Optimize` side of the boundary, because
it reflects how the numerical problem is assembled. Its stabilized result is
then surfaced through `FitResult.diagnostics["n_varying_parameters"]`.

## Future Backends

This architecture is intentionally compatible with future backend expansion.

A later campaign may:

- broaden the exposed SciPy methods
- clarify method-selection guidance
- add a plugin-based alternative backend such as `lmfit`

The maintained rule should stay the same:

- backend-specific raw artifacts remain on `Optimize`
- stable scientific interpretation remains on `FitResult`

This avoids entangling the public scientific object with whichever numerical
engine happens to be active.

## Non-goals

This note does not:

- redesign the fitting DSL
- define script ergonomics policy
- define initialization heuristics
- define constraint architecture beyond current behavior
- define a future `solution` surface
- define a plugin API for external fitting backends

Those are natural follow-up topics, but they are not part of the current
boundary contract.

## Maintainer Guidance

When adding a new fitting-related result surface, ask:

1. Is it backend-native raw numerical state?
   - keep it on `Optimize`
2. Is it a stable scientific interpretation built from solver state?
   - expose it on `FitResult`
3. Is it merely a normalized summary of backend status?
   - surface it through `FitResult.diagnostics`

This decision order should remain stable even if new backends are added.

## Next Architectural Pressure Point

Now that `FitResult` is mature, the next major architectural pressure point is
the **user-facing API of `Optimize` itself**:

- method surface
- backend selection
- DSL ergonomics
- initialization strategy
- documentation of when to use which optimizer

That should be treated as a separate audit and not as an extension of the
FitResult campaign.
