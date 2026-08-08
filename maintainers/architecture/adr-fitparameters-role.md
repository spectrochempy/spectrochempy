[Maintainer Docs](../README.md) · [Architecture Index](INDEX.md)

# Fitting Model Architecture Contract: Role of FitParameters

## Status

Accepted architecture contract.

This document records the target
architectural role of `FitParameters` at the end of the 2026-07 fitting
architecture campaign.

It started as an ADR, but the decision is broad enough to serve as the
maintainer contract for the fitting model pipeline:

```text
FitParameters is not a permanent architectural concept.
```

Future fitting architecture documents should cite this contract when deciding
whether new parser, preparation, backend, or programmatic-model work should
depend on `FitParameters`.

## Date

2026-07-09

## Decision

`FitParameters` is not a permanent domain concept in the target fitting
architecture.

It should be treated as:

- a parser-shaped internal AST;
- a legacy compatibility object;
- a migration bridge while the parser and public compatibility surface are
  moved toward the canonical fitting model representation.

It should not be treated as:

- the scientific model object;
- the backend runtime representation;
- the future public programmatic model API.

The target architecture is:

```text
DSL
    ->
Parser
    ->
Canonical Fitting Model (_FitModelSpec or successor)
    ->
prepare_model()
    ->
Backend
    ->
FitResult
```

During migration, the implementation may continue to use:

```text
DSL
    ->
Parser
    ->
FitParameters
    ->
Canonical Fitting Model (_FitModelSpec)
    ->
prepare_model()
    ->
Backend
    ->
FitResult
```

This migration pipeline is explicitly not the target architecture.

## Architecture Summary

Historical implementation:

```text
DSL
    ->
Parser
    ->
FitParameters
    ->
Optimize
```

Target architecture:

```text
DSL
    ->
Parser
    ->
Canonical Fitting Model
    ->
prepare_model()
    ->
Backend
    ->
FitResult
```

`FitParameters` has not disappeared from the implementation. It has changed
status: from runtime model object to parser AST and compatibility layer.

## Context

The fitting architecture has been progressively separated into clearer
responsibilities:

```text
DSL
    ->
Parser
    ->
FitParameters
    ->
_FitModelSpec
    ->
prepare_model()
    ->
Solver backend
    ->
FitResult
```

Recent phases established that:

- `_FitModelSpec` is the canonical fitting model representation of model
  topology, parameters, bounds, references, and fitted values;
- `prepare_model()` resolves references independently from `Optimize`;
- `getmodel()` can operate without parser naming conventions through
  component parameter views;
- the `Optimize` backend can operate on `_FitModelSpec`;
- `FitParameters` remains mainly as parser output and compatibility state.

The remaining question is conceptual:

```text
What does FitParameters represent?
```

There are three possible answers:

- parser AST;
- compatibility object;
- business/domain object.

The current architecture points to the first answer. `FitParameters` is a
parser-shaped AST with compatibility responsibilities attached to it. It is not
the fitting domain object.

## Current Responsibilities

### Parser Responsibilities

`FitParameters` currently stores parser output:

- flat parameter values in `data`;
- lower and upper bounds in parallel dictionaries;
- fixed/varying flags;
- reference flags and reference expressions;
- COMMON-block membership flags;
- model labels and model shape names;
- parser-derived flat keys such as `{parameter}_{model_label}`;
- experiment metadata (`expvars`, `expnumber`);
- sentinel values for open bounds.

It is also the current return type of `_validate_script_content(script, ...)`.

These responsibilities are parser-AST responsibilities. They describe how the
DSL was parsed, not the canonical fitting model.

### Compatibility Responsibilities

`FitParameters` currently preserves historical behavior:

- `Optimize.fp`;
- `str(fp)` script rendering;
- script round-tripping through the historical formatting rules;
- direct construction behavior covered by tests;
- dict-like parameter access for legacy internal or user code;
- compatibility fallback in `_optimize()`.

These responsibilities justify keeping `FitParameters` during migration. They
do not justify keeping it as a permanent model representation.

### Runtime Responsibilities

Historically, `FitParameters` was also runtime state:

- reference resolution through `Optimize._prepare()`;
- bounds transforms through `to_internal()` / `to_external()`;
- solver parameter vector construction;
- fitted value restoration;
- varying-parameter counting and extraction;
- model-data evaluation through parser-derived names.

Those responsibilities have been moved or made available on canonical paths:

- transforms are standalone helpers;
- reference resolution is handled by `prepare_model(spec)`;
- `_get_modeldata()` evaluates a prepared `_FitModelSpec`;
- `_optimize()` can vectorize and restore `_FitModelSpec`;
- diagnostics and result parameter extraction can use `_FitModelSpec`;
- `getmodel()` can use `_ComponentParamsView`.

Remaining runtime uses of `FitParameters` should be treated as compatibility
fallbacks, not target architecture.

### User-Visible Responsibilities

`FitParameters` is user-visible today mainly because `Optimize.fp` exists and
because `str(fp)` renders a fitting script.

That does not make it a recommended public modeling API.

The intended user-facing surfaces are:

- the fitting script DSL;
- validation helpers such as `validate_script()` and `validate_constraints()`;
- `Optimize.fit()`;
- `FitResult`;
- future public programmatic APIs, if any, built on top of the canonical
  fitting model representation rather than on `FitParameters`.

## Target Architecture Options

### Option A: Keep FitParameters in the Permanent Pipeline

```text
DSL
    ->
Parser
    ->
FitParameters
    ->
Canonical Fitting Model
    ->
Preparation
    ->
Backend
```

Consequences:

- backward compatibility is straightforward;
- parser output remains stable;
- every backend and future programmatic workflow must keep crossing a
  parser-shaped object;
- flat key encoding remains architecturally significant;
- sentinel bound conventions remain visible in core data flow;
- future APIs inherit a structure designed for parsing and script rendering,
  not for model manipulation.

This option preserves implementation history as architecture.

### Option B: Remove FitParameters from the Target Pipeline

```text
DSL
    ->
Parser
    ->
Canonical Fitting Model
    ->
Preparation
    ->
Backend
```

Consequences:

- the parser produces a canonical fitting model representation directly, or
  produces a parser-local AST that is immediately lowered to the canonical
  fitting model;
- backend code depends on one fitting model representation;
- reference resolution and model preparation are backend-independent;
- parser conventions stay near the parser;
- compatibility can be preserved by generating `FitParameters` only where
  legacy behavior requires it;
- future programmatic APIs can target the canonical fitting model instead of
  inheriting the parser AST.

This option reflects the architecture already reached by the migration.

## Public API Position

Users should not be encouraged to manipulate `FitParameters` directly.

`FitParameters` may remain accessible through `Optimize.fp` during migration for
backward compatibility, but it should be documented as legacy or internal once
a supported inspection path exists.

For documentation:

- examples should prefer scripts, validation helpers, `Optimize.fit()`, and
  `FitResult`;
- `FitParameters` should not be introduced as the way to build models;
- direct `FitParameters` manipulation should not become a new teaching path.

For assistant workflows:

- assistants should generate or inspect the DSL when working at the user level;
- assistants should reason about the canonical fitting model (`_FitModelSpec`
  today) when working at the internal architecture level;
- assistants should not construct `FitParameters` as the preferred programmatic
  modeling route.

For future programmatic APIs:

- any public model-building API should produce the canonical fitting model
  representation or a validated public facade around it;
- it should not expose `FitParameters` as the model object.

## Migration Implications

If `FitParameters` disappears from the target architecture, the parser output
should become one of:

- `_FitModelSpec` directly;
- a parser-local AST that is immediately converted to `_FitModelSpec`.

The compatibility layer that remains is:

- a generated `FitParameters` view when `Optimize.fp` or `str(fp)` compatibility
  is required;
- a legacy serializer preserving historical script formatting until canonical
  serialization is fully characterized;
- a compatibility adapter for any private tests or downstream code that still
  imports `_optimize()` with `FitParameters`.

Existing code that still requires `FitParameters`:

- `_validate_script_content()` currently returns `FitParameters`;
- the `script` trait validator assigns `self.fp`;
- constraints validation currently checks parameter names through
  `FitParameters.keys()`;
- `FitParameters.__str__` remains the historical script renderer;
- parser characterization tests still assert `FitParameters` behavior;
- `Optimize._prepare()` and `_parsing()` remain as compatibility/reference
  behavior.

Appropriate deprecation strategy:

1. Keep `FitParameters` during migration.
2. Move parser output to `_FitModelSpec` or a parser-local AST plus immediate
   lowering.
3. Replace constraints validation with canonical parameter identifiers derived
   from the model spec.
4. Replace public script rendering with a characterized canonical serializer.
5. Keep `Optimize.fp` as a legacy compatibility property until a supported
   replacement exists.
6. Only then consider documentation deprecation and runtime warnings.
7. Remove or privatize `FitParameters` only when no maintained code depends on
   it.

This strategy separates migration safety from target architecture.

## Consequences

Accepted consequences:

- `FitParameters` remains available during migration.
- The target architecture does not depend on `FitParameters`.
- Parser details stay near the parser.
- Backends consume canonical fitting model data.
- Future public APIs should be designed around canonical fitting model
  concepts, not parser artifacts.

Rejected consequences:

- `FitParameters` should not become the permanent business object.
- `FitParameters` should not become the future public model-building API.
- Backend migration should not wait for parser replacement.
- This contract does not propose builders, `lmfit` integration, DSL redesign,
  parser redesign, or new public APIs.

## Recommendation

Adopt Option B:

```text
DSL -> Parser -> Canonical Fitting Model -> prepare_model() -> Backend -> FitResult
```

Classify `FitParameters` as a parser AST and temporary compatibility layer.

This means:

- migration may keep `FitParameters` for as long as compatibility requires;
- target architecture should not route through it;
- remaining campaign work should progressively move parser output, constraints
  validation, and script rendering toward canonical fitting model data;
- the fitting business object is the canonical fitting model representation,
  not `FitParameters`.

The architecture decision is therefore:

```text
FitParameters is not a permanent architectural concept.
It is a compatibility layer around a parser AST that should eventually
disappear from the canonical fitting architecture.
```

## Implementation Report — 2026-07-09

This contract is now partially implemented in the first post-acceptance parser
migration PR.

### Parser Changes

- introduced a canonical parser entry point that parses the DSL directly into
  `_FitModelSpec`;
- kept `_validate_script_content()` as a compatibility adapter returning
  `FitParameters`;
- added `_FitModelSpec.to_fitparameters()` so legacy compatibility objects are
  generated from the canonical fitting model instead of being the parser's
  primary output;
- updated `Optimize.script` validation to store the canonical model directly
  and derive `Optimize.fp` secondarily for compatibility;
- updated constraint-name validation so it can validate against canonical model
  parameter identifiers without requiring `FitParameters`.

### Remaining Reasons For FitParameters

`FitParameters` still remains necessary for:

- `Optimize.fp` backward compatibility;
- `FitParameters.__str__` historical script rendering;
- parser characterization tests that intentionally freeze historical behavior;
- compatibility fallback paths where callers still hand `FitParameters`
  directly to older internals.

These are migration and compatibility reasons, not target-architecture reasons.

### Compatibility Strategy

The parser now conceptually follows:

```text
DSL -> _FitModelSpec -> compatibility FitParameters (only when needed)
```

The public behavior remains unchanged because:

- `_validate_script_content()` still returns `(FitParameters, errors)`;
- `Optimize.fp` is still populated;
- all parser, `_FitModelSpec`, `Optimize`, and `Optimize` result tests remain
  green.

### Remaining Migration Work

The next migration steps should focus on removing remaining *primary* uses of
`FitParameters`, not on deleting the class prematurely:

1. keep replacing internal consumers that still reconstruct canonical state
   from `Optimize.fp`;
2. reduce dependence on `FitParameters.__str__` as the authoritative script
   serializer once canonical rendering is fully characterized;
3. narrow legacy-only paths until `FitParameters` is purely a compatibility
   surface.

### Roadmap Impact

No roadmap reordering is recommended from this implementation. The accepted
sequence remains valid: move parser/runtime consumers toward canonical fitting
models first, then shrink `FitParameters` into a legacy compatibility layer.
