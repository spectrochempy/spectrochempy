[Maintainer Docs](../README.md) · [Architecture Index](INDEX.md)

# Canonical DSL Contract

## Status

Implemented architecture note.

This note is the authoritative maintainer reference for the role of the
fitting DSL within the post-migration Canonical Fitting Model architecture.

This note does not redesign the DSL and does not specify full parser syntax.

It defines only one thing:

```text
What does the fitting DSL guarantee as an interchange format?
```

## Purpose

The fitting DSL is a serialization and interchange format for the Canonical
Fitting Model.

Its architectural role is to support:

- human-readable model declaration;
- interchange between users, notebooks, assistants, and compatibility layers;
- serialization of fitting-model semantics into text;
- parsing of text back into the Canonical Fitting Model.

It is therefore part of the following maintained picture:

```text
Canonical Fitting Model
    ->
Canonical DSL serialization
    ->
Text
```

and also:

```text
Text
    ->
Parser
    ->
Canonical Fitting Model
```

The crucial boundary is:

- the Canonical Fitting Model is the semantic center of the architecture;
- the DSL is one text-based interchange surface around that model.

The DSL is **not**:

- the canonical model itself;
- the unique way to define fitting models;
- a lossless textual representation of all authoring details;
- the concept that defines the fitting architecture.

This matches the broader model-centered architecture:

```text
Input mechanisms
    ->
Canonical Fitting Model
    ->
prepare_model()
    ->
Backend
    ->
FitResult
```

## Guarantees

As an interchange format, the DSL should guarantee semantic preservation of
fitting-model declaration.

The maintained guarantees are:

### 1. Model Declaration Preservation

The DSL preserves the declared fit problem at the level of model semantics.

That includes, for the maintained supported surface:

- component identity and labels;
- component model names;
- parameter declarations;
- fixed versus varying state;
- lower and upper bounds;
- COMMON parameter declarations;
- declared references;
- other declarations that belong to the Canonical Fitting Model contract.

This is the primary guarantee of the DSL.

### 2. Round-Trip Through the Canonical Fitting Model

The DSL should support semantic round-trip through the Canonical Fitting
Model:

```text
text
    ->
parser
    ->
Canonical Fitting Model
    ->
serializer
    ->
text
```

The architectural expectation is not textual identity.

The expectation is that reparsing the serialized text yields the same model
declaration, within the maintained semantic contract.

### 3. Canonical-Model Interchange

The DSL is a valid interchange surface between different model clients:

- users writing scripts;
- assistants showing or exchanging fitting models as text;
- compatibility layers that still require textual form;
- future tooling that needs a human-readable serialization.

That means the DSL is guaranteed to carry model meaning across these
boundaries, not to preserve incidental formatting choices.

### 4. Separation From Execution

The DSL guarantees model declaration interchange, not prepared runtime state
or fit results.

It is not expected to encode:

- prepared reference-resolved values as an architectural contract;
- backend vectorization state;
- optimization diagnostics;
- fit-result interpretation.

Those belong to other layers.

## Non-Guarantees

The DSL intentionally does not guarantee lossless textual reproduction.

The following are outside the DSL contract unless explicitly promoted later:

### 1. Whitespace and Indentation

Whitespace, indentation, and line-layout style are not semantically relevant
and are not guaranteed to be preserved across round-trip.

### 2. Comments

Comments are not part of the Canonical Fitting Model declaration and are not
guaranteed to survive parse/serialize round-trip.

The DSL is a semantic interchange format, not a comment-preserving source
format.

### 3. Formatting Style

The DSL does not guarantee preservation of:

- numeric formatting style;
- spacing style;
- alignment choices;
- historical renderer quirks.

These are serializer choices unless explicitly elevated to contract status.

### 4. Textual Identity

Byte-identical round-trip is not guaranteed.

Two DSL texts may differ while declaring the same Canonical Fitting Model.

This is expected behavior under a semantic interchange contract.

### 5. Insertion Order When Semantically Irrelevant

Ordering that is not part of model semantics is not guaranteed to survive.

If a serializer chooses a canonical or normalized order for declarations that
are semantically unordered, that is compatible with the DSL contract.

### 6. Historical Compatibility Formatting

Legacy formatting artifacts from compatibility renderers are not part of the
architectural DSL contract merely because they exist today.

They may remain temporarily for compatibility, but they do not define the
maintained semantics of the format.

## Canonical Serialization

`_FitModelSpec.to_script()` is the natural candidate to become the canonical
DSL serializer because it serializes directly from the Canonical Fitting Model
rather than from a compatibility object.

Architecturally, that is the correct direction.

However, it should become the authoritative serializer only when the following
conditions are satisfied:

- semantic round-trip is characterized for the maintained DSL surface;
- ordering and formatting policy are understood well enough to be treated as
  canonical serializer behavior rather than incidental implementation detail;
- any remaining compatibility-only differences with `FitParameters.__str__()`
  are explicitly classified as either acceptable non-guarantees or temporary
  blockers;
- maintainers are comfortable declaring that the DSL is modeled from the
  Canonical Fitting Model first, not from `FitParameters`.

The important architectural point is:

```text
Canonical serializer means authoritative serializer of model semantics,
not lossless reproducer of historical text.
```

So the correct success criterion for canonical serialization is not byte
identity with historical output. It is authoritative semantic interchange from
the Canonical Fitting Model.

## Compatibility

`FitParameters.__str__()` remains temporarily important for compatibility.

Its temporary role is to support:

- existing compatibility surfaces such as `Optimize.fp`;
- historical behavior expectations;
- legacy internal or downstream paths still shaped around `FitParameters`.

But this compatibility role does not define the architectural DSL contract.

`FitParameters.__str__()` is a compatibility renderer, not the conceptual
center of the DSL architecture.

Its existence answers a migration need.
It does not answer the long-term architectural question of what the DSL is.

## Future Evolution

The DSL fits naturally into a model-centered multi-serializer architecture.

Future serializers may include:

- DSL text;
- JSON;
- YAML;
- structured dictionaries;
- other interchange or storage forms.

These formats do not compete with the Canonical Fitting Model.
They orbit around it.

The maintained long-term picture is therefore:

```text
Canonical Fitting Model
    ->
multiple serializers / interchange formats
    ->
multiple external representations
```

This means future evolution should be evaluated by asking:

- does this format preserve model declaration semantics?
- is it a serialization concern rather than a model-definition concern?

If yes, it fits naturally into the architecture.

## Summary

The Canonical DSL Contract is:

- the DSL is a semantic serialization and interchange format of the Canonical
  Fitting Model;
- it guarantees preservation of model declaration semantics;
- it does not guarantee lossless preservation of text presentation details;
- `FitParameters.__str__()` remains a compatibility renderer, not the
  architectural definition of the format;
- `_FitModelSpec.to_script()` is the natural future canonical serializer when
  its semantic authority is fully characterized.

The decisive distinction is:

```text
The DSL is a semantic interchange format, not a lossless textual
representation.
```
