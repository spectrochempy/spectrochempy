[Maintainer Docs](../README.md) · [Architecture Index](INDEX.md)

# Framework Principles

## Status

Architecture reference.

## Date

2026-07-02

## Scope

Philosophical foundation for how SpectroChemPy thinks about scientific data,
meaning, persistence, and framework design.

## Table of Contents

1. [Purpose](#purpose)
2. [Conceptual Levels](#conceptual-levels)
3. [Core Principles](#core-principles)
4. [The Scientific Object](#the-scientific-object)
5. [Decision Framework](#decision-framework)
6. [Relationship to Existing Architecture](#relationship-to-existing-architecture)
7. [Deliverables](#deliverables)

## Purpose

SpectroChemPy has accumulated architecture documents, RFCs, audits, and
implementation decisions. These documents repeatedly reveal the same implicit
design principles.

This document makes those principles explicit.

It is the philosophical foundation of the framework. It describes how
SpectroChemPy thinks about data, operations, persistence, and scientific
meaning. It is intentionally abstract. It does not describe current classes,
APIs, or implementation details.

Every future architectural decision should ideally be justifiable by
referring to one or more principles defined here.

This document sits above all other architecture documents. It does not
replace them. It explains the reasoning that produced them and the framework
that should guide future ones.

---

## Conceptual Levels

SpectroChemPy operates at several distinct levels of abstraction. Decisions
at higher levels constrain what is possible at lower levels. Understanding
the levels clarifies which arguments are relevant to a given decision.

### Scientific Principles

The most abstract level. These are truths about spectroscopy and chemometrics
that do not depend on SpectroChemPy. Signals have support. Measurements have
units. Provenance enables reproducibility. Scientific context is essential
for interpretation.

These principles are not chosen by the framework. They are imposed by the
domain. A framework that violates them is not just badly designed — it is
scientifically unsound.

### Framework Concepts

SpectroChemPy's particular expression of the scientific principles. The
concepts of dataset, coordinate system, model, result, transformation,
project, and provenance graph. These are the framework's answer to the
question: *what abstractions best capture the scientific domain?*

Framework concepts are design choices. They can be changed, but only when
the change better expresses the scientific principles below them.

### Architecture Contracts

Concrete normative decisions about how framework concepts behave. The
metadata taxonomy (five categories with defined propagation rules). The
result object contract (Base → AnalysisResult / FitResult). The reader
normalization policy (semantic destinations). The portable persistence
subset (what survives NetCDF round-trips).

Architecture contracts are the most detailed level before implementation.
They define what must be true, not how to achieve it.

### Implementation

The actual code. NDDataset, Coord, CoordSet, Project, ResultBase,
ProcessingConfigurable, and all the rest. Implementation expresses
architecture contracts under real constraints (Python, traitlets, numpy,
pint, xarray, pluggy, etc.).

Implementation is the most volatile level. It can be restructured,
optimized, or replaced as long as it continues to satisfy the architecture
contracts above it.

### Decision Flow

Decisions should flow downward:

```text
Scientific principles
    → what must be true about spectroscopy
    ↓
Framework concepts
    → what SpectroChemPy models
    ↓
Architecture contracts
    → how those models behave
    ↓
Implementation
    → how those contracts are realized
```

A proposal that violates a scientific principle must be rejected regardless
of its implementation elegance. A proposal that satisfies all higher levels
but requires implementation changes is normal and healthy.

A proposal that tries to justify implementation convenience as a principle
must be examined critically. Most bad architectural decisions come from
allowing implementation constraints to override higher-level principles.

---

## Core Principles

Each principle includes:

- **Idea.** What the principle means.
- **Rationale.** Why the principle exists and what problem it solves.
- **Trade-offs.** What the principle costs.
- **Consequences.** How the principle affects future design decisions.

---

### Principle 1: Scientific meaning is preserved

**Idea.** Operations on scientific objects should preserve the full scientific
context of the data unless there is an explicit, documented reason not to.
Context includes coordinates, units, metadata, and provenance.

**Rationale.** In spectroscopy, context is not decoration — it determines
scientific meaning. A spectrum with coordinates but no units is ambiguous.
An analysis result without provenance is unverifiable. The framework exists
precisely to keep context attached to data through every operation. Every
operation that strips context without warning is a failure of the framework.

**Trade-offs.** Preservation costs complexity, memory, and computation.
Operations must be implemented with preservation in mind. Some operations
naturally change context (a Fourier transform changes the coordinate domain;
a normalization strips absolute intensity). These changes should be explicit
and documented, not silent side effects.

**Consequences.**

- Every operation family must specify its behavior for coordinates, units,
  metadata, and provenance. This specification is part of the architecture
  contract, not optional documentation.
- Operations that intentionally strip or change context must do so explicitly.
  Implicit context loss is a design defect.
- New operation types (processing, analysis, future unknown families) must
  define their preservation semantics before implementation begins.
- Performance optimizations that would silently discard context must be
  rejected or redesigned.

---

### Principle 2: Data without context is incomplete

**Idea.** A raw array of numbers is not sufficient for scientific computing.
The numbers must be accompanied by what they represent (units), where they
live (coordinates), and what they mean (metadata).

**Rationale.** This principle is the reason SpectroChemPy exists. NumPy and
generic array frameworks are excellent at numerical computation, but they
treat context as optional. For spectroscopy, context is not optional. A
floating-point number at position 42 is meaningless; a floating-point number
at 1000 cm⁻¹ with units of absorbance is a measurement.

**Trade-offs.** Objects are heavier. Every dataset carries coordinates, units,
and metadata in addition to raw values. APIs are more complex because they
must handle context. Simple operations (e.g., multiplying two arrays) require
unit compatibility checking that a raw array framework would skip.

**Consequences.**

- Coordinates and units are part of what a dataset is, not decorations on it.
- Removing context from data must require explicit intent. There is no
  implicit path from a dataset to a raw array.
- Import operations must reconstruct context, not just numbers.
- Export to context-free formats (CSV, plain text) is possible but lossy,
  and the loss must be documented.

---

### Principle 3: Coordinates describe the structure of data

**Idea.** The support structure — axes, coordinates, their units and labels —
describes where measurements live. It is not metadata about the data; it is
structural information that determines how data can be sliced, combined, and
interpreted.

**Rationale.** Two datasets with identical numerical values but different
coordinate systems represent different scientific phenomena. A spectrum at
high resolution and one at low resolution differ in their coordinate
structure. Coordinate compatibility must be checked during operations because
mixing incompatible coordinates produces scientifically meaningless results.

**Trade-offs.** Coordinate handling adds overhead to every operation.
Complex coordinate systems (same-dimension alternatives, shared coordinates)
require sophisticated lifecycle management. Every operation must consider
whether coordinates should be propagated, recomputed, or dropped.

**Consequences.**

- Coordinate compatibility is checked during arithmetic, concatenation, and
  other combining operations.
- Coordinate propagation is not optional — every operation family specifies
  coordinate behavior.
- The coordinate system has defined lifecycle behavior through slicing,
  reduction, reshape, concatenation, and interpolation.
- Future operations (model application, prediction, simulation) must specify
  coordinate behavior before implementation.

---

### Principle 4: Units are first-class

**Idea.** Units are not strings or metadata annotations. They are a
fundamental property of numerical data. Operations must be unit-aware,
unit-safe, and unit-preserving.

**Rationale.** Unit errors are among the most costly mistakes in scientific
computing. In spectroscopy, units carry essential meaning — 4000 cm⁻¹ and
4000 nm describe completely different physical phenomena. The framework must
prevent unit errors at the architectural level, not merely document that
users should be careful.

**Trade-offs.** Unit handling adds computational cost and implementation
complexity. Integration with pint requires careful management of magnitude
vs. value. Some operations naturally change units (e.g., derivative
spectroscopy changes absorbance units per wavenumber). These must be handled
explicitly, not silently.

**Consequences.**

- All numerical data has units. There is no unit-less data path.
- Operations check unit compatibility before computation.
- Unit conversion is explicit. Units are never silently coerced.
- Operations that change units (derivatives, Fourier transforms,
  normalization) document their unit transformation as part of their contract.
- Pint is the unit engine, but the framework may wrap or extend it for
  spectroscopy-specific contexts (FT-IR conventions, Raman shift, ppm).

---

### Principle 5: Metadata has semantics, not just storage

**Idea.** Metadata is not a homogeneous bag of key-value pairs. It has
categories — scientific identity, structural, provenance, presentation,
extension — and those categories determine how metadata propagates through
operations.

**Rationale.** Before the metadata taxonomy, every piece of metadata was
treated equally. A title and a creation timestamp propagated through the
same code path, even though they should behave differently. The title should
survive smoothing; the timestamp should be updated. Categories make these
distinctions explicit and testable.

**Trade-offs.** Categories add conceptual overhead. Every metadata field must
be classified. Borderline cases require explicit decisions. The taxonomy must
be maintained as new metadata types are introduced.

**Consequences.**

- Every metadata field has a category, and the category determines its
  propagation behavior.
- New metadata fields must be classified before or during introduction.
- Propagation rules are defined per category, not per field, unless a field
  has exceptional behavior that is explicitly documented.
- The taxonomy is extensible (the extension/private category exists for
  unnormalized payloads), but fields should be promoted to an explicit
  category when their semantics become stable.

---

### Principle 6: Provenance is part of the scientific record

**Idea.** Where data came from and what was done to it is not optional
context — it is part of the scientific record. Provenance should be preserved
through every operation and across persistence boundaries.

**Rationale.** Scientific reproducibility requires provenance. A baseline-
corrected spectrum means something different from a raw spectrum. Analysis
results without provenance cannot be verified. Provenance is not a log —
it is the chain of scientific custody.

**Trade-offs.** Provenance tracking costs memory and serialization space.
Every operation must contribute to provenance. Provenance complicates
equality (are two datasets equal if they differ only in provenance?).
Structured provenance graphs require more infrastructure than linear history.

**Consequences.**

- Every operation contributes to provenance. Processing adds history entries.
  Analysis creates result objects with provenance context.
- Provenance survives persistence. Both native and portable formats preserve
  provenance, though at different levels of detail.
- Provenance is conceptually distinct from history. History is the event
  trail; provenance includes source lineage, authorship, and context.
- Future structured provenance (operation → inputs → outputs graph) should
  build on this principle, not replace it.

---

### Principle 7: Native persistence preserves completeness

**Idea.** The native persistence format should round-trip the full runtime
scientific object — all metadata, coordinates, provenance, and structure that
the runtime model supports.

**Rationale.** The native format is the canonical preservation format for
SpectroChemPy users who want full fidelity. If native save/load loses
information, users cannot trust it for their primary data storage. Trust
in persistence is essential for a scientific computing framework.

**Trade-offs.** Native format is SpectroChemPy-specific and not directly
interoperable with other tools. It requires schema versioning, migration
code, and compatibility testing. Native format evolution must be managed
carefully to avoid breaking existing files.

**Consequences.**

- Native format preserves everything the runtime model supports.
- Schema versioning is mandatory. Migration paths must exist for older
  versions.
- Native format is not the interchange format. Interchange is portable
  persistence.
- Implementation changes that would break native round-trips must be
  accompanied by migration support.

---

### Principle 8: Portable persistence enables interoperability

**Idea.** The portable format preserves a well-defined, documented subset of
the scientific object that is usable by other tools and systems.

**Rationale.** Scientific data must be shareable. NetCDF via xarray is a
widely supported, self-describing format. The portable subset is the
interchange contract. Users who need full fidelity use native format; users
who need interoperability use portable format.

**Trade-offs.** The portable format loses information. What is included and
excluded must be explicitly documented. Users must understand the difference
between native and portable persistence. The portable format constrains the
runtime model — if a concept cannot be expressed in the portable format,
it might still exist in the runtime model but cannot be shared portably.

**Consequences.**

- The portable persistence subset is explicitly documented, including what
  is included, what is excluded, and why.
- Round-trip through portable format is lossy by design, but the losses are
  documented and predictable.
- The portable format does not define the runtime model. The runtime model
  is richer. The portable format is a projection.
- Future portable formats (CSDM alignment, other interchange standards)
  follow the same principle: they project a documented subset, not the full
  runtime model.

---

### Principle 9: Visualization is a view, not state

**Idea.** Plots and figures are presentations of data, not part of the data
model. They are ephemeral and should not be persisted as part of scientific
objects.

**Rationale.** Plots depend on display context — screen resolution, color
space, backend, user preferences. Persisting them would couple data to a
specific presentation, which is fragile and unnecessary. A dataset is not
defined by how it looks in a particular rendering. Separating view from
state also simplifies serialization and comparison.

**Trade-offs.** Plot state (zoom level, axis limits, annotation positions)
cannot be saved with the dataset. Users who want to reproduce a specific
view must save the rendering commands separately. Some display-related
metadata (coordinate reversal convention, preferred colormap) may
legitimately belong to the dataset and must be distinguished from
visualization state.

**Consequences.**

- Plot state is not stored on datasets or other scientific objects.
- Plotting is delegation to matplotlib. SpectroChemPy provides convenience
  methods, not a plotting model.
- Figures and axes are the user's responsibility to manage and persist.
- Display metadata that is part of the scientific object (coordinate
  reversal, axis labeling conventions) is preserved through the display
  architecture, not through plot state.

---

### Principle 10: Plugins extend without fragmenting

**Idea.** Plugins can add readers, writers, processing methods, and analysis
methods, but they must respect the core contracts — metadata taxonomy,
coordinate system, result contract, persistence boundaries.

**Rationale.** The framework must remain coherent as it grows. If every
plugin invented its own metadata conventions, persistence mechanism, or
object model, the framework would fragment into incompatible islands.
Core contracts provide the common language that keeps plugins interoperable.

**Trade-offs.** Plugins are constrained. Some plugin ideas may not fit the
core contracts and would need to be implemented outside the framework or
drive an extension to the contracts. Plugin authors must understand and
follow the contracts, which adds a learning curve.

**Consequences.**

- Plugin objects follow the same contracts as core objects. A result from a
  plugin analysis method follows the result contract. A reader from a plugin
  normalizes to the same semantic destinations.
- Plugin readers, writers, and methods are discoverable through the same
  mechanisms as core ones.
- Core/plugin boundary is explicit. What belongs in core and what belongs
  in a plugin is governed by documented criteria, not convenience.
- A plugin that cannot respect core contracts without distorting them
  suggests either that the contracts need extension or that the plugin
  belongs outside the official ecosystem.

---

### Principle 11: APIs express scientific concepts

**Idea.** The public API should use the vocabulary of spectroscopy and
chemometrics, not the vocabulary of the implementation. Users should think
about spectra, coordinates, models, and results, not arrays, mixins,
traitlets, and serializers.

**Rationale.** SpectroChemPy is a framework for scientists. Its API should
speak their language. When a user reads `dataset.coords` they should think
about coordinate axes, not about CoordSet internals. When they call
`estimator.fit(data)` they should think about training a model, not about
setting `_fitted = True`.

**Trade-offs.** Abstraction hides detail. Power users who need to understand
the implementation must learn both the scientific API and the underlying
mechanisms. Sometimes a clean scientific API requires complex implementation
machinery behind the scenes.

**Consequences.**

- Public API names are chosen for scientific clarity, not implementation
  convenience.
- Implementation details are private or explicitly documented as
  implementation-only.
- When the implementation changes (e.g., CoordSet storage redesign), the
  public API remains stable as long as the scientific concepts are preserved.
- Error messages and documentation use scientific vocabulary first,
  implementation vocabulary second.

---

### Principle 12: Explicit contracts over implicit behavior

**Idea.** Behavior should be documented, discoverable, and testable, not
emergent from implementation details. If an operation preserves coordinates,
that should be a documented contract, not an accidental consequence of how
copying works.

**Rationale.** Implicit behavior that changes between releases breaks user
code. Implicit behavior that differs between similar operations (Group A vs
Group B processing assembly) confuses users and maintainers alike. Explicit
contracts make the framework predictable, auditable, and maintainable over
years of evolution.

**Trade-offs.** Contracts require documentation, tests, and maintenance.
They limit implementation freedom — once a contract is defined, the
implementation must conform to it. Defining contracts takes time that could
otherwise be spent on features.

**Consequences.**

- Each operation family has a documented contract with defined behavior for
  coordinates, units, metadata, and provenance.
- Tests verify contract compliance. A test that only tests numerical
  correctness is insufficient; it must also test context preservation.
- Implementation changes that preserve the contract are safe. Implementation
  changes that break the contract require a contract revision, not just a
  code review.
- Contracts are versioned. When a contract changes, the change is documented
  and the transition is managed.

---

### Principle 13: Behavioral compatibility over implementation stability

**Idea.** Users depend on what the framework does, not on how it does it.
The implementation can change — even dramatically — as long as the
observable behavior visible through the public API is preserved.

**Rationale.** This principle gives maintainers freedom to refactor,
optimize, and improve internals without breaking user code. It aligns with
semantic versioning — a major version change signals behavioral changes; a
minor or patch change should not break documented behavior.

**Trade-offs.** Behavioral compatibility is harder to verify than
implementation stability. It requires comprehensive contract tests. Some
optimizations may be blocked by behavioral constraints. Some behavioral
details (e.g., performance characteristics) are not part of the behavioral
contract and may change freely.

**Consequences.**

- Tests focus on observable behavior, not implementation structure.
- Implementation details are explicitly documented as implementation-specific.
- Breaking behavioral changes require deprecation cycles documented in the
  changelog.
- Performance is not part of the behavioral contract unless explicitly
  guaranteed.
- Serialization format is part of the behavioral contract — changes to
  serialization require migration support.

---

## The Scientific Object

The concept of a Scientific Object appears across multiple architecture
documents and RFCs without being formally defined. This section discusses
the architectural implications of making the Scientific Object an explicit
unifying concept.

### What a Scientific Object Is

A Scientific Object is the framework's representation of a meaningful unit
of scientific work. It is not a base class, not an interface, and not a
protocol. It is a conceptual contract — a set of expectations that the
framework has about objects that carry scientific meaning.

### Implied Contracts

If a concept is a Scientific Object, it implies:

**Identity contract.** The object has stable identity. It can be identified
across sessions, copies, and transformations. Identity is explicit (UUID),
not implicit (memory address). Copy produces a new identity; transformation
may produce a new identity or preserve the original, depending on whether
the transformation changes the scientific meaning.

**Provenance contract.** The object carries provenance about its creation
and transformation history. Provenance includes where it came from, what
was done to it, and what parameters were used. Provenance survives
persistence.

**Persistence contract.** The object has defined persistence behavior. It
knows what state to serialize, how to reconstruct itself, and what
compatibility rules apply across versions. Persistence is explicit and
versioned.

**Display contract.** The object has defined terminal and HTML
representation. Display is semantic (structured sections, typed items), not
format-specific. Display is separate from data — visualization is a view,
not state.

**Equality contract.** The object has defined equality semantics. Equality
may be structural (all fields equal), scientific (same coordinates and
values), or identity-based (same UUID), depending on context. The contract
defines which kind of equality applies to which use case.

**Transformability contract.** The object participates in defined
transformation operations. Some operations preserve the object type
(processing a dataset yields a dataset); others produce a different type
(fitting an estimator yields a model and a result).

### Current Maturity

Different objects in SpectroChemPy fulfill these contracts to different
degrees:

- **Dataset** fulfills all contracts completely. It has identity (UUID),
  provenance (history, typed provenance fields), persistence (native and
  portable), display (sections model), equality (structural), and
  transformability (processing, slicing, arithmetic).

- **Project** fulfills most contracts. Identity, persistence, display are
  mature. Provenance is dataset-level aggregated, not explicit at the
  project level. Equality is structural.

- **Result** fulfills identity (name, estimator identity) and display
  partially. Provenance is present but minimal. Persistence is absent.
  Transformability is undefined (results are outputs, not inputs to further
  operations).

- **Model** does not exist as a concept, so none of the contracts are
  fulfilled. This is the largest gap.

### Architectural Implications of Formalizing the Scientific Object

Formalizing the Scientific Object as an explicit concept would have several
architectural implications:

**Unified lifecycle.** All scientific objects would participate in a common
lifecycle: creation, transformation, persistence, display, destruction.
Lifecycle events would be consistent across object types.

**Contract testing.** A shared test suite for Scientific Object contracts
would verify that every scientific object fulfills identity, provenance,
persistence, display, equality, and transformability contracts.

**Discovery.** The framework could enumerate scientific objects. Plugins
could register new types of scientific objects. The plugin discovery
mechanism would know about scientific object types.

**Persistence registry.** Persistence could be generalized. Instead of each
object type implementing its own save/load, a persistence registry could
dispatch type-aware serialization.

**Project membership.** Project typed membership could be defined in terms
of Scientific Object contracts. Any object that fulfills the contracts
would be eligible for Project membership.

**Provenance graph.** If all scientific objects carry provenance, a
provenance graph connecting them becomes possible. Each object knows its
inputs and the transformation that produced it.

These implications are not implementation proposals. They are consequences
that would follow from making the Scientific Object an explicit concept.
The framework may adopt some, all, or none of them, depending on how it
chooses to formalize the concept.

---

## Decision Framework

The following questions should be asked for every architectural proposal.
They provide a structured way to evaluate proposals against the framework
principles.

### Preservation Questions

- Does the proposal preserve scientific meaning through operations?
- If context (coordinates, units, metadata) is intentionally changed or
  stripped, is the change explicit and documented?
- Does the proposal make it harder or easier to lose scientific context
  accidentally?

### Consistency Questions

- Does the proposal improve conceptual consistency across the framework?
- Does it reduce or increase the number of special cases?
- Does it make the framework more predictable (same inputs → same behavior
  across similar operations)?
- Does it align with the existing metadata taxonomy and propagation rules?

### Simplification Questions

- Does the proposal simplify the framework overall, even if it adds
  complexity in one area?
- Does it reduce the burden on future maintainers?
- Does it make the API more intuitive for scientists?

### Reproducibility Questions

- Does the proposal improve or preserve scientific reproducibility?
- Does provenance survive the proposed change?
- Are parameters and transformations recorded explicitly enough for
  reproduction?

### Interoperability Questions

- Does the proposal improve or preserve interoperability with other tools?
- Does it respect the dual persistence model (native rich, portable
  lossy-but-documented)?
- Does it preserve the portable persistence subset?

### Contract Questions

- Does the proposal strengthen or weaken existing contracts?
- Does it introduce new implicit behavior that should be explicit?
- Are the new contracts documented, testable, and versioned?
- Does the proposal clarify what is guaranteed vs. what is implementation-
  specific?

### Objection Handling

When a proposal fails one of these questions, the next question is whether
the failure is:

- **Fundamental.** The proposal violates a scientific principle and cannot
  be accepted in any form.
- **Mitigable.** The proposal can be adjusted to satisfy the principle
  without changing its core intent.
- **Warranted exception.** The proposal has a compelling reason to violate
  the principle, and the exception is explicitly documented.

Warranted exceptions should be rare. Each exception weakens the principle
and should be justified by a concrete, demonstrated need.

---

## Relationship to Existing Architecture

This document does not replace the existing architecture documents. It
positions itself above them.

### Framework Vision Audit

The Framework Vision Audit asks: *what should SpectroChemPy model?* This
document asks: *why should it model those things and not others?* The
principles here justify the conceptual choices identified in the audit.

If the Framework Vision Audit proposes that a Model abstraction should
exist, this document explains *why* models matter (Principle 1: scientific
meaning is preserved; Principle 6: provenance is part of the scientific
record) and what constraints a Model abstraction must satisfy (coordinate
awareness, unit awareness, explicit contracts).

### Metadata and Support Model

The metadata architecture defines the five-category taxonomy and its
propagation rules. This document explains *why* categories exist (Principle
5: metadata has semantics) and *why* propagation rules are necessary
(Principle 12: explicit contracts over implicit behavior).

### Portable Persistence Model

The portable persistence document describes what survives NetCDF round-trips.
This document explains *why* there are two persistence tiers (Principle 7:
native completeness; Principle 8: portable interoperability) and *why*
the portable subset is a projection, not a definition.

### Result Object Contract

The result contract defines the ResultBase → AnalysisResult / FitResult
hierarchy. This document explains *why* results exist as explicit objects
(Principle 6: provenance is part of the scientific record) and *why* they
should eventually have defined persistence (Principle 1: scientific meaning
is preserved).

### Display Architecture

The display architecture defines the semantic display model and rendering
pipeline. This document explains *why* display is separate from data
(Principle 9: visualization is a view, not state) and *why* display is
semantic rather than format-specific.

### CoordSet Architecture

The CoordSet storage architecture defines coordinate organization and
lifecycle. This document explains *why* coordinates are owned by datasets
and not independent (Principle 3: coordinates describe the structure of
data) and *why* coordinate lifecycle is explicit.

### Future Model Abstraction

When the framework introduces a Model abstraction, it should be evaluated
against the principles in this document:

- Does it preserve scientific meaning? (Principle 1)
- Does it treat coordinates and units as first-class? (Principles 3, 4)
- Does it carry provenance? (Principle 6)
- Does it have explicit persistence contracts? (Principles 7, 8)
- Does it express scientific concepts in its API? (Principle 11)
- Does it have explicit behavioral contracts? (Principle 12)

### How to Reference This Document

Future RFCs and architecture decisions should reference this document by
principle number and name:

> This proposal satisfies Principle 1 (Scientific meaning is preserved)
> because coordinate propagation is explicit and documented. It satisfies
> Principle 6 (Provenance is part of the scientific record) because every
> result carries estimator identity and parameters.

This document should be cited as the authoritative source for *why* a
decision was made, not just *what* was decided.

---

## Deliverables

### Core Principles

The minimal set of principles that should remain stable for many years:

1. **Scientific meaning is preserved.** Context survives operations.
2. **Data without context is incomplete.** Arrays alone are insufficient.
3. **Coordinates describe the structure of data.** Support is structural,
   not decorative.
4. **Units are first-class.** Unit safety is non-negotiable.
5. **Metadata has semantics.** Categories determine propagation.
6. **Provenance is part of the scientific record.** Lineage is preserved.
7. **Native persistence preserves completeness.** Full round-trip fidelity.
8. **Portable persistence enables interoperability.** Documented subset for
   interchange.
9. **Visualization is a view, not state.** Plots are ephemeral.
10. **Plugins extend without fragmenting.** Core contracts are universal.
11. **APIs express scientific concepts.** Vocabulary matches the domain.
12. **Explicit contracts over implicit behavior.** Behavior is documented
    and testable.
13. **Behavioral compatibility over implementation stability.** Observable
    behavior is the guarantee.

These thirteen principles form the stable foundation. Proposals that
conflict with them should be rare and require strong justification.

### Consequences

**For operation semantics.** Every operation family must specify its
coordinate, unit, metadata, and provenance behavior. These specifications
are architecture contracts, not documentation afterthoughts. The metadata
contract campaign (current roadmap priority 1) is the first systematic
application of this consequence.

**For persistence design.** The dual persistence model (native complete,
portable projected) is not a transitional state — it is the permanent
architecture. Future persistence work (result persistence, model
persistence, format evolution) must respect both tiers.

**For plugin governance.** Plugins are not exempt from core contracts. A
plugin that violates metadata semantics, unit safety, or the result
contract is not a valid extension. Plugin review must verify contract
compliance.

**For API design.** New public APIs must be named and structured around
scientific concepts, not implementation mechanisms. Internal restructuring
(mixins, traitlets, MRO changes) should not leak into the public surface.

**For the Model abstraction.** When the Model concept is introduced, it
must satisfy all applicable principles: coordinate-aware, unit-safe,
provenance-bearing, persistent (with defined contracts), API-expressed
for scientists, and governed by explicit behavioral contracts.

**For the Scientific Object contract.** Formalizing the Scientific Object
as an explicit concept would extend the principles into a concrete
contract framework. The degree of formalization is an open decision, but
the principles already imply what the contracts should contain.

### Open Questions

**What is the precise boundary between scientific object and infrastructure
value?** This document treats datasets, models, results, and projects as
scientific objects. Coordinates, units, and metadata containers are
infrastructure. The boundary is clear in current practice but not formally
defined.

**Should scientific object contracts be expressed as a Protocol, a base
class, or remain conceptual?** This document takes no position. The
architectural implications of formalizing the Scientific Object are
discussed, but the mechanism is an implementation decision.

**How should persistence contracts be versioned and migrated?** The
principle of explicit contracts requires versioning, but the mechanism
(type identifiers, schema versions, migration paths) is an open design
question deferred to future serialization RFCs.

**What is the role of the Experiment concept?** The Framework Vision Audit
identifies Experiment as a missing concept. This document does not endorse
or reject it. If Experiment is introduced, it must satisfy the same
principles as other scientific objects.

**How do performance and correctness trade off?** Principle 1 prioritizes
correctness, but the trade-off is not absolute. An explicit performance
escape hatch (e.g., raw array access for numerical libraries) is compatible
with the principles as long as it is explicit and documented.

**Should the Scientific Object be a formal concept across the entire
framework, or should some objects remain exempt?** Current practice treats
datasets and projects as full scientific objects, results as partial,
estimators as computational objects, and infrastructure values as carriers.
This graduated approach is pragmatic but may benefit from formalization.

**Should the framework eventually provide a declarative workflow or
pipeline concept?** Transformation composition is natural from the
principles (Principle 12: explicit contracts), but whether it becomes a
first-class concept is an open question.

These open questions are not failures. They identify areas where the
principles provide guidance but not a unique answer. Resolving them will
require focused RFCs that apply the decision framework defined above.

### Position in the Documentation

This document sits at the top of the architecture documentation hierarchy:

```text
framework-principles.md          ← you are here
    why the framework exists and how it thinks

    ↓ informs

metadata-and-support-model.md    ← what the runtime model is
reader-normalization-architecture.md
portable-persistence-model.md
coordset-storage-architecture.md
display-architecture.md
result-object-architecture.md
array-class-responsibility.md

    ↓ contractually specifies

../rfcs/                         ← normative design contracts
    dimensional-semantics-contract.md
    metadata-taxonomy-contract.md
    label-semantics-contract.md
    provenance-and-history-contract.md
    scientific-object-model-and-persistence-boundaries.md
    ... (24 RFCs total)

    ↓ guides

private roadmap planning         ← campaign ordering

    ↓ constrains

private audit notes              ← investigations (non-normative)
```

Future RFCs should reference `framework-principles.md` as the source of
architectural philosophy and cite specific principles when justifying their
design:

> Following Principle 5 (Metadata has semantics), this RFC proposes that
> the new provenance field be classified as provenance metadata with
> defined propagation through all operation families.

This document should be read by every new maintainer as part of their
orientation. It should be reviewed periodically (every 1-2 years) to verify
that it still reflects the framework's actual design philosophy and to
incorporate lessons from implementation experience.
