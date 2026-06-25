# Scientific Object Model and Persistence Boundaries

**Status:** Proposed Maintainer RFC

**Scope:** Object roles, persistence boundaries, Project eligibility, and
extension governance.

The key words MUST, SHOULD, and MAY express the proposed normative contract.
They do not alter current behavior unless this RFC is accepted and implemented
through separately approved work.

**Related documents:**

- `maintainers/architecture/result-object-contract-rfc.md`
- `maintainers/architecture/result-object-migration-roadmap.md`
- `maintainers/rfcs/analysis-fit-result-architecture.md`
- `maintainers/rfcs/metadata-contract.md`
- recent maintainer review of `Project` and serialization boundaries

## Motivation

SpectroChemPy now has clearer boundaries for datasets, coordinates, metadata,
display, Projects, and analysis results. The next architectural question is
broader than whether `ResultBase` should be serializable:

> Which objects are official scientific objects in SpectroChemPy, and which of
> those objects should cross a persistence boundary?

Without an explicit answer, several reasonable future features can pull the
architecture in incompatible directions:

- Result persistence can accidentally make every runtime detail durable.
- Project typed members can turn `Project` back into a generic object bag.
- plugins can invent incompatible persistence and membership rules.
- provenance can become confused with object snapshots or workflow replay.
- serialization changes can expose implementation details as permanent schema.

This RFC defines a common vocabulary and a conservative current boundary. It
is a foundation for later RFCs; it does not authorize an implementation.

## Non-goals

This RFC does not:

- change any public API;
- change native `.scp` or `.pscp` serialization;
- add save/load support to Result objects;
- broaden the set of objects accepted by `Project`;
- define a serialization registry or migration mechanism;
- define a complete provenance model;
- require plugins to expose new object types.

## Current object landscape

### Core data objects

`NDDataset` is the primary first-class scientific data object. It owns a
labeled numerical signal together with coordinates, units, masks, metadata,
identity fields, and history. It has native top-level save/load support.

`Coord` represents scientific support along an axis. `CoordSet` organizes one
or more coordinates, including same-dimension alternatives and references.
They carry scientific meaning, but currently persist as structural components
of an `NDDataset`, not as independent top-level native files.

### Containers

`Project` is a typed, recursive, persistent organizational container. It
currently accepts only `NDDataset` and nested `Project` objects. It is not a
workflow engine, object registry, provenance graph, or general-purpose Python
object store.

### Computational objects

Analysis and fitting classes such as `PCA`, `NMF`, `MCRALS`, `FastICA`,
`PLSRegression`, `EFA`, `SIMPLISMA`, and `Optimize` are runtime computational
objects. They own algorithms, configuration, fitting lifecycle, backend state,
and user operations.

A fitted estimator remains a computational object. Fitting does not by itself
make the estimator a durable scientific record.

### Result objects

`ResultBase`, `AnalysisResult`, and `FitResult` are explicit runtime records of
one analysis or fit. They own named output datasets, parameters, diagnostics,
and estimator identity. They are scientific objects in the broad object model,
but they do not currently implement a persistence schema or native save/load.

### Infrastructure objects and values

`Meta`, units, quantities, masks, timestamps, names, and history entries support
the meaning and interpretation of scientific objects. They normally persist as
owned values or components of another object.

In particular, history is currently persisted state on `NDDataset`, not a
standalone `History` entity with its own lifecycle and schema. Structured
provenance may become a distinct concept later, but should not be inferred from
the existence of the current history field.

### Plugin-defined objects

Official plugins such as `spectrochempy-iris`, `spectrochempy-tensor`, and
future plugins may need domain-specific data, model, or result objects. Such an
object can be a legitimate SpectroChemPy scientific object even when its class
and schema are owned by a plugin. Plugin ownership does not automatically make
the object persistent or Project-compatible.

## Object categories

A single hierarchy is insufficient because an object's scientific role and its
persistence status answer different questions. This RFC therefore classifies
objects on two independent axes.

### Role axis

| Role | Meaning | Current examples |
|---|---|---|
| Scientific data object | Owns scientific observations or derived labeled numerical data | `NDDataset` |
| Structural scientific component | Defines support, interpretation, or structure owned by a larger object | `Coord`, `CoordSet` |
| Organizational container | Owns a named hierarchy of eligible scientific objects | `Project` |
| Computational object | Performs algorithms and owns live runtime configuration or backend state | `PCA`, `NMF`, `MCRALS`, `FastICA`, `PLSRegression`, `EFA`, `SIMPLISMA`, `Optimize` |
| Scientific result record | Owns the outputs and diagnostics of one completed computation | `ResultBase`, `AnalysisResult`, `FitResult` |
| Infrastructure value | Supports identity, interpretation, metadata, units, or lineage within another object | `Meta`, `Unit`, `Quantity`, masks, history entries |
| Extension scientific object | Provides a domain object whose semantics are owned by a plugin | future IRIS or tensor objects |

These roles are about responsibility, not implementation inheritance. A future
plugin object may be a scientific data object or a result record as well as an
extension-owned object.

### Persistence axis

| Status | Meaning | Current examples |
|---|---|---|
| Top-level persistent | Has an official native save/load entry point and round-trips as the same object type | `NDDataset`, `Project` |
| Embedded persistent | Round-trips as owned state within a top-level persistent object | `Coord`, `CoordSet`, `Meta`, units, quantities, masks, history |
| Runtime-only | Exists as a public or internal runtime object without an official persistence contract | estimators, `ResultBase`, `AnalysisResult`, `FitResult` |
| Persistence candidate | May become persistent after an explicit schema and compatibility decision | Result objects, selected future plugin scientific objects |

`Persistence candidate` is not an implemented capability and carries no
compatibility promise.

### Why the axes must remain separate

- A scientific object does not have to be independently persistent.
- Embedded persistence does not imply eligibility as a Project member.
- A runtime-only object can still be public and scientifically meaningful.
- A persistent container is not automatically a scientific data object.
- A plugin-defined object is not less scientific, but its provider and loading
  dependencies require additional governance.

## Persistence boundary

### Definition

A persistent object is an object for which SpectroChemPy or an approved
extension promises to reconstruct the same type and scientific meaning from a
maintained serialized representation across supported versions.

Persistence is therefore a contract, not merely the ability to call
`json.dumps()`, pickle an instance, or write some values to disk.

### Required criteria

An independently persistent object MUST have:

1. a stable, unambiguous type identity;
2. a documented owner for the type and its schema;
3. an explicit schema version or equivalent evolution marker;
4. a defined set of owned serialized state;
5. round-trip semantics for scientific meaning, not necessarily byte equality;
6. compatibility and migration rules for supported older representations;
7. validation and explicit failure behavior for invalid or unsupported state;
8. no requirement for live callables, open resources, GUI state, or a live
   estimator instance to reconstruct the object;
9. focused round-trip and backward-compatibility tests;
10. a defined trust and security posture for the serialized payload.

Embedded persistent components MAY rely on the containing object's type and
schema for identity and versioning. They still need deterministic ownership and
round-trip semantics within that container.

### Ownership rule

Persistence SHOULD serialize state owned by the object. References to external
objects, caches, backend instances, and process-local services MUST either be
excluded or represented through an explicit, stable reference contract.

Persisting a snapshot of some values is not enough to claim that the original
runtime object is persistent.

### Native persistence is not interchange or archival

The current `.scp` and `.pscp` formats are useful native persistence formats,
but they are not yet a general versioned object-schema system. Their JSON-shaped
payloads include Python-specific encoded array data and targeted compatibility
branches.

Accordingly, this RFC distinguishes:

- native persistence, which reconstructs supported SpectroChemPy objects;
- interchange export, which communicates selected content to another system;
- archival preservation, which would require a stronger long-term format and
  governance promise.

No claim of interchange safety or archival stability follows automatically
from native save/load support.

## Project boundary

### Definition

A Project-compatible object is a persistent scientific object or persistent
scientific container that can participate coherently in Project naming,
ownership, display, save/load, and compatibility rules.

Project compatibility is a stronger contract than persistence:

```text
Project-compatible => persistent
the converse is not necessarily true
```

### Eligibility criteria

Before a new object type can become a Project member, it MUST have:

- stable typed identity and an owned schema;
- stable round-trip behavior;
- a clear user-facing name or key policy;
- a safe minimal text representation;
- explicit ownership semantics for nested datasets or other scientific state;
- no hidden dependency on live runtime services;
- defined behavior for copy, rename, replacement, removal, and hierarchy;
- defined compatibility behavior when its type or schema is unavailable;
- a clear scientific reason to be organized in a Project.

HTML display is desirable but is not a persistence prerequisite. Display must
never be used as a substitute for typed identity or schema.

### Container policy

`Project` is not a generic object bag. Convenience, Python serializability, or
the existence of `repr()` is insufficient for membership.

The removal of arbitrary `_others` storage is a durable architectural boundary.
Any future broadening must add an explicitly governed member type or protocol,
not restore untyped catch-all storage.

### Current decision

Today, `Project` SHOULD remain limited to:

- `NDDataset`;
- nested `Project`.

This RFC identifies future eligibility criteria but does not approve any new
member type.

## Result object status

### Current classification

`ResultBase`, `AnalysisResult`, and `FitResult` are:

- first-class runtime scientific result records;
- owners of named outputs and diagnostics;
- runtime-only under the current persistence contract;
- candidates for explicit export;
- possible future standalone persistence candidates;
- not currently Project-compatible.

Calling Result objects first-class scientific records recognizes their public
semantics and ownership role. It does not imply top-level persistence, Project
membership, or workflow replay.

### Option 1: Remain runtime-only

Results remain lightweight records attached to the runtime analysis lifecycle.
Users persist selected outputs separately.

This is the safest current status and imposes no premature schema burden, but
it cannot preserve a complete grouped analysis result as one object.

### Option 2: Runtime object with explicit exports

Results remain runtime-only but provide deliberate transformations to stable
objects or external representations, such as datasets, Projects, reports, or
interchange JSON.

This is the preferred bridge if persistence demand appears before a Result
schema is ready. The export may be lossy, provided that its contract states
what is retained.

### Option 3: Standalone persistent Result

Results gain their own type identity, schema, versioning, save/load behavior,
and compatibility policy without becoming Project members.

This is the preferred first experiment if native Result persistence is pursued.
It isolates Result schema design from Project extensibility and absent-plugin
loading.

### Option 4: Persistent and Project-compatible Result

Results become legal typed members of `Project` after satisfying both the
persistence and Project eligibility contracts.

This may eventually provide strong workflow ergonomics, but it is premature
until standalone Result persistence and Project typed-member governance have
been validated independently.

### Result subtype policy

`AnalysisResult` and `FitResult` currently specialize one common ownership
contract. They SHOULD share persistence infrastructure if persistence is later
approved, while retaining distinct stable type identities when their schemas
or semantics differ.

No Result persistence design should serialize a live estimator, backend model,
callable, or arbitrary diagnostics dictionary without an explicit value-domain
contract.

## Export versus persistence

Export and persistence MUST be treated as distinct architectural operations.

Persistence reconstructs the same object category and scientific meaning:

```text
Result -> serialized Result -> Result
```

Export creates another representation or another scientific object:

```text
Result -> NDDataset
Result -> Project of NDDataset objects
Result -> human-readable report
Result -> interchange JSON document
```

An export may flatten structure, omit diagnostics, convert values, or lose
runtime identity. Those losses can be valid when they are explicit. They are
not valid under a same-object persistence claim.

The target format does not determine the distinction. JSON can be either a
versioned persistence representation or an export document. A `Project`
produced from a Result is an export unless loading it reconstructs the original
Result under an approved Result schema.

This distinction SHOULD become a stable project-wide principle. It prevents
stable existing objects such as `NDDataset` from becoming hidden carriers for
opaque Result state.

## Plugin implications

### Scientific object ownership

A plugin MAY define a scientific object. Its domain semantics, Python class,
and detailed schema may be plugin-owned. Official status does not require that
every scientific object class live in core.

### Persistent plugin objects

A plugin MAY define a persistent object only after providing:

- a globally stable, namespaced type identifier;
- an explicit schema and schema version;
- a compatibility and migration policy;
- round-trip tests;
- a declared provider and support lifecycle;
- a trust and validation policy;
- defined behavior when the provider is absent or incompatible.

Core SHOULD own the native container dispatch and failure policy even when a
plugin owns the member schema.

### Project-compatible plugin objects

A persistent plugin object is not automatically Project-compatible. Project
membership additionally requires the Project eligibility criteria and a core-
approved typed-member extension mechanism.

The absent-plugin case MUST be decided before plugin objects can enter native
Project files. Possible policies include a hard error, an inspectable opaque
placeholder, or a restricted metadata-only placeholder. Silent omission is not
acceptable because it would silently lose scientific state.

### Symmetry rule

Core and official plugin objects SHOULD be judged by the same persistence and
Project eligibility criteria. Core ownership may simplify availability, but it
must not excuse an unstable schema. Plugin ownership must not exclude a sound
scientific object merely because it is extension-defined.

Plugins SHOULD NOT independently embed opaque objects into `NDDataset.meta` or
invent ad hoc Project member encodings.

## Architectural principles

1. **Scientific role is separate from persistence status.** A meaningful
   scientific object can remain runtime-only or embedded.
2. **Persistence is a versioned contract.** Writing bytes is not sufficient.
3. **Ownership precedes serialization.** Only clearly owned state should cross
   the persistence boundary.
4. **Export is not persistence.** Conversion to a stable carrier does not make
   the source object persistent.
5. **Project is typed organization, not generic storage.** Membership requires
   stronger guarantees than serializability.
6. **Typed identity is explicit.** Reconstruction must not depend only on field
   shape, container position, or display text.
7. **Runtime services stay runtime-only.** Algorithms, callables, open handles,
   GUI state, and backend instances do not belong in scientific persistence.
8. **Extensions follow the same contract.** Plugin objects require namespaced
   identity, schema ownership, and absent-provider behavior.
9. **Compatibility is designed before admission.** A type should not become
   persistent or Project-compatible first and acquire migration rules later.
10. **Provenance is not object pickling.** Scientific lineage should be modeled
    explicitly rather than inferred from saved estimator state.

## Recommendations

### Official current boundary

The following are officially top-level persistent today:

- `NDDataset` as the primary persistent scientific data object;
- `Project` as the persistent recursive container for datasets and Projects.

The following are officially embedded persistent today:

- `Coord` and `CoordSet` within `NDDataset`;
- `Meta`, units, quantities, masks, timestamps, names, and history as owned
  state within supported persistent objects.

The following are officially runtime-only today:

- analysis and fitting estimators, fitted or unfitted;
- `ResultBase`, `AnalysisResult`, and `FitResult`;
- plugin runtime services and plugin objects without an approved persistence
  contract.

### Future candidates

Result objects MAY become standalone persistent objects after a dedicated
Result persistence RFC defines their schema, value domain, provenance, and
compatibility policy.

Result objects MAY later become Project members, but only after standalone
persistence is proven and a separate Project typed-member RFC is accepted.

Selected plugin scientific objects MAY become persistent and eventually
Project-compatible under the same contracts, with additional absent-provider
governance.

`Coord` and `CoordSet` do not currently need independent top-level persistence.
Their embedded persistence is sufficient unless a concrete standalone use case
demonstrates otherwise.

### Premature work

The following are premature:

- adding `ResultBase` directly to `Project`;
- generalizing `NDIO` into a plugin registry before schemas and failure policy
  are specified;
- treating arbitrary JSON-compatible diagnostics as stable persistent state;
- promising workflow reproducibility from current history or Result fields;
- making provenance depend on serialized estimator internals.

### Work that should probably never be done

SpectroChemPy SHOULD NOT:

- restore generic arbitrary-object storage in `Project`;
- persist live estimator, callable, GUI, figure, file-handle, or service state
  as native scientific objects;
- hide Result payloads in `NDDataset.meta` or private dataset fields;
- silently discard unknown plugin members while loading a Project;
- equate pickleability with a supported persistence contract.

A declarative estimator configuration or workflow specification could become a
separate future object. That would not justify persistence of a live estimator
instance.

## Future work

This RFC should guide, but not combine, the following future RFCs:

1. **Serialization evolution RFC**
   Define type identifiers, schema versions, migrations, validation, security,
   and compatibility windows for native persistence.
2. **Result persistence RFC**
   Define owned fields, allowed value domains, structured provenance, subtype
   schemas, and standalone round-trip semantics.
3. **Project typed-member RFC**
   Define membership lifecycle, naming, dispatch, hierarchy, placeholders, and
   absent-provider behavior without restoring generic storage.
4. **Provenance RFC**
   Separate human-readable history, structured lineage, reproducibility
   metadata, and workflow specifications.
5. **Plugin scientific-object RFC**
   Define namespaced identity, schema ownership, support obligations, registry
   governance, and cross-plugin dependency rules.

The recommended order is serialization foundations, standalone Result
persistence, Project typed members, and then plugin member admission. Provenance
design may proceed in parallel but should be shared by Result and Project work.

## Open questions

- What compatibility window should a native object schema promise?
- Should type identity be embedded in every object payload or supplied by the
  containing schema for embedded components?
- What restricted value domain is sufficient for Result parameters and
  diagnostics?
- Should Result persistence use the existing native archive family or a
  distinct format while the contract is validated?
- What minimum structured provenance belongs to every persistent Result?
- Should Project hold nested objects by value only, or may a future schema
  support explicit references and deduplication?
- What should users be able to inspect when a Project contains an object whose
  plugin is unavailable?
- Which organization approves and retires namespaced plugin type identifiers?
- Is a future declarative workflow specification a scientific object, a
  provenance object, or a separate application artifact?

Until these questions are answered by focused RFCs, this RFC recommends keeping
the current conservative boundary: `NDDataset` and `Project` are top-level
persistent, coordinates and infrastructure values persist within them,
estimators and Results remain runtime-only, and Project remains a typed dataset
hierarchy.
