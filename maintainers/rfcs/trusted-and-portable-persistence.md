# Trusted and Portable Persistence

**Status:** Proposed Maintainer RFC

**Scope:** Definition and conceptual distinction of trusted and portable persistence, their roles in SpectroChemPy, and their relationship to current and future native formats.

**Out of scope:** API design, implementation, format changes, migration mechanics, or code.

**Related documents:**

- `maintainers/rfcs/scientific-object-model-and-persistence-boundaries.md`
- `audit/~serialization-security-and-trust-boundary-audit.md`

---

## Motivation

Today SpectroChemPy faces a conceptual ambiguity that creates practical risks and architectural confusion.

### The problem

SCP and PSCP files present themselves as portable scientific exchange formats. Their JSON-shaped wrapper, ZIP structure, and `.scp` / `.pscp` suffixes suggest they are meant for sharing, archiving, and long-term preservation.

Yet their implementation carries the trust semantics of a Python runtime snapshot:

- Core numerical payloads are transported as Python pickle streams, base64-encoded within JSON;
- A crafted file can execute arbitrary Python code during deserialization before any schema or type validation occurs;
- Users have no indicator that loading an `.scp` file may execute payload-selected Python instructions;
- `NDDataset.load()` and `Project.load()` are presented as normal scientific-workflow operations with no special trust context.

This creates three concrete problems:

1. **Security risk:** Users who exchange `.scp` or `.pscp` files—between collaborators, in repositories, attached to publications—may unknowingly load malicious code. The file extension provides no security boundary.

2. **User confusion:** A scientist viewing `Dataset.load(filename)` has no reason to suspect untrusted-input restrictions. The API suggests this is as safe as opening a `.csv` or `.h5` file, when it is not.

3. **Architectural drift:** Future decisions about Result persistence, plugin objects, and format evolution cannot be made coherently without distinguishing what SCP/PSCP actually promises.

### Why now

Recent analysis has clarified two facts:

- **Audit conclusion:** The arbitrary-code-execution vulnerability is technically credible. The public load path reaches Python pickle deserialization with attacker-controlled payloads before any type or schema validation. This is a high-severity unsafe-deserialization issue in any workflow where `.scp` files are exchanged, downloaded, or obtained from untrusted repositories.

- **Object model RFC conclusion:** SpectroChemPy now recognizes that persistence is a versioned contract with explicit trust boundaries, not merely the ability to write and read bytes. Future objects such as Result and plugin-defined scientific objects will require explicit schema ownership, compatibility rules, and a declared security posture.

Together, these conclusions show that SCP/PSCP's ambiguity can no longer be treated as a minor documentation detail. It must be resolved at the architectural level before broader persistence is designed.

### The clarifying question

At its core, the question is simple:

> **What does SpectroChemPy actually promise when a user opens a file?**

- Does it promise that the file has been authenticated and comes from a trusted source?
- Does it promise that the file contains only data-model constructs, with no executable payload?
- Does it promise both, depending on the format or the caller's intent?
- Does it promise neither, and leave the user to manage trust separately?

This RFC proposes explicit definitions and then answers this question for each context.

---

## Definitions

### Trusted persistence

**Trusted persistence** is a serialization format that prioritizes efficient reconstruction of Python-specific runtime objects over portability or untrusted-input safety.

#### Characteristics

- Requires Python runtime and specific class/module availability for deserialization.
- May reconstruct arbitrary Python object graphs, including live state such as callables, generators, or backend handles.
- Deserialization may execute Python code selected by the serialized payload (e.g., pickle).
- Is explicitly restricted to authenticated, protected input from known sources.
- Is typically used for:
  - interim runtime snapshots during interactive work;
  - developer workflows where all data is internal;
  - archiving a Python process state for debugging or replay in the same environment.

#### Examples

- **pickle:** Python's native serialization format; design accepts arbitrary code execution as a consequence of full object reconstruction.
- **joblib:** Built on pickle; adds compression and efficient persistence of numeric arrays and machine-learning objects. Same trust model as pickle.
- **dill:** Extends pickle to handle more Python constructs (e.g., lambdas, closures). Same trust restrictions as pickle.
- **In-process runtime snapshots:** Saving a fitted scikit-learn estimator, a TensorFlow model, or a live estimator instance within a Python process for later restoration in the same session.

#### Trust model

```
User or administrator    ->    Trusted persistence file    ->    Python runtime

↑
Authenticity is external to the file; source is known and protected
```

The file itself contains no authentication mechanism. Trust is a property of the file's origin, storage, and who is allowed to access it—not the file format.

#### When appropriate

- Loading within a controlled environment where all input is known to be internal.
- Snapshots during development or for operator replay in specific operational contexts.
- Explicit developer workflows that document the trust requirement to users.

#### When not appropriate

- Scientific data exchanged between collaborators, especially unknown collaborators.
- Files attached to publications or submitted to repositories.
- Scenarios where the user cannot verify the file's provenance.
- Any public-facing, multi-user, or untrusted-network scenario.

### Portable persistence

**Portable persistence** is a serialization format that prioritizes untrusted-input safety, cross-language compatibility, and long-term schema stability over the convenience of full object capture.

#### Characteristics

- Uses an explicit, versioned data schema independent of any single language's object model.
- Restricts payload to typed data: numbers, text, arrays, structures—not executable instructions.
- Validation occurs before any data interpretation.
- Deserialization cannot execute code chosen by the payload.
- Designed to be read by multiple programming languages, tools, and systems.
- Typically used for:
  - scientific data exchange between researchers, especially across institutions;
  - archival preservation of scientific results;
  - long-term interoperability beyond any single software ecosystem.

#### Examples

- **netCDF:** Defines dimensions, variables, attributes, and groups with explicit types and scientific conventions. Widely adopted in climate, oceanography, and geophysics. Not Python-specific.
- **xarray with netCDF backend:** Combines labeled dimensions, coordinates, and attributes in a structured, portable model with strong scientific semantics.
- **Zarr:** Chunked, compressed arrays with explicit type and shape. Cloud-friendly, multi-language.
- **HDF5 with scientific conventions:** A typed, hierarchical container format. Portability requires explicit schema conventions (e.g., CF, HDF-EOS) to constrain semantics beyond the bare format. Without external conventions, HDF5 is a flexible low-level storage mechanism, not a portable scientific schema.
- **Protocol Buffers, Parquet, Apache Arrow:** Language-independent schemas with code generation and strong typing.
- **TIFF with EXIF metadata:** Simple image + metadata model, not Python-specific.

#### Trust model

```
Untrusted persistence file    ->    Strict schema + type validation    ->    Data

↑
No executable payload; validation prevents resource exhaustion or misinterpretation
```

The file format itself provides a security boundary by refusing to accept and execute executable instructions.

Portable persistence is a property of the format contract, not of a visual container choice or implementation detail. A ZIP/JSON wrapper is not portable if the payload semantics depend on executable Python object reconstruction.

#### When appropriate

- Any scenario where the file's provenance cannot be fully verified.
- Scientific data that needs to be opened in tools other than SpectroChemPy.
- Publication supplementary materials or archived research data.
- Multi-site collaboration where file exchange is routine and trust cannot be assumed.

#### When not appropriate

- Internal snapshots for replay in a controlled development environment.
- Live estimator state that has no schema or that intentionally includes callables.
- Streaming or caching layers where latency and efficiency override portability.

### Trust, authenticity, and validation are separate concerns

These three properties are often conflated but should be kept distinct:

- **Trust:** The assurance that the file's source is known and its integrity has been verified. This can be achieved through cryptographic signatures, secure delivery, or organizational policy. It is independent of the file format.
- **Portability:** The ability to read the file across different programming languages, systems, and versions. This depends on the schema and codec.
- **Untrusted-input safety:** The guarantee that a file following the schema cannot cause arbitrary code execution or excessive resource consumption. This is a property of the format and parser design.

A trusted file can use pickle. A portable file should not require pickle. A format can be both trusted and portable if it is signed and the consumer validates the signature. But a file is safer by default when its format itself prevents code execution—even if the user must still trust the file source to avoid malicious data content.

---

## Current SCP/PSCP positioning

### What SCP/PSCP is mechanically

SCP and PSCP are hybrid formats:

- **Container:** ZIP archive.
- **Layout:** Single JSON member plus optional `.npy` binary members.
- **Object state:** JSON with Python-specific encoding for complex and array values.
- **Array/complex encoding:** Base64-encoded pickle streams, embedded within JSON fields marked with `"__class__": "NUMPY_ARRAY"` or `"__class__": "COMPLEX"`.

The ZIP structure and JSON appearance suggest portability. The pickle encoding makes that appearance misleading.

### What SCP/PSCP trust boundary is today

**SCP and PSCP are trusted Python persistence formats.**

They are not portable scientific exchange formats in the sense defined above, because:

- Arbitrary Python code can execute during deserialization.
- The format is not language-independent or tool-independent.
- No version or schema negotiation occurs before pickle is invoked.
- Users have no mechanism to reject unsafe historical payloads without losing access to legitimate old files.

### Compatibility dependencies

This categorization is not arbitrary; it reflects a genuine dependency on large existing archives:

- The modern JSON/base64 layout has existed since version 0.2.0 (January 2021) and is present in all releases from 0.2.10 onward (February 2021).
- Every non-empty `NDDataset` written in the modern format depends on pickle deserialization for its primary numerical data.
- Every `Project` containing such datasets depends on the same mechanism transitively.
- These files represent years of user work, publications, and accumulated research data.

A naive removal of pickle support would make this entire corpus unreadable, and many of those files cannot be regenerated because the original data has been lost or is inaccessible.

### The architectural conclusion

SCP/PSCP occupies a necessary and useful role: **it is SpectroChemPy's native trusted persistence format for runtime objects and their structures.**

As such, it is:

- valuable for intermediate work, development snapshots, and archiving complete runtime state;
- appropriately restricted to known-origin files;
- not appropriate for untrusted interchange without additional authentication;
- not appropriate as the sole or default file format for scientific archival or multi-site collaboration.

This conclusion is strengthened by recent security analysis and must now be made explicit.

---

## Relation to the Scientific Object Model

The scientific-object RFC established that persistence is a versioned contract with explicit ownership, schema, and compatibility rules. It also established that not all scientific objects need to be persistent, and not all persistent objects need to be portable.

### Object categories and persistence axes

Recall the RFC's two independent axes:

| **Role** | **Examples** |
|---|---|
| Scientific data object | `NDDataset` |
| Structural scientific component | `Coord`, `CoordSet` |
| Organizational container | `Project` |
| Computational object | `PCA`, `NMF`, `MCRALS`, analysis estimators |
| Scientific result record | `ResultBase`, `AnalysisResult`, `FitResult` |
| Infrastructure value | `Meta`, units, quantities, masks, history |

| **Persistence Status** | **Examples** |
|---|---|
| Top-level persistent | `NDDataset`, `Project` |
| Embedded persistent | `Coord`, `CoordSet`, `Meta`, units, history (within `NDDataset`) |
| Runtime-only | estimators, result objects |
| Persistence candidate | future plugin objects, future Result standalone persistence |

### Current trusted persistence contracts

Today, **trusted persistence** is used for:

- `NDDataset`: top-level persistent via native `.scp` files.
- `Project`: top-level persistent via native `.pscp` files.
- `Coord`, `CoordSet`, `Meta`, units, masks, history: embedded persistent within `NDDataset`.

These objects are reconstructed from pickle-encoded payloads within the JSON/base64 encoding, and their persistence therefore inherits the trusted-persistence contract.

### Future portable persistence candidates

Objects that may later become **portable persistent** include:

- **`NDDataset` export format:** An optional `.h5` or netCDF export path that does not use pickle and can be read by non-Python tools. This would be export, not replacement of native persistence.
- **Standalone `Result` objects:** If Results become independently persistent (as discussed in the scientific-object RFC), they may use a portable format separate from the native archive family.
- **Plugin scientific objects:** A plugin's persistent object may define its own portable schema rather than inheriting the native format.
- **Project interchange:** A future portable Project representation might export nested datasets in a portable codec, possibly with references rather than full embedding.

Each of these candidates would require explicit schema design, security review, and a deliberate decision to move from trusted to portable persistence or to support both in parallel.

### Plugin implications

The RFC's principle stands: plugin objects follow the same contracts as core objects. A plugin-defined scientific object may be:

- trusted persistent (using pickle or other Python-specific mechanisms, with explicit trust labeling);
- portable persistent (using a schema-driven, code-execution-free codec);
- runtime-only (with explicit exports if interchange is needed).

The choice should be driven by the plugin's use cases and scientific domain, not by technological convenience.

---

## Loading policy principles

When a user invokes `NDDataset.load(...)`, `Project.load(...)`, or the generic `load(...)` function, what should they reasonably expect?

The following principles are proposed:

### 1. Safe by default

The default load path should not execute arbitrary code from the file. If a user has not explicitly opted into a legacy or special mode, they should not be surprised by code execution.

**Implication:** A future default load path should prefer schemas that do not require pickle or equivalent mechanisms.

### 2. Explicit trust boundaries

When a format requires trust (e.g., because it may execute code), that requirement must be:

- documented in the API;
- discoverable in error messages;
- opted in to explicitly, not assumed from file extension or syntax.

**Implication:** If `.scp` or `.pscp` files continue to use pickle for legacy compatibility, loading them should be clearly documented as requiring trust and should offer a way for users to verify file source or reject untrusted files.

### 3. Compatibility requires transparency

Existing `.scp` and `.pscp` files cannot be discarded. Users who have legitimate archives created years ago need to be able to load them.

However, compatibility does not mean silently accepting every legacy mechanism. Instead:

- the load path should distinguish between legacy-compatible reads and forward-looking reads;
- users should be able to inspect what a file contains before committing to loading it;
- migration or conversion tools should be available for those who want to convert old archives to a new format.

**Implication:** A future compatibility strategy should support coexistence of trusted and portable formats, with clear labeling of which format a file uses.

### 4. Minimal surprise

Users should not discover that their normal scientific workflow involves executing untrusted code. The API contract should match user expectations.

**Implication:** If `.scp` loading requires trust, that should be mentioned prominently in documentation and teaching examples. Alternatives (portable formats) should be highlighted for multi-site collaboration and archival scenarios.

### 5. Gradual migration

Existing archives are a reality. Users cannot all migrate overnight. But migration should be possible and should be supported with tools and clear guidance.

**Implication:** Any new portable format should coexist with the current trusted format for a transition period, allowing users to convert at their own pace.

---

## Compatibility policy

Four conceptual options are available for the future:

### Option A: SCP remains trusted

SCP/PSCP remain the native format with explicit trusted-persistence semantics. Pickle support is maintained for reading legacy files. The security boundary is explicitly documented.

**Advantages:**
- No format migration needed; backward compatibility is perfect.
- Existing archives and tools remain functional.
- Efficient representation of complex runtime structures.

**Disadvantages:**
- Users who are unaware of the trust requirement may exchange files unsafely.
- Export to portable formats requires a separate step.
- Cannot be the default for untrusted-input scenarios.

### Option B: SCP transitions to portable

A new portable encoding is designed and becomes the default. Legacy `.scp` files with pickle are still readable in a special mode, but future archives use the portable codec.

**Advantages:**
- New archives are safe by default.
- Files can be exchanged more freely.
- Stronger long-term data preservation.

**Disadvantages:**
- Files written after the transition cannot be read by older SpectroChemPy versions.
- Conversion of legacy files requires a trusted transformation step.
- Some runtime metadata may be lost if the portable schema is simplified.

### Option C: Coexistence

SCP remains trusted and continues to use pickle. A new portable format (e.g., `.scp.h5` or `.scp.nc`) coexists alongside it. Users choose which to write and read.

**Advantages:**
- Users can choose portability when they need it.
- Existing tools and archives continue to work.
- Legacy files are never forced to migrate.
- Both use cases are supported without compromise.

**Disadvantages:**
- Two formats must be maintained and documented.
- Users must understand the difference and make informed choices.
- Testing and validation must cover both paths.

### Option D: New portable format, SCP archived

SCP is declared legacy, archived, and no longer written. A new portable format becomes the default. Old `.scp` files can be read with a legacy loader but are not updated.

**Advantages:**
- Clean break; one format to maintain going forward.
- Portable files are created by default.
- Documentation and teaching can focus on one path.

**Disadvantages:**
- Large existing corpus becomes legacy.
- Users with old files must either convert or use compatibility mode indefinitely.
- Breaking change for workflows built around `.scp` as the primary format.

### Recommended framing

No single option is universally correct. The choice depends on:

- the size and distribution of existing `.scp` archives;
- the willingness of the community to maintain multiple formats;
- the demand for portable interchange versus the cost of migration;
- policy decisions about archival support and long-term compatibility windows.

This RFC recommends that the choice be made explicitly in a follow-up RFC that can assess these factors with complete information.

---

## Architectural principles

The following principles are proposed to guide all future decisions about persistence:

### 1. Safe by default

All public load operations should refuse arbitrary-code execution unless explicitly opted in. The default should favor schemas that cannot execute payload-selected code.

### 2. Explicit trust boundaries

When a format or operation requires trust (e.g., source authentication), that requirement must be explicit in the API, documentation, and error messages. Trust should never be implicit in a file extension or syntax.

### 3. Persistence requires schema ownership

A persistent object type must have an assigned owner (core, an official plugin, or a named maintainer) who is responsible for schema definition, versioning, migration, and validation. Persistence is not inferred from serializability.

### 4. Export is not persistence

Converting an object to another representation (e.g., exporting a Result as a JSON document or a Project as netCDF) is not the same as implementing persistence. Export can be lossy or format-specific. Persistence is a versioned contract on the original object type.

### 5. Compatibility is designed before admission

An object should not become persistent or Project-compatible first and acquire compatibility rules and migration policy later. Persistence contracts must be specified upfront.

### 6. Portability and trust are independent

A file can be trusted but not portable (pickle), portable but not trusted (unreliable schema), or both (signed, portable format). These properties should be explicitly discussed, not assumed from the format.

### 7. Project membership is not generic storage

Project is a typed container for specific scientific objects, not a bag that accepts any serializable Python object. Future membership must satisfy explicit eligibility criteria, not merely pickleability.

### 8. Extensions follow the same contract

Plugin-defined scientific objects must satisfy the same persistence, compatibility, and schema-ownership requirements as core objects. Plugin status does not excuse a weaker contract.

### 9. Payloads should not hide runtime dependencies

A persistent file should not silently require live callables, open handles, backend services, or imported modules that are not part of the reconstruction contract. Such dependencies make persistence fragile and opaque.

### 10. User expectations guide API design

The API should match what users reasonably expect from its name and signature. `Dataset.load(file)` should not execute arbitrary code without explicit indication that the file must be trusted.

---

## Recommendations

### Immediate recognition

SpectroChemPy should formally recognize that:

**SCP and PSCP are trusted Python persistence formats, not portable scientific exchange formats.**

This recognition should be reflected in:

1. **Documentation:** The distinction between trusted and portable persistence should be introduced early in user documentation. Users should understand what they are choosing when they use `.scp` files.

2. **API contracts:** The security and trust posture of `NDDataset.load()` and `Project.load()` should be documented. The pickle-dependent encoding should be explicitly mentioned.

3. **Error messages:** When loading fails, messages should not hide the fact that the file was examined for Python code; they should indicate whether trust was required.

4. **Plugin guidance:** Plugin developers should be advised that if they define persistent objects, they must declare whether those objects use trusted or portable persistence.

### Forward-looking policy

Before designing new persistent object types, SpectroChemPy should adopt a clear policy:

**Current recommendation:** Option C appears to provide the best balance between security, compatibility, and migration cost:

- SCP remains the trusted native format, explicitly labeled as such.
- A new portable format (or formats) is designed and provided as an option.
- Users and workflows can choose based on their interchange needs.
- Both formats are maintained and documented.

This option balances the reality of existing archives, the need for safe defaults in new work, and the cost of maintaining multiple persistence mechanisms.

### Explicit decisions still needed

This RFC does not make all necessary decisions. The following remain open and should be addressed in focused follow-up RFCs:

1. **Immediate security fix:** Should the current default load path for `.scp` files be changed? If so, how should legacy files be handled? (This is a security remediation question, not a long-term design question.)

2. **Portable format design:** If a portable format is to be supported, what should it be? (netCDF, HDF5, Zarr, or a new schema?)

3. **Result persistence:** Should Result objects become persistent? If so, should they use trusted or portable persistence or both? (Addressed by a future Result persistence RFC.)

4. **Project typed members:** Should Project membership be broadened? If so, what new member types should be supported, and what persistence contract should they follow?

5. **Plugin object governance:** How should plugin-defined scientific objects be admitted to Project or declared as persistent? What schema registry is needed?

---

## Future work

This RFC clarifies concepts and boundaries. It should guide, but not combine with, the following focused RFCs:

1. **Serialization evolution and security RFC**
   - Type identification and versioning for persistent objects.
   - Schema migration and compatibility windows.
   - Trusted vs. portable enforcement in the load path.
   - Resource limits, parser hardening, and validation.

2. **Portable format RFC**
   - Choice of portable codec (netCDF, HDF5, Zarr, etc.).
   - Mapping of SpectroChemPy object model to the portable schema.
   - Coexistence strategy with trusted SCP format.
   - User-facing API and guidance.

3. **Result persistence RFC**
   - Owned fields and allowed value domains for Result objects.
   - Structured provenance representation.
   - Distinction between Results and workflow specifications.
   - Standalone round-trip semantics and testing strategy.

4. **Project typed-member RFC**
   - Membership eligibility criteria and evaluation process.
   - Dispatch, loading, and error handling for unknown member types.
   - Absent-plugin behavior and fallback strategies.
   - Implementation of the typed-member registry.

5. **Plugin object governance RFC**
   - Namespaced type identifiers for plugin objects.
   - Schema ownership and support lifecycle.
   - Cross-plugin dependency and version negotiation.
   - Approval process for persistent plugin objects.

6. **Provenance and lineage RFC**
   - Distinction between history, lineage, and reproducibility metadata.
   - Structured provenance model for analysis workflows.
   - Relationship to Result persistence and workflow specifications.

The recommended order is:

1. Serialization evolution and security (address the current vulnerability and clarify trust boundaries).
2. Portable format (provide safe default for new archives).
3. Result persistence (extend persistence to a new object category with clear contracts).
4. Project typed members (broaden Project governance safely).
5. Plugin object governance and provenance (enable ecosystem expansion).

These RFCs should be sequenced to avoid architectural conflict and to allow each to be reviewed, validated, and optionally revised before the next begins.

---

## Open questions

The following questions remain and should be resolved through implementation experience or follow-up RFCs:

- What compatibility window should a persistent native object schema promise? (e.g., 5 years, 1 major version, indefinite?)
- Should SCP and PSCP use compatible or separate format families?
- Should type identity be embedded in every object payload, or can it be supplied by the container's schema?
- What is the minimal set of structured provenance that every persistent Result should include?
- Should a portable format coexist with trusted SCP indefinitely, or should one replace the other after a transition period?
- How should users convert legacy `.scp` files to a new portable format? Should conversion be automated, manual, or delegated to third-party tools?
- When a Project file references a plugin object that is not installed, what should happen? (hard error, opaque placeholder, metadata-only placeholder, etc.?)
- Who approves and administers the namespaced type-identifier registry for plugin objects?

Until these questions are answered, this RFC recommends maintaining the current conservative boundary: SCP/PSCP as labeled trusted formats, new portable formats as opt-in, and Result and plugin objects as runtime-only or explicitly future-work candidates.

---

## Implementation guidance

This RFC guides architectural decisions but does not implement them. Guidance for whoever undertakes implementation work:

- Treat this document as conceptual framework, not a specification.
- Use this RFC to inform but not constrain follow-up RFCs; they may refine or revise these concepts as needed.
- This RFC should be reviewed and accepted before substantial code work begins on persistence changes.
- Update this RFC if implementation experience reveals gaps or incorrect assumptions.

---

## Summary

SpectroChemPy must make explicit what SCP/PSCP formats are and what they promise to users. This RFC proposes:

1. **Clear definitions** of trusted and portable persistence with examples from the ecosystem.
2. **Explicit recognition** that SCP/PSCP are trusted formats, not portable exchange formats.
3. **Architectural principles** to guide future persistence design.
4. **Coexistence strategy** (Option C) to support both trusted legacy formats and future portable alternatives.
5. **Sequenced follow-up RFCs** to address security, portable formats, Result persistence, Project membership, and plugin governance.

The goal is not to redesign persistence immediately, but to clarify the foundation so that future decisions can be coherent, well-informed, and aligned with SpectroChemPy's commitment to scientific reproducibility, safety, and long-term compatibility.
