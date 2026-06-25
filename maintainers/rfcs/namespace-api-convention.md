# Namespace API Convention

## Status

`ACCEPTED`

## Purpose

Define a clean, stable, and uniform public API for all SpectroChemPy I/O
operations — both reading and writing — that remains valid independently of
whether the implementation lives in the core package or in an optional plugin.

This RFC is **intentionally conservative**.  It does not redesign the I/O
subsystem.  It proposes API consistency so that users see the same surface
regardless of where an operation is implemented.

A namespace represents a **functional domain** (e.g., `jcamp`, `nmr`,
`perkinelmer`), not necessarily a single reader or writer.  The same namespace
may host multiple parsers, exporters, or operations.

Importantly, a namespace is **not necessarily a file format**.  The following
namespaces coexist with the identical API:

- `jcamp` — an open standard;
- `csv` — a generic text format;
- `perkinelmer` — a vendor;
- `nmr` — a scientific domain.

Despite their different natures, they all expose the same surface:
`scp.<namespace>.read(...)` and, when applicable,
`scp.<namespace>.write(...)`.  This uniformity is intentional.

## Motivation

The PerkinElmer `.sp` plugin introduced a short, namespace-based API:

```python
scp.perkinelmer.read("file.sp")
```

This is natural and discoverable, but it is not consistently available across
all namespaces.  Today several public APIs coexist:

```python
# Generic dispatcher
scp.read("file.spa")

# Top-level functions (core)
scp.read_jcamp("file.jdx")
scp.read_omnic("file.spa")
scp.read_opus("file.0")

# Top-level functions (plugins)
scp.read_perkinelmer("file.sp")

# Namespaced functions (plugins)
scp.perkinelmer.read("file.sp")
scp.nmr.read("path/to/fid")
```

This inconsistency makes the plugin boundary visible to users.  A user who
learns `scp.perkinelmer.read(...)` cannot guess whether `scp.jcamp.read(...)`
or `scp.omnic.read(...)` exists.

The goal is to make the namespace API predictable.

## Scope

Inside scope:

- Public I/O API conventions (namespace, function name, signatures).
- Backwards-compatible migration strategy for both readers and writers.
- Documentation and example conventions.
- Impact on core I/O operations, official plugins, and third-party plugins.

Outside scope:

- Internal parser refactoring.
- Moving core readers into plugins (that is a separate architectural campaign).
- `Importer()` implementation changes.
- Plugin discovery mechanics.
- Dynamic replacement of `KNOWN_PLUGIN_READERS` / `KNOWN_PLUGIN_NAMESPACES`.

## Current Situation

### Core namespaces

Core I/O operations expose top-level functions:

```python
# Readers
scp.read_jcamp(...)
scp.read_omnic(...)
scp.read_opus(...)
scp.read_csv(...)
scp.read_matlab(...)
scp.read_spc(...)

# Writers
scp.write_jcamp(...)
scp.write_csv(...)
scp.write_matlab(...)
```

There are no namespaces.  The generic `scp.read(...)` and `scp.write(...)`
dispatch to the appropriate private `_read_*` / `_write_*` methods via the
importer / exporter machinery.

### Plugin namespaces

Official plugins currently expose **both** a top-level alias and a
namespaced function:

```python
# Top-level aliases (via features.py stub or plugin registration)
scp.read_perkinelmer(...)
scp.read_topspin(...)

# Namespaced functions (via PluginNamespace)
scp.perkinelmer.read_perkinelmer(...)
scp.nmr.read_nmr(...)   # or read_topspin, etc.
```

The PerkinElmer plugin additionally exposes a short `read` inside the module
so that `scp.perkinelmer.read(...)` works.

Some core namespaces (e.g., `jcamp`) also provide writers:

```python
scp.write_jcamp(...)
```

but there is no equivalent `scp.jcamp.write(...)` today.

### Redundancy

The following forms are redundant:

```python
scp.perkinelmer.read_perkinelmer(...)  # namespace + full name
scp.nmr.read_nmr(...)                  # namespace + full name
scp.jcamp.write_jcamp(...)             # namespace + full name
```

Once a namespace exists, repeating the operation name inside the method name
provides no additional information.

## Proposed API

### Canonical explicit API

Every namespace SHOULD expose canonical short `read()` and, when applicable,
`write()` functions:

```python
# Readers
scp.jcamp.read(...)
scp.omnic.read(...)
scp.opus.read(...)
scp.csv.read(...)
scp.perkinelmer.read(...)
scp.nmr.read(...)
scp.tensor.read(...)      # if applicable
scp.iris.read(...)        # if applicable
scp.carroucell.read(...)

# Writers (when implemented)
scp.jcamp.write(...)
scp.csv.write(...)
scp.matlab.write(...)
```

The namespace already identifies the domain.  The method name (`read` or
`write`) is sufficient.

A namespace should normally expose only operations that are actually
implemented.  For example, a read-only plugin such as PerkinElmer should expose
`scp.perkinelmer.read(...)` but would not expose `scp.perkinelmer.write(...)`
unless writing is implemented.  Future operations (e.g., `info`, `validate`,
`convert`) may be added when they make sense for the domain.

### Minimal and consistent namespace API

Each namespace should expose a **minimal and consistent API** whenever
applicable.  For example:

```python
scp.jcamp.read("file.jdx")
scp.jcamp.write("file.jdx", dataset)
scp.jcamp.info("file.jdx")
```

The namespace provides context; methods should never repeat the domain name:

```python
# Correct
scp.jcamp.read(...)
scp.jcamp.write(...)

# Incorrect — redundant prefix
scp.jcamp.read_jcamp(...)
scp.jcamp.write_jcamp(...)
```

This rule is extensible and applies to all namespace operations, not only
readers or writers.

### Generic dispatchers

The generic dispatchers remain unchanged and are never deprecated:

```python
scp.read("file.sp")        # auto-detect format
scp.write("file.jdx", dataset)   # auto-detect format
```

These continue to be the recommended APIs when the user wants automatic format
detection.

### Top-level aliases

Existing top-level functions (`scp.read_jcamp`, `scp.read_omnic`,
`scp.read_perkinelmer`, `scp.write_jcamp`, `scp.write_csv`, etc.) become
**compatibility aliases**.  They continue to work indefinitely, but
documentation should progressively prefer the namespaced form.

### Namespaced redundant methods

Methods such as `scp.perkinelmer.read_perkinelmer(...)`,
`scp.nmr.read_nmr(...)`, and `scp.jcamp.write_jcamp(...)` are redundant.
They MAY be retained during the transition, but they SHOULD eventually emit a
`FutureWarning` pointing to the short form.

## Examples

### New user (recommended)

```python
import spectrochempy as scp

# Automatic format detection — always works
ds = scp.read("file.sp")
scp.write("file.jdx", ds)

# Explicit namespace — consistent across core and plugins
ds = scp.perkinelmer.read("file.sp")
ds = scp.jcamp.read("file.jdx")
ds = scp.nmr.read("path/to/fid")
scp.jcamp.write("file.jdx", ds)
```

### Existing code (still valid)

```python
# All of these continue to work
ds = scp.read("file.sp")
ds = scp.read_perkinelmer("file.sp")
ds = scp.perkinelmer.read_perkinelmer("file.sp")   # redundant, still valid

scp.write_jcamp("file.jdx", ds)
scp.jcamp.write_jcamp("file.jdx", ds)               # redundant, still valid
```

### Plugin author

A third-party plugin providing a reader for `.xyz` files should register:

```python
# Preferred public API
scp.xyz.read("file.xyz")
```

If the plugin also implements writing, it should expose:

```python
scp.xyz.write("file.xyz", dataset)
```

The plugin MAY also request top-level aliases (`scp.read_xyz`, `scp.write_xyz`)
for consistency, but the namespace form is the canonical API.

## Migration Strategy

The migration is split into four phases.  No public code is broken during any
phase.

### Phase 1 — Introduce namespaces everywhere

- Add lightweight namespace modules or `PluginNamespace` entries for **all**
  core I/O domains (`jcamp`, `omnic`, `opus`, `csv`, `matlab`, `spc`, etc.).
- Each namespace exposes `read()` and, when a writer exists, `write()`
  functions that forward to the existing implementation.
- Plugin namespaces follow the same pattern.
- Nothing is deprecated.
- Documentation is updated to mention the new forms as alternatives.

Timeline: one minor release.

### Phase 2 — Update documentation and examples

- The namespace API becomes the **canonical documentation target**.  User guide,
  tutorials, and gallery examples progressively use `scp.<namespace>.read(...)`
  and `scp.<namespace>.write(...)` as the explicit forms.
- Top-level `scp.read_<name>(...)` and `scp.write_<name>(...)` are still
  documented but marked as the legacy convenience aliases.
- The generic `scp.read(...)` and `scp.write(...)` remain the primary
  recommended APIs.

Timeline: one to two minor releases after Phase 1.

### Phase 3 — Warn on redundant namespaced methods

- Redundant namespaced methods such as `scp.perkinelmer.read_perkinelmer(...)`
  or `scp.jcamp.write_jcamp(...)` emit a `FutureWarning`:

  ```text
  FutureWarning: scp.perkinelmer.read_perkinelmer is redundant;
  use scp.perkinelmer.read instead.
  ```

- Top-level aliases (`scp.read_perkinelmer`, `scp.write_jcamp`) do **not**
  warn yet.

Timeline: at least two minor releases after Phase 1.

### Phase 4 — Evaluate whether deprecating top-level aliases is worthwhile

- Evaluate whether `scp.read_<name>(...)` and `scp.write_<name>(...)` aliases
  should emit `FutureWarning`.
- This is **optional and deferred**.  The aliases are lightweight and
  widely used; removing them may not be worth the breakage.
- If deprecated, the deprecation cycle MUST be at least two minor releases
  before removal.

The RFC explicitly leaves open the possibility that top-level aliases remain
indefinitely.  The maintenance cost is near zero, so indefinite retention is
a valid and likely outcome.

## Compatibility Considerations

### Public API preservation

Per `AGENTS.md` and `CONTRIBUTING.md`, public APIs and backward compatibility
must be preserved.  This RFC respects that by:

- Never removing existing functions.
- Only adding new namespace paths.
- Using `FutureWarning` (not hard errors) for any discouraged forms.
- Keeping the generic `scp.read(...)` and `scp.write(...)` untouched.

### Serialization compatibility

No serialization format is affected.  This is purely a public API surface
change.

### Warning behavior

The project convention uses `@deprecated` and `FutureWarning` for deprecation.
Any warnings introduced by this RFC must follow that convention and be covered
by tests.

## Architecture Considerations

### API vs. implementation

This RFC **strictly separates** API from implementation.

Creating `scp.jcamp.read(...)` does **not** mean JCAMP becomes a plugin.
It means the core package exposes a namespace that wraps the existing
`read_jcamp` implementation.

Likewise, migrating an operation from core to plugin in the future should not
change the public API.  Only the implementation location changes.

### Namespace stability

A namespace should remain **stable even if the underlying implementation
changes**.  For example:

```python
scp.perkinelmer.read("file.sp")
```

might today be implemented by `read_perkinelmer.py` and tomorrow by
`perkinelmer_reader_v2.py` — with **no user-visible change**.  The namespace
shields users from implementation churn.

### Core namespace implementation

Core namespaces can be implemented as lightweight forwarders:

```python
# In spectrochempy/core/readers/__init__.py or similar
class _JcampNamespace:
    def read(self, *paths, **kwargs):
        from spectrochempy.core.readers.read_jcamp import read_jcamp
        return read_jcamp(*paths, **kwargs)

    def write(self, path, dataset, **kwargs):
        from spectrochempy.core.writers.write_jcamp import write_jcamp
        return write_jcamp(path, dataset, **kwargs)

# Registered on the scp package
scp.jcamp = _JcampNamespace()
```

No internal refactoring is required.

### Plugin namespace implementation

Plugin namespaces already work via `PluginNamespace`.  The only change is that
plugin authors should prefer exposing `read` and, when applicable, `write`
(in addition to any legacy `read_<name>` / `write_<name>`) inside their
module, and register them in `register_readers`.

The PerkinElmer plugin already demonstrates this for reading:

```python
def __getattr__(name):
    if name in ("read", "read_perkinelmer", "read_sp"):
        from .read_perkinelmer import read_perkinelmer
        return read_perkinelmer
```

A plugin that also implements writing would similarly expose `write`.

### Filetype registration

The `Importer` protocol dispatch (`.sp` → `perkinelmer`) is independent of
the namespace API.  It continues to work exactly as today.

### Static registries

The `KNOWN_PLUGIN_READERS` and `KNOWN_PLUGIN_NAMESPACES` static lists in
`features.py` are used only for missing-plugin error messages.  They are
outside the scope of this RFC.

A future architectural RFC may propose replacing them with dynamic entry-point
discovery.  That work is intentionally separate.

## Impact on Ecosystem

### Official plugins

Official plugins (nmr, iris, tensor, carroucell, perkinelmer) should adopt the
`read` convention in their next releases, and `write` when applicable.
Legacy redundant namespaced methods may be retained with `FutureWarning`.

### Third-party plugins

Third-party plugins are encouraged to expose `scp.<name>.read(...)` and,
when applicable, `scp.<name>.write(...)` as the canonical API.  The plugin
developer guide should document this convention.

### Future operations

Any new I/O operation (core or plugin) should expose the namespace form from day
one.  A namespace should normally expose only the operations it implements
(`read`, `write`, or both), but additional domain-specific operations (e.g.,
`info`, `validate`, `convert`) may be added when relevant.  Top-level aliases
may be added for consistency, but they are secondary.

### Documentation

The user guide should present the following hierarchy:

1. **Primary**: `scp.read(...)` / `scp.write(...)` — automatic format detection.
2. **Explicit**: `scp.<namespace>.read(...)` / `scp.<namespace>.write(...)` —
   when you know the domain.
3. **Legacy**: `scp.read_<name>(...)` / `scp.write_<name>(...)` — still works,
   documented for compatibility.

The namespace API becomes the **canonical documentation target**.  All new
documentation, tutorials, and examples should use the namespace form even
though the legacy aliases remain available.

## Open Questions

1. **Should all core I/O domains receive a namespace in Phase 1, or only the
   most commonly used ones?**

   Recommendation: all core operations that have a `read_*` or `write_*`
   function should receive a namespace for consistency.

2. **Should the namespace be a real module or a dynamic namespace object?**

   For plugins, `PluginNamespace` already handles this.  For core operations,
   a lightweight object or module is acceptable.  The exact mechanism is an
   implementation detail.

3. **Should `scp.read_dir`, `scp.read_zip`, and similar generic utilities also
   get namespaces?**

   They are generic utilities rather than domain-specific operations.  This RFC
   does not propose namespaces for them, but it does not forbid it either.

4. **What is the deprecation timeline for redundant namespaced methods?**

   At least two minor releases after they are marked with `FutureWarning`.
   The exact schedule is left to the maintainers.

5. **Should top-level aliases ever be deprecated?**

   This RFC deliberately leaves this open.  The aliases are harmless,
   widely used, and their maintenance cost is near zero; indefinite retention
   is a valid and likely option.

6. **How does this interact with the vendor-readers migration roadmap?**

   The vendor-readers migration (`maintainers/roadmap/vendor-readers-migration.md`)
   moves implementation from core to plugin.  This RFC ensures that such
   migrations do not change the public API.  The two efforts are complementary.

7. **How should a namespace indicate that it does not support writing?**

   Simply by not exposing `write()`.  Attempting `scp.perkinelmer.write(...)`
   on a read-only namespace should raise `AttributeError`, which is the
   standard Python behavior for missing attributes.

## Recommended Implementation Phases

| Phase | Deliverable | Breaking? |
|-------|-------------|-----------|
| 1 | Add `scp.<namespace>.read` and `scp.<namespace>.write` namespaces for all core and plugin domains. | No |
| 2 | Update docs/examples to prefer namespace form. | No |
| 3 | Add `FutureWarning` to redundant namespaced methods. | No |
| 4 | *Optional* — evaluate whether `FutureWarning` for top-level aliases is worthwhile. | No |

## Relationship to Existing Work

- **`maintainers/roadmap/vendor-readers-migration.md`** — the migration moves
  implementation; this RFC stabilizes the public API across that boundary.
- **`maintainers/rfcs/reader-metadata-normalization-contract.md`** — defines
  what readers produce and how metadata is normalized; this RFC defines how
  users invoke readers and writers.
- **`docs/sources/devguide/plugins/api_policy.rst`** — already establishes that
package-level namespaces are for I/O and object creation.  This RFC applies
that policy uniformly to all I/O operations.
- **`maintainers/architecture/reader-normalization-architecture.md`** — defines
  normalization of imported data; this RFC is the user-facing counterpart.

## Summary

The canonical explicit API should be:

```python
scp.<namespace>.read(...)
scp.<namespace>.write(...)
```

The generic dispatchers remain:

```python
scp.read(...)
scp.write(...)
```

Top-level aliases are compatibility shims that may remain indefinitely.
Redundant namespaced methods should eventually warn.

Each namespace should expose a minimal and consistent API without redundant
prefixes.  A namespace should normally expose only operations that are
implemented, but additional domain-specific operations may be added when
relevant.

Implementation is lightweight, requires no internal refactoring, and preserves
full backward compatibility.

---

*Drafted: 2026-06-25*
*Related issue: #897 (PerkinElmer plugin)*
*Related roadmap: vendor-readers-migration*
