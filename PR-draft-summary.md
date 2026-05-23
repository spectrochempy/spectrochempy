# PR: Further decouple core from domain-specific plugin semantics

> This file will be removed before merging.

## Roadmap context

The `plugins` branch introduced official plugins (NMR, IRIS, Cantera) with a working plugin
system.  This PR takes the next incremental step: **making the core domain-neutral** so that
domain-specific concepts live only in their respective plugins.

### Architectural principle

The core exposes generic extension mechanisms but must not know domain-specific concepts
from plugins.  In particular, the core must not contain references to:

- TopSpin / Bruker NMR semantics
- NMR-specific ppm↔Hz conversion

### What this PR achieves

| Change | Status |
|--------|--------|
| `set_nmr_context` moved from `core/units/__init__.py` to `plugins/spectrochempy-nmr/src/spectrochempy_nmr/units.py` | ✅ Merged from #948 |
| Generic `register_unit_contexts()` plugin hook (hooks, manager, registries, registry) | ✅ Merged from #948 |
| `NDArray.ito()` asks plugin registry for an applicable unit context instead of hardcoding NMR | ✅ Merged from #948 |
| Deprecated shim in core (`spectrochempy.core.units.set_nmr_context`) | ✅ Merged from #948 |
| Plugin tests `plugins/spectrochempy-nmr/tests/test_units.py` | ✅ Merged from #948 |
| Roadmap extended with "Further decoupling" section | ✅ Done |
| **New generic `register_handlers()` hook** — single dict-based override mechanism | ✅ This PR |
| `Coord.reversed` dispatches via `"coord.reversed"` handler (NMR plugin provides ppm reversal) | ✅ This PR |
| `coord.py` fallback comment fixed — ppm is now documented as a *generic spectroscopic convention* (no longer "NMR-specific") | ✅ This PR |
| `concatenate` **full** TopSpin metadata extraction moved to NMR plugin via `"concatenate.extract_metadata"` handler | ✅ This PR |
| `concatenate` postprocess simplified — no longer checks `origin == "topspin"` in core | ✅ This PR |
| `HandlerRegistry.register_handler()` warns on override collision (log warning, last wins) | ✅ This PR |
| Dedicated tests for `register_handlers`, `HandlerRegistry`, dispatch, collisions, fallback | ✅ This PR |
| `Coord.larmor` (trait + property + init kwarg) removed from core — frequency metadata now stored via `coord.meta["acquisition_frequency"]` | ✅ This PR |
| `fft.py` no longer assigns `newcoord.larmor` — uses `meta["acquisition_frequency"]` instead | ✅ This PR |
| NMR plugin reads acquisition frequency from `coord.meta` via renamed helpers | ✅ This PR |
| `_ExecutionPlan.select()` / `execute()` dispatch through `"ndmath.execution_branch"` and `"ndmath.execute"` handlers | ✅ This PR |
| NMR plugin registers `ndmath.*` handlers replicating quaternion detection/decomposition | ✅ This PR |

### What this PR does NOT do (future PRs)

These items still have domain coupling in core and need hook mechanisms before they can move:

| Core file | Coupling | Needed hook |
|-----------|----------|-------------|
| `core/readers/importer.py` | TopSpin-specific protocol/download logic | `file.resolve` or reader-side file resolution |
| `utils/file.py` | `_topspin_check_filename`, `.topspin` extension | `file.resolve` handler |
| `core/readers/filetypes.py` | `"topspin"` entry in FileTypeRegistry | Plugin reader registration should populate this |
| `processing/fft/fft.py` | `is_nmr`, `ppm=True`, encoding modes | `fft.preprocess` / `fft.postprocess` handlers |
| `processing/fft/phasing.py` | Entire module (NMR spectral processing) | Move to plugin |
| `processing/fft/shift.py` | Entire module (NMRGLUE adaptation) | Move to plugin |
| `core/dataset/arraymixins/ndmath.py` | quaternion execution (handlers exist but core fallback still uses `quat_as_complex_array`/`as_quaternion`) | Generic numeric backend interface |
| `core/dataset/basearrays/ndcomplex.py` | Quaternion properties, `is_quaternion`, `set_quaternion` | Generalize or move |
| `core/dataset/basearrays/ndarray.py` | `typequaternion` dtype check for `is_quaternion` property | Generalize or move |
| `processing/fft/fft.py` | Quaternion FFT branches (`is_quaternion`, quaternion-aware FFT) | Move to plugin |
| `utils/quaternion.py` | Full module (optional quaternion import, helper functions) | Move to plugin or generic backend |

### Quaternion decoupling status

This PR prepares quaternion decoupling by adding `"ndmath.execution_branch"` and
`"ndmath.execute"` handler hooks, but does **not** fully extract quaternion from core.
Quaternion-specific code remains in:

| Location | What remains | Why deferred |
|----------|-------------|--------------|
| `ndmath.py` | `_ExecutionPlan.QUATERNION` branch, imports of `typequaternion`/`as_quaternion`/`quat_as_complex_array`, `is_quaternion` detection in `_preprocess_op_inputs` | Need a generic numeric backend interface first |
| `ndcomplex.py` | `is_quaternion`, `set_quaternion`, `_make_quaternion`, quaternion-aware properties | Deep integration with array slicing/dtype dispatch |
| `ndarray.py` | `typequaternion` dtype check for `is_quaternion` property | Core dtype infrastructure |
| `fft.py` | Quaternion-aware FFT branches (`typequaternion` imports, `is_quaternion` checks) | FFT module needs full plugin extraction first |
| `utils/quaternion.py` | Entire module (optional `numpy-quaternion` wrapper) | Must decide: move to NMR plugin or create generic backend |

A follow-up PR to define a **generic numeric backend interface** (allowing plugins to
register custom dtype execution paths) is the recommended next step.

### Notable design decisions

- **Handler collision policy**: if two plugins declare the same handler name, the last-registered plugin wins (with a log warning). This is acceptable for an override mechanism where plugins are loaded in a deterministic order.
- **ppm in `Coord.reversed`**: the `"ppm"` fallback is retained as a *generic spectroscopic convention*, not NMR-specific. The NMR plugin handler can still override this (e.g. return `None` to fall through, or `True`/`False`).
- **`concatenate.extract_metadata`** is a separate handler from `concatenate.postprocess` because the extraction requires access to `datasets` pre-mutation (while postprocess receives the constructed `out` object).

### The `register_handlers()` hook design

Instead of adding one `@hookspec` method per concern, a single `register_handlers()` method
returns a `dict[str, Callable]`:

```python
class NMRPlugin(SpectroChemPyPlugin):
    def register_handlers(self):
        return {
            "coord.reversed": _nmr_coord_reversed,
            "concatenate.postprocess": _nmr_concat_postprocess,
        }
```

Core code dispatches lazily:

```python
handler = registry.get_handler("coord.reversed")
if handler is not None:
    result = handler(self)
    if result is not None:
        return result
# fall through to default logic
```

This keeps the core API surface small and makes it trivial to add new extension points
without modifying the plugin base class.

### Files changed

```
M  PR-draft-summary.md                                                        # this document
M  docs/sources/userguide/plugins_roadmap.rst                                # roadmap section
M  plugins/spectrochempy-nmr/src/spectrochempy_nmr/__init__.py               # handlers (extract + postprocess + coord.reversed) + ndmath handlers + larmor→meta
M  plugins/spectrochempy-nmr/src/spectrochempy_nmr/read_topspin.py           # larmor→meta["acquisition_frequency"]
A  plugins/spectrochempy-nmr/src/spectrochempy_nmr/units.py                  # moved set_nmr_context
A  plugins/spectrochempy-nmr/tests/test_units.py                             # NMR unit context tests
M  plugins/spectrochempy-nmr/tests/test_units.py                             # larmor→meta in test
M  src/spectrochempy/core/dataset/arraymixins/ndmath.py                      # _ExecutionPlan handler dispatch + removed larmor refs
M  src/spectrochempy/core/dataset/coord.py                                   # removed larmor trait/property/init, handler dispatch, generic comment
M  src/spectrochempy/core/dataset/basearrays/ndarray.py                      # plugin unit-context lookup
M  src/spectrochempy/core/units/__init__.py                                  # deprecation shim (unchanged)
M  src/spectrochempy/plugins/hooks.py                                        # register_handlers hookspec + ndmath handler docs
M  src/spectrochempy/plugins/manager.py                                      # _collect_handlers
M  src/spectrochempy/plugins/registries.py                                   # HandlerRegistry + override collision warning
M  src/spectrochempy/plugins/registry.py                                     # compose + forward
M  src/spectrochempy/processing/fft/fft.py                                   # larmor→meta["acquisition_frequency"]
M  src/spectrochempy/processing/transformation/concatenate.py                # handler dispatch (extract_metadata + postprocess)
M  tests/test_core/test_dataset/test_coord.py                                # larmor→meta in test
M  tests/test_core/test_units/test_units.py                                  # removed test_nmr_context
M  tests/test_plugins/test_registry.py                                       # HandlerRegistry unit tests + PluginRegistry forwarding tests
M  tests/test_plugins/test_integration.py                                    # handler collection, dispatch, collisions, edge cases
```

### Validation

```
pytest plugins/spectrochempy-nmr/tests           → 22 passed
pytest tests/test_plugins/                       → 227 passed
pytest tests/test_plugins/ -k handler            → 17 passed (new handler tests)
pytest tests/test_core/test_dataset/test_coord.py → 54 passed
pytest tests/test_core/test_units/test_units.py   → passed
pytest .../test_concatenate.py                    → 2 passed
python -m py_compile <all touched files>          → OK
pre-commit run --all-files                        → passed
```
