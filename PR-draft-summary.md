# PR: Further decouple core from domain-specific plugin semantics

> This file will be removed before merging.

## Roadmap context

The `plugins` branch introduced official plugins (NMR, IRIS, Cantera) with a working plugin
system.  This PR takes the next incremental step: **making the core domain-neutral** so that
domain-specific concepts live only in their respective plugins.

### Architectural principle

The core exposes generic extension mechanisms but must not know domain-specific concepts
from plugins.  In particular, the core must not contain references to:

- NMR, `larmor`, `nmr.larmor`
- TopSpin / Bruker NMR semantics
- NMR-specific ppm↔Hz conversion
- `Quaternion` (if only useful for NMR)

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
| `concatenate` dispatches via `"concatenate.postprocess"` handler (NMR plugin provides metadata coords) | ✅ This PR |

### What this PR does NOT do (future PRs)

These items still have domain coupling in core and need hook mechanisms before they can move:

| Core file | Coupling | Needed hook |
|-----------|----------|-------------|
| `core/readers/importer.py` | TopSpin-specific protocol/download logic | `file.resolve` or reader-side file resolution |
| `utils/file.py` | `_topspin_check_filename`, `.topspin` extension | `file.resolve` handler |
| `core/readers/filetypes.py` | `"topspin"` entry in FileTypeRegistry | Plugin reader registration should populate this |
| `processing/fft/fft.py` | `is_nmr`, `ppm=True`, encoding modes, larmor assignment | `fft.preprocess` / `fft.postprocess` handlers |
| `processing/fft/phasing.py` | Entire module (NMR spectral processing) | Move to plugin |
| `processing/fft/shift.py` | Entire module (NMRGLUE adaptation) | Move to plugin |
| `utils/quaternion.py` | Bruker conventions | Move to plugin or generalize |
| `core/dataset/basearrays/ndcomplex.py` | Quaternion handling | Generalize or move |

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
M  docs/sources/userguide/plugins_roadmap.rst          # roadmap section
M  plugins/spectrochempy-nmr/src/spectrochempy_nmr/__init__.py  # handlers + unit context
A  plugins/spectrochempy-nmr/src/spectrochempy_nmr/units.py     # moved set_nmr_context
A  plugins/spectrochempy-nmr/tests/test_units.py                # NMR unit context tests
M  src/spectrochempy/core/dataset/coord.py                      # handler dispatch
M  src/spectrochempy/core/dataset/basearrays/ndarray.py         # plugin unit-context lookup
M  src/spectrochempy/core/units/__init__.py                     # deprecation shim
M  src/spectrochempy/plugins/hooks.py                           # register_handlers hookspec
M  src/spectrochempy/plugins/manager.py                         # _collect_handlers
M  src/spectrochempy/plugins/registries.py                      # HandlerRegistry
M  src/spectrochempy/plugins/registry.py                        # compose + forward
M  src/spectrochempy/processing/transformation/concatenate.py   # handler dispatch
M  tests/test_core/test_units/test_units.py                     # removed test_nmr_context
```

### Validation

```
pytest plugins/spectrochempy-nmr/tests           → 22 passed
pytest tests/test_plugins/                       → 104+ passed
pytest tests/test_core/test_dataset/test_coord.py → 54 passed
pytest tests/test_core/test_units/test_units.py   → passed
python -m py_compile <all touched files>          → OK
pre-commit run --all-files                        → passed
```
