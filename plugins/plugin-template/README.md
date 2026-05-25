# spectrochempy-myplugin

A plugin for [SpectroChemPy](https://github.com/spectrochempy/spectrochempy).

## What this plugin demonstrates

This template shows how to build a SpectroChemPy plugin that can:

- **Register file readers and writers** — Add support for custom file formats via `register_readers()` and `register_writers()`.
- **Expose dataset accessors** — Add methods or namespace-style accessors on `NDDataset` objects via `register_accessors()`.
- **Override core handlers** — Intercept core behaviour (importer, math, coordinates, concatenation) via `register_handlers()`.
- **Provide unit contexts** — Register custom Pint unit-conversion contexts via `register_unit_contexts()`.
- **Declare optional dependencies** — Use `requires` to specify packages that will be checked at load time.

## Installation

```bash
pip install spectrochempy-myplugin
```

If your plugin has optional heavy dependencies, list them in `requires` and they will be validated when SpectroChemPy loads the plugin.

## Development

```bash
git clone <repo-url>
cd spectrochempy-myplugin
pip install -e ".[dev]"
```

## Testing

Run the plugin tests:

```bash
python -m pytest tests/ -v
```

The template tests verify that all declared contributions (readers, writers, accessors, handlers, unit contexts) are correctly registered.

## Plugin structure

```
src/plugin_name/
    __init__.py          # Plugin class + operational methods
    my_analysis.py       # Optional analysis module (lazy-imported)
tests/
    test_plugin.py       # Registration and lifecycle tests
pyproject.toml           # Package metadata + entry point
```

The entry point in `pyproject.toml` tells SpectroChemPy where to find the plugin class:

```toml
[project.entry-points."spectrochempy.plugins"]
myplugin = "plugin_name:MyPlugin"
```

## License

CeCILL-B
