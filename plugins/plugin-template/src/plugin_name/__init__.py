# ruff: noqa: PLC0415 — defer imports in plugin methods to avoid startup cost
"""
spectrochempy-myplugin — A SpectroChemPy plugin.

Replace this module docstring with your plugin's description.

For SpectroChemPy official plugins only:
- Place a conda recipe file (recipe.yaml or meta.yaml) in the plugin root
  so the monorepo CI can discover and build it automatically.
- Third-party plugins may omit the recipe and publish via their own CI.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from spectrochempy.api.plugins import CORE_PLUGIN_API_VERSION
from spectrochempy.api.plugins import PluginCapability
from spectrochempy.api.plugins import SpectroChemPyPlugin

if TYPE_CHECKING:
    from spectrochempy import NDDataset


# ------------------------------------------------------------------
# Class-based accessor (optional)
# ------------------------------------------------------------------
# If your plugin exposes namespace-style accessors (e.g. dataset.myplugin.foo),
# define a class here and register it via register_accessors() below.
# The core will instantiate it with the dataset when accessed.

# class MyAccessor:
#     """Example dataset accessor."""
#
#     def __init__(self, dataset: NDDataset) -> None:
#         self._dataset = dataset
#
#     def do_something(self) -> str:
#         """Return a friendly greeting."""
#         return f"Hello from myplugin on a {type(self._dataset).__name__}!"


class MyPlugin(SpectroChemPyPlugin):
    """Minimal SpectroChemPy plugin example."""

    name = "myplugin"
    version = "0.1.0"
    description = "My SpectroChemPy plugin"
    spectrochempy_min_version = "0.9.0"
    PLUGIN_API_VERSION = CORE_PLUGIN_API_VERSION
    capabilities = [
        PluginCapability.READER,
        PluginCapability.WRITER,
        PluginCapability.ANALYSIS,
        PluginCapability.ACCESSOR,
    ]

    # Optional pip-style dependencies.
    # Plugins with missing deps are marked FAILED with a clear message.
    # Example: requires = ["scipy>=1.10", "pydantic"]
    requires: list[str] = []

    # ------------------------------------------------------------------
    # Declarative hooks — auto-collected by PluginManager.
    #
    # Available hooks:
    #   register_readers()      -> list[dict]
    #   register_writers()      -> list[dict]
    #   register_processors()   -> list[dict]
    #   register_visualizers()  -> list[dict]
    #   register_analyses()     -> list[dict]
    #   register_simulations()  -> list[dict]
    #   register_accessors()    -> list[dict]
    #   register_unit_contexts() -> list[dict]
    #   register_handlers()     -> dict[str, Callable]
    #
    # Each list-based hook dict must contain "name" and "func".
    # Optional keys: "description" (str), "extensions" (list[str]).
    # ------------------------------------------------------------------

    def register_readers(self) -> list[dict]:
        """Declare file readers provided by this plugin."""
        return [
            {
                "name": "myformat",
                "func": self._read_myformat,
                "description": "Read MyFormat files",
                "extensions": [".myf", ".myformat"],
            },
        ]

    def register_writers(self) -> list[dict]:
        """Declare file writers provided by this plugin."""
        return [
            {
                "name": "myformat",
                "func": self._write_myformat,
                "description": "Write MyFormat files",
            },
        ]

    def register_analyses(self) -> list[dict]:
        """Declare analysis workflows provided by this plugin."""
        return [
            {
                "name": "my_analysis",
                "func": self._perform_analysis,
                "description": "Example analysis workflow",
            },
        ]

    def register_accessors(self) -> list[dict]:
        """Declare dataset accessor methods provided by this plugin."""
        return [
            {
                # Function-based accessor (legacy but still supported)
                "namespace": "myplugin",
                "name": "analysis",
                "legacy_names": ["my_analysis"],
                "func": self._perform_analysis,
                "description": "Example dataset-bound analysis",
            },
            # Uncomment to register a class-based accessor:
            # {
            #     "namespace": "myplugin",
            #     "name": "accessor",
            #     "accessor_class": MyAccessor,  # class instantiated with dataset
            #     "description": "Example class-based accessor",
            # },
        ]

    # ------------------------------------------------------------------
    # Unit contexts (optional)
    # ------------------------------------------------------------------
    # Override to provide a custom Pint unit-context conversion.
    # Return an empty list when no contexts are provided.

    def register_unit_contexts(self) -> list[dict]:
        """Declare Pint unit-context contributions."""
        return []
        # Example entry:
        # {
        #     "name": "my_context",
        #     "func": self._setup_unit_context,
        #     "predicate": self._context_applies,
        #     "argument_extractor": self._get_context_args,
        #     "description": "Custom unit conversion context",
        # }

    # ------------------------------------------------------------------
    # Handler overrides (optional)
    # ------------------------------------------------------------------
    # Handlers let plugins override named behaviour in core classes.
    # Keys follow the convention "<domain>.<action>".
    # Return None to fall back to the core default.

    def register_handlers(self) -> dict[str, Callable]:
        """Return a dict of named handler overrides for core extension points."""
        return {
            # --- Importer hooks ---
            # "importer.infer_filetype_key": self._infer_filetype,
            #     Return a filetype key (e.g. ".myformat") for extensionless
            #     files owned by this plugin, or None.
            # "importer.resolve_directory_target": self._resolve_directory,
            #     Resolve a directory path into concrete files for this format,
            #     or None to use core directory globbing.
            # "importer.remote_download_target": self._remote_target,
            #     Return the parent directory to download for remote formats,
            #     or None to download the requested path directly.
            # --- Math hooks ---
            # "ndmath.execution_branch": self._execution_branch,
            #     Return a branch name ("real", "custom_numeric", …) for a
            #     math operation, or None to use the core default.
            # "ndmath.execute": self._execute_branch,
            #     Execute a math operation for a given branch, or None.
            # "ndmath.numpy_method.absolute": self._custom_absolute,
            #     Override the numpy absolute method for plugin-specific data.
            # --- Coordinate hooks ---
            # "coord.reversed": self._coord_reversed,
            #     Return True/False/None for coordinate axis reversal.
            # --- Concatenation hooks ---
            # "concatenate.extract_metadata": self._extract_metadata,
            # "concatenate.postprocess": self._concat_postprocess,
        }

    # ------------------------------------------------------------------
    # Operational methods (with deferred imports for optional deps)
    # ------------------------------------------------------------------

    def _read_myformat(self, path: str) -> NDDataset:
        """Read a MyFormat file and return an NDDataset."""
        import numpy as np

        from spectrochempy import NDDataset

        data = np.loadtxt(path)
        return NDDataset(data)

    def _write_myformat(self, dataset: NDDataset, path: str) -> None:
        """Write an NDDataset to a MyFormat file."""
        import numpy as np

        np.savetxt(path, dataset.data)

    def _perform_analysis(self, dataset: NDDataset) -> dict:
        """Run an example analysis workflow."""
        import numpy as np

        return {
            "mean": float(np.mean(dataset.data)),
            "std": float(np.std(dataset.data)),
        }

    # --- Optional handler implementations (uncomment to activate) ---

    # def _infer_filetype(self, path, **kwargs) -> str | None:
    #     """Return '.myformat' for extensionless plugin-owned files."""
    #     if path.name == "mydata":
    #         return ".myformat"
    #     return None

    # def _resolve_directory(self, path, **kwargs):
    #     """Resolve a directory to concrete plugin files."""
    #     import pathlib
    #     f = pathlib.Path(path) / "data.myf"
    #     return f if f.exists() else None

    # def _execution_branch(self, fname: str, data, args) -> str | None:
    #     """Return a custom numeric branch if data requires it."""
    #     # Example: return "custom_numeric" if data.dtype is plugin-specific
    #     return None

    # def _execute_branch(self, branch: str, f, d, args) -> np.ndarray | None:
    #     """Execute a math operation for a plugin-specific branch."""
    #     if branch == "custom_numeric":
    #         # Perform custom math here
    #         return f(d, *args)
    #     return None

    # def _custom_absolute(self, dataset, *args, **kwargs) -> NDDataset | None:
    #     """Override absolute for plugin-specific data types."""
    #     # Return a modified dataset or None to fall back to numpy.
    #     return None

    # --- Optional unit-context implementations ---

    # def _setup_unit_context(self, *args, **kwargs):
    #     """Create a Pint context (called once at registration)."""
    #     pass

    # def _context_applies(self, obj) -> bool:
    #     """Return True if the unit context should be used for *obj*."""
    #     return False

    # def _get_context_args(self, obj):
    #     """Extract arguments for the unit-context setup function."""
    #     return None


# ------------------------------------------------------------------
# Package-level namespace access (optional)
# ------------------------------------------------------------------
# Plugins can expose public functions at the package level so users
# can write ``scp.myplugin.foo()`` after installing the plugin.
#
# Uncomment to enable lazy module-level access:

# def __getattr__(name: str):
#     if name == "my_analysis":
#         from .my_analysis import perform_analysis  # noqa: PLC0415
#         return perform_analysis
#     msg = f"module {__name__!r} has no attribute {name!r}"
#     raise AttributeError(msg)
#
#
# def __dir__() -> list[str]:
#     return ["MyPlugin", "my_analysis"]
