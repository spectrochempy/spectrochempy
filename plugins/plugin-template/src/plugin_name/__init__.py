# ruff: noqa: PLC0415 — defer imports in plugin methods to avoid startup cost
"""
spectrochempy-myplugin — A SpectroChemPy plugin.

Replace this module docstring with your plugin's description.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from spectrochempy.api.plugins import CORE_PLUGIN_API_VERSION

if TYPE_CHECKING:
    from spectrochempy import NDDataset
from spectrochempy.api.plugins import PluginCapability
from spectrochempy.api.plugins import SpectroChemPyPlugin


class MyPlugin(SpectroChemPyPlugin):
    """Minimal SpectroChemPy plugin example."""

    name = "myplugin"
    version = "0.1.0"
    description = "My SpectroChemPy plugin"
    spectrochempy_min_version = "0.8.0"
    PLUGIN_API_VERSION = CORE_PLUGIN_API_VERSION
    capabilities = [PluginCapability.READER, PluginCapability.ANALYSIS]
    requires: list[str] = []
    """
    Optional pip-style dependencies.
    Plugins with missing deps are marked FAILED with a clear message.
    Example: requires = ["cantera>=3.0", "pint"]
    """

    # ------------------------------------------------------------------
    # Declarative hooks — these are auto-collected by PluginManager.
    #
    # Available hooks:
    #   register_readers()      -> list[dict]
    #   register_writers()      -> list[dict]
    #   register_processors()   -> list[dict]
    #   register_visualizers()  -> list[dict]
    #   register_analyses()     -> list[dict]  (analysis workflows)
    #   register_simulations()  -> list[dict]  (simulation engines)
    #   register_accessors()    -> list[dict]  (dataset methods)
    #
    # Each dict must contain "name" and "func".
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

    # ------------------------------------------------------------------
    # Operational methods (with deferred imports for optional deps)
    # ------------------------------------------------------------------

    def _read_myformat(self, path: str) -> NDDataset:
        """Read a MyFormat file and return an NDDataset."""
        # Defer heavy imports to avoid slowing down SpectroChemPy startup
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


# ------------------------------------------------------------------
# Optional: attach methods to NDDataset
# Methods listed here are discoverable via ndd.method_name()
# ------------------------------------------------------------------

__dataset_methods__ = ["read_myformat"]
