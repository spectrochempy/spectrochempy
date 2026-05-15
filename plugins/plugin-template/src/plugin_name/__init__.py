"""
spectrochempy-myplugin — A SpectroChemPy plugin.

Replace this module docstring with your plugin's description.
"""

from spectrochempy.api.plugins import (
    CORE_PLUGIN_API_VERSION,
    PluginCapability,
    ReaderContribution,
    SpectroChemPyPlugin,
)


class MyPlugin(SpectroChemPyPlugin):
    """Minimal SpectroChemPy plugin example."""

    name = "myplugin"
    version = "0.1.0"
    description = "My SpectroChemPy plugin"
    spectrochempy_min_version = "0.8.0"
    PLUGIN_API_VERSION = CORE_PLUGIN_API_VERSION
    capabilities = [PluginCapability.READER]

    # ------------------------------------------------------------------
    # Declarative hooks — these are auto-collected by PluginManager.
    #
    # You can implement any combination of:
    #   register_readers()    -> list[dict]
    #   register_writers()    -> list[dict]
    #   register_processors() -> list[dict]
    #   register_visualizers()  -> list[dict]
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

    # ------------------------------------------------------------------
    # Operational methods (with deferred imports for optional deps)
    # ------------------------------------------------------------------

    def _read_myformat(self, path: str) -> "NDDataset":
        """Read a MyFormat file and return an NDDataset."""
        # Defer heavy imports to avoid slowing down SpectroChemPy startup
        import numpy as np

        from spectrochempy import NDDataset

        data = np.loadtxt(path)
        return NDDataset(data)

    def _write_myformat(self, dataset: "NDDataset", path: str) -> None:
        """Write an NDDataset to a MyFormat file."""
        import numpy as np

        np.savetxt(path, dataset.data)


# ------------------------------------------------------------------
# Optional: attach reader to NDDataset's method namespace
# ------------------------------------------------------------------

__dataset_methods__ = ["read_myformat"]  # exported for ndd.read_myformat()
