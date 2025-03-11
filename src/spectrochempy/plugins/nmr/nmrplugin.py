# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
#
from spectrochempy.core.readers.filetypes import registry
from spectrochempy.plugins.pluginmanager import SpectroChemPyPlugin


class NMRPlugin(SpectroChemPyPlugin):
    """NMR plugin for SpectroChemPy."""

    def __init__(self):
        self._is_initialized = False

        # Init here all methods that should be discovered by spectrochempy
        self.read_topspin = None

    @property
    def name(self) -> str:
        return "nmr"

    @property
    def require_dependencies(self) -> dict[str:str]:
        """Dict of required package dependencies with eventual renaming."""
        return {}

    @property
    def require_plugins(self) -> list[str]:
        """List of required plugins."""
        return ["quaternion"]

    def initialize(self, preferences) -> None:
        """Initialize NMR plugin."""
        if self._is_initialized:
            return

        # Checks dependencies
        super().initialize(preferences)

        # Register NMR files types
        registry.register_filetype(
            "topspin",
            "Bruker TOPSPIN files (fid ser 1[r|i] 2[r|i]* 3[r|i]*)",
        )

        # Register NMR-specific functionality
        from .readers.read_topspin import read_topspin  # noqa: F401

        self.read_topspin = read_topspin
        # self.processors = processors
        # self.plotters = plotters

        # Mark initialization as complete
        self._is_initialized = True


# Export plugin class
plugin_class = NMRPlugin

# ======================================================================================
if __name__ == "__main__":
    import pathlib

    import spectrochempy as scp

    # Access NMR plugin functionalit
    path = pathlib.Path("nmrdata") / "bruker" / "tests" / "nmr" / "topspin_2d"
    ndd = scp.read_topspin(path, expno=1, remove_digital_filter=True)
    print(ndd)  # noqa: T201
