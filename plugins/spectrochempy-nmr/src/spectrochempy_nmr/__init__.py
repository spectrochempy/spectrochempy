# ruff: noqa: PLC0415 - defer imports in plugin methods to avoid startup cost
"""NMR readers and tools plugin for SpectroChemPy."""

from __future__ import annotations

from typing import TYPE_CHECKING

from spectrochempy.api.plugins import CORE_PLUGIN_API_VERSION
from spectrochempy.api.plugins import PluginCapability
from spectrochempy.api.plugins import SpectroChemPyPlugin

if TYPE_CHECKING:
    pass


class NMRPlugin(SpectroChemPyPlugin):
    """NMR plugin, currently providing the Bruker TopSpin reader."""

    name = "nmr"
    version = "0.1.0"
    description = "NMR readers and tools for SpectroChemPy"
    spectrochempy_min_version = "0.8.0"
    PLUGIN_API_VERSION = CORE_PLUGIN_API_VERSION
    capabilities = [PluginCapability.READER]

    def register_readers(self) -> list[dict]:
        """Declare the TopSpin file reader."""
        # Deferred import: read_topspin pulls in numpy-quaternion
        from .read_topspin import read_topspin

        return [
            {
                "name": "topspin",
                "func": read_topspin,
                "description": "Bruker TOPSPIN fid, series, or processed data",
                "extensions": [
                    ".fid",
                    ".ser",
                    "1r",
                    "1i",
                    "2rr",
                    "2ri",
                    "3rrr",
                    "3rri",
                ],
            },
        ]


# Export reader name so NDDataset can discover it via ndd.read_topspin()
__dataset_methods__ = ["read_topspin"]
