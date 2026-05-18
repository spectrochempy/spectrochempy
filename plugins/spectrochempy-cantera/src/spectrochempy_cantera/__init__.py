# ruff: noqa: PLC0415 — defer imports in plugin methods to avoid startup cost
"""
Cantera PFR plugin for SpectroChemPy.

Currently provides:
- ``PFR``: Plug-flow reactor model using Cantera.
"""

from __future__ import annotations

from spectrochempy.api.plugins import CORE_PLUGIN_API_VERSION
from spectrochempy.api.plugins import PluginCapability
from spectrochempy.api.plugins import SpectroChemPyPlugin


class CanteraPlugin(SpectroChemPyPlugin):
    """Cantera plugin for SpectroChemPy."""

    name = "cantera"
    version = "0.1.0"
    description = "Plug-flow reactor (PFR) simulation via Cantera"
    spectrochempy_min_version = "0.8.0"
    PLUGIN_API_VERSION = CORE_PLUGIN_API_VERSION
    capabilities = [
        PluginCapability.SIMULATION,
    ]
    requires = ["cantera"]

    # ------------------------------------------------------------------
    # Declarative hooks
    # ------------------------------------------------------------------

    def register_simulations(self) -> list[dict]:
        return [
            {
                "name": "PFR",
                "func": PFR,
                "description": "Plug-flow reactor (CSTR-in-series) model",
            },
        ]

    def register_analyses(self) -> list[dict]:
        return []

    def register_readers(self) -> list[dict]:
        return []


# Re-export PFR from the _pfr module
from ._pfr import PFR  # noqa: E402,F401
