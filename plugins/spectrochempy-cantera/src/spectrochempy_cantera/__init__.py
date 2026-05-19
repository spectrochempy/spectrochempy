# ruff: noqa: PLC0415 — defer imports in plugin methods to avoid startup cost
"""
Cantera PFR plugin for SpectroChemPy.

Currently provides:
- ``PFR``: Plug-flow reactor model using Cantera.

PFR is lazily imported so that plugin discovery stays lightweight.
"""

from __future__ import annotations

from spectrochempy.api.plugins import CORE_PLUGIN_API_VERSION
from spectrochempy.api.plugins import PluginCapability
from spectrochempy.api.plugins import SpectroChemPyPlugin


def _lazy_pfr(*args, **kwargs):
    """Lazy wrapper — imports ``_pfr.PFR`` only on call."""
    from ._pfr import PFR  # noqa: PLC0415

    return PFR(*args, **kwargs)


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
    root_exports = {
        "PFR": {
            "target": "PFR",
            "deprecated": True,
            "replacement": "scp.cantera.PFR",
        },
    }

    # ------------------------------------------------------------------
    # Declarative hooks
    # ------------------------------------------------------------------

    def register_simulations(self) -> list[dict]:
        return [
            {
                "name": "PFR",
                "func": _lazy_pfr,
                "description": "Plug-flow reactor (CSTR-in-series) model",
            },
        ]

    def register_analyses(self) -> list[dict]:
        return []

    def register_readers(self) -> list[dict]:
        return []


# ------------------------------------------------------------------
# Lazy module-level access
# ------------------------------------------------------------------


def __getattr__(name: str):
    if name == "PFR":
        from ._pfr import PFR  # noqa: PLC0415

        return PFR
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    return ["CanteraPlugin", "PFR", "_lazy_pfr"]
