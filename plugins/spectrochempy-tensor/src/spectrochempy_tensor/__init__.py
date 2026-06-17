# ruff: noqa: PLC0415 - defer imports in plugin methods to avoid startup cost
"""Tensor decomposition plugin for SpectroChemPy."""

from __future__ import annotations

from spectrochempy.api.plugins import CORE_PLUGIN_API_VERSION
from spectrochempy.api.plugins import PluginCapability
from spectrochempy.api.plugins import SpectroChemPyPlugin
from spectrochempy.plugins.proxies import lazy_proxy


def _resolve_CP():
    """Lazily import and return the CP decomposition class."""
    from spectrochempy_tensor.decompositions.cp import CP

    return CP


class TensorPlugin(SpectroChemPyPlugin):
    """TensorLy-backed tensor decomposition plugin."""

    name = "tensor"
    version = "0.1.1"
    description = "TensorLy-backed tensor decompositions for SpectroChemPy"
    spectrochempy_min_version = "0.9.0"
    PLUGIN_API_VERSION = CORE_PLUGIN_API_VERSION
    requires = ["tensorly"]
    capabilities = [PluginCapability.ANALYSIS]
    root_exports = {
        "CP": {
            "target": "CP",
            "deprecated": True,
            "replacement": "scp.tensor.CP",
        },
    }

    def register_analyses(self) -> list[dict]:
        """Declare tensor decomposition classes."""
        return [
            {
                "name": "CP",
                "func": lazy_proxy(
                    _resolve_CP,
                    name="spectrochempy.tensor.CP",
                ),
                "namespace": "tensor",
                "description": "CP/PARAFAC tensor decomposition",
            },
        ]


def __getattr__(name: str):
    if name == "CP":
        from spectrochempy_tensor.decompositions.cp import CP

        return CP
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    return ["CP", "TensorPlugin"]
