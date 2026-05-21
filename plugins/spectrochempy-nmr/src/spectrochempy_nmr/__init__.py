# ruff: noqa: PLC0415 — defer imports in plugin methods to avoid startup cost
"""NMR readers and tools plugin for SpectroChemPy."""

from __future__ import annotations

from spectrochempy.api.plugins import CORE_PLUGIN_API_VERSION
from spectrochempy.api.plugins import PluginCapability
from spectrochempy.api.plugins import SpectroChemPyPlugin
from spectrochempy.plugins.proxies import lazy_proxy


def _resolve_read_topspin():
    """Lazily import and return the real ``read_topspin`` function."""
    from .read_topspin import read_topspin  # noqa: PLC0415

    return read_topspin


def _resolve_set_nmr_context():
    """Lazily import and return the NMR unit-context setup function."""
    from .units import set_nmr_context  # noqa: PLC0415

    return set_nmr_context


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
        return [
            {
                "name": "topspin",
                "func": lazy_proxy(
                    _resolve_read_topspin, name="spectrochempy.nmr.read_topspin"
                ),
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

    def register_unit_contexts(self) -> list[dict]:
        """Declare the NMR ppm/frequency Pint context."""
        return [
            {
                "name": "nmr",
                "func": lazy_proxy(
                    _resolve_set_nmr_context,
                    name="spectrochempy.nmr.set_nmr_context",
                ),
                "description": "NMR ppm/frequency conversion context",
            },
        ]


# ------------------------------------------------------------------
# Lazy module-level access for public API
# ------------------------------------------------------------------


def __getattr__(name: str):
    if name == "read_topspin":
        from .read_topspin import read_topspin  # noqa: PLC0415

        return read_topspin
    if name == "set_nmr_context":
        from .units import set_nmr_context  # noqa: PLC0415

        return set_nmr_context
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    return ["NMRPlugin", "read_topspin", "set_nmr_context"]
