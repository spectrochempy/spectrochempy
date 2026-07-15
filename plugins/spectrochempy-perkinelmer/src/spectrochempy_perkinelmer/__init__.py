# ruff: noqa: PLC0415 — defer imports in plugin methods to avoid startup cost
"""PerkinElmer file reader plugin for SpectroChemPy."""

from __future__ import annotations

from spectrochempy.api.plugins import CORE_PLUGIN_API_VERSION
from spectrochempy.api.plugins import PluginCapability
from spectrochempy.api.plugins import SpectroChemPyPlugin
from spectrochempy.plugins.proxies import lazy_proxy


def _resolve_read_perkinelmer():
    """Lazily import and return the real ``read_perkinelmer`` function."""
    from .read_perkinelmer import read_perkinelmer

    return read_perkinelmer


def _ensure_filetype_registered() -> None:
    """Register the plugin-owned PerkinElmer key in the legacy importer registry."""
    from spectrochempy.core.readers.filetypes import registry

    known = {name for name, _description in registry.filetypes}
    if "perkinelmer" not in known:
        registry.register_filetype(
            "perkinelmer",
            "PerkinElmer SP files (*.sp)",
            aliases=["sp"],
        )


class PerkinElmerPlugin(SpectroChemPyPlugin):
    """PerkinElmer plugin providing the ``.sp`` file reader."""

    name = "perkinelmer"
    version = "0.1.3"
    description = "PerkinElmer file reader for SpectroChemPy"
    spectrochempy_min_version = "0.9.0"
    PLUGIN_API_VERSION = CORE_PLUGIN_API_VERSION
    capabilities = [PluginCapability.READER]

    def register_readers(self) -> list[dict]:
        """Declare the PerkinElmer ``.sp`` file reader."""
        _ensure_filetype_registered()
        func = lazy_proxy(
            _resolve_read_perkinelmer,
            name="spectrochempy.perkinelmer.read_perkinelmer",
        )
        return [
            {
                "name": "perkinelmer",
                "func": func,
                "description": "PerkinElmer SP binary format",
                "extensions": [".sp"],
            },
            {
                "name": "sp",
                "func": func,
                "description": "PerkinElmer SP binary format (alias)",
                "extensions": [".sp"],
            },
        ]


def __getattr__(name: str):
    if name in ("read_perkinelmer", "read_sp", "read"):
        from .read_perkinelmer import read_perkinelmer

        return read_perkinelmer
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    return ["PerkinElmerPlugin", "read", "read_perkinelmer", "read_sp"]
