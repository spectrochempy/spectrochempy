# ruff: noqa: PLC0415 — defer imports in plugin methods to avoid startup cost
"""Carroucell experiment reader plugin for SpectroChemPy."""

from __future__ import annotations

from spectrochempy.api.plugins import CORE_PLUGIN_API_VERSION
from spectrochempy.api.plugins import PluginCapability
from spectrochempy.api.plugins import SpectroChemPyPlugin
from spectrochempy.plugins.proxies import lazy_proxy


def _resolve_read_carroucell():
    """Lazily import and return the real ``read_carroucell`` function."""
    from .read_carroucell import read_carroucell  # noqa: PLC0415

    return read_carroucell


def _infer_carroucell_filetype_key(filename, **kwargs):
    """Return the Carroucell filetype key when protocol is carroucell."""
    protocol = kwargs.get("protocol")
    if protocol is None:
        return None
    if isinstance(protocol, str):
        protocol = [protocol]
    if "carroucell" in protocol:
        return ".carroucell"
    return None


def _ensure_carroucell_filetype_registered():
    """Register the carroucell filetype with the legacy FileTypeRegistry."""
    from spectrochempy.core.readers.filetypes import registry  # noqa: PLC0415

    known = {name for name, _description in registry.filetypes}
    if "carroucell" not in known:
        registry.register_filetype(
            "carroucell",
            "Carroucell experiment data (*.spa)",
        )


class CarroucellPlugin(SpectroChemPyPlugin):
    """Carroucell plugin, providing the Carroucell experiment reader."""

    name = "carroucell"
    version = "0.1.4"
    description = "Carroucell experiment reader for SpectroChemPy"
    spectrochempy_min_version = "0.9.0"
    PLUGIN_API_VERSION = CORE_PLUGIN_API_VERSION
    capabilities = [PluginCapability.READER]

    def register_readers(self) -> list[dict]:
        """Declare the Carroucell file reader."""
        _ensure_carroucell_filetype_registered()
        return [
            {
                "name": "carroucell",
                "func": lazy_proxy(
                    _resolve_read_carroucell,
                    name="spectrochempy.carroucell.read_carroucell",
                ),
                "description": "Carroucell experiment data (*.spa)",
                "extensions": [".carroucell"],
            },
        ]

    def register_handlers(self) -> dict:
        """Register handler overrides for core extension points."""
        return {
            "importer.infer_filetype_key": _infer_carroucell_filetype_key,
        }


def __getattr__(name: str):
    if name == "read_carroucell":
        from .read_carroucell import read_carroucell  # noqa: PLC0415

        return read_carroucell
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    return ["CarroucellPlugin", "read_carroucell"]
