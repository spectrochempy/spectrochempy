# ruff: noqa: PLC0415 — defer imports in plugin methods to avoid startup cost
"""NMR readers and tools plugin for SpectroChemPy."""

from __future__ import annotations

from collections.abc import Callable

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


def _coord_has_larmor(obj) -> bool:
    """Return whether an object should use the NMR unit context."""
    implements = getattr(obj, "_implements", None)
    return (
        callable(implements)
        and implements("Coord")
        and bool(getattr(obj, "larmor", None))
    )


def _coord_larmor_argument(obj):
    """Extract the Larmor frequency for the NMR unit-context setup function."""
    return obj.larmor


def _nmr_coord_reversed(coord) -> bool | None:
    """Return True if the Coord has ppm units (NMR axis reversal)."""
    if coord.units == "ppm":
        return True
    return None


def _nmr_concat_extract_metadata(datasets):
    """Extract metadata coordinates for variable-parameter TopSpin datasets."""
    import numpy as np  # noqa: PLC0415

    if datasets[0].origin != "topspin":
        return None
    metacoords: dict[str, list] = {}
    meta0 = datasets[0].meta
    for i, dataset in enumerate(datasets):
        if i == 0:
            continue
        meta = dataset.meta
        for key in meta0:
            if key in ["file_size", "pprog", "phc0", "phc1", "nsold"]:
                continue
            keepitem = key if key != "date" else "timestamp"
            if np.any(meta0[key][-1] != meta[key][-1]):
                if hasattr(meta0[key][-1], "size") and meta0[key][-1].size > 1:
                    for i in range(meta0[key][-1].size):
                        if np.any(meta0[key][-1][i] == meta[key][-1][i]):
                            continue
                        itemi = f"{key}{i}"
                        if itemi not in metacoords:
                            metacoords[itemi] = [
                                meta0[key][-1][i],
                                meta[key][-1][i],
                            ]
                        else:
                            metacoords[itemi].append(meta[key][-1][i])
                    continue
                if keepitem not in metacoords:
                    metacoords[keepitem] = [meta0[key][-1], meta[key][-1]]
                else:
                    metacoords[keepitem].append(meta[key][-1])
    return metacoords


def _nmr_concat_postprocess(out, datasets, **kwargs):
    """Post-process concatenation for NMR TopSpin datasets."""
    from spectrochempy.core.dataset.coord import Coord  # noqa: PLC0415
    from spectrochempy.core.dataset.coordset import CoordSet  # noqa: PLC0415

    metacoords = kwargs.get("metacoords", {})
    if metacoords:
        coords = []
        for key, value in metacoords.items():
            coords.append(Coord(value, title=key))
        out.y = CoordSet(coords)
    return out


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
                "predicate": _coord_has_larmor,
                "argument_extractor": _coord_larmor_argument,
                "description": "NMR ppm/frequency conversion context",
            },
        ]

    def register_handlers(self) -> dict[str, Callable]:
        """Register handler overrides for core extension points."""
        return {
            "coord.reversed": _nmr_coord_reversed,
            "concatenate.extract_metadata": _nmr_concat_extract_metadata,
            "concatenate.postprocess": _nmr_concat_postprocess,
        }


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
