# ruff: noqa: PLC0415 — defer imports in plugin methods to avoid startup cost
"""NMR readers and tools plugin for SpectroChemPy."""

from __future__ import annotations

import re
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
        and bool(obj.meta.get("acquisition_frequency") or obj.meta.get("larmor"))
    )


def _coord_larmor_argument(obj):
    """Extract the Larmor frequency for the NMR unit-context setup function."""
    return obj.meta.get("acquisition_frequency") or obj.meta.get("larmor")


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
                    for idx in range(meta0[key][-1].size):
                        if np.any(meta0[key][-1][idx] == meta[key][-1][idx]):
                            continue
                        itemi = f"{key}{idx}"
                        if itemi not in metacoords:
                            metacoords[itemi] = [
                                meta0[key][-1][idx],
                                meta[key][-1][idx],
                            ]
                        else:
                            metacoords[itemi].append(meta[key][-1][idx])
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


_VALID_TOPSPIN_FILENAMES = {"fid", "ser", "1r", "2rr", "3rrr"}


def _is_topspin_protocol(**kwargs) -> bool:
    protocol = kwargs.get("protocol")
    if protocol is None:
        return False
    if isinstance(protocol, str):
        protocol = [protocol]
    return "topspin" in protocol


def _resolve_topspin_directory_target(filename, **kwargs):
    """Resolve a TopSpin experiment directory to concrete data files."""
    if not _is_topspin_protocol(**kwargs):
        return None
    if filename.name in _VALID_TOPSPIN_FILENAMES:
        return None

    if kwargs.get("iterdir", False) or kwargs.get("glob") is not None:
        glob = kwargs.get("glob")
        if glob:
            files_ = list(filename.glob(glob))
        elif not kwargs.get("processed", False):
            files_ = list(filename.glob("**/ser"))
            files_.extend(filename.glob("**/fid"))
        else:
            files_ = list(filename.glob("**/1r"))
            files_.extend(filename.glob("**/2rr"))
            files_.extend(filename.glob("**/3rrr"))
    else:
        expno = kwargs.get("expno")
        procno = kwargs.get("procno")

        if expno is None:
            expnos = sorted(filename.glob("[0-9]*"))
            # A missing local directory can still be fetched remotely. TopSpin
            # experiment directories conventionally begin at experiment 1.
            expno = expnos[0] if expnos else 1

        if procno is None:
            f = filename / str(expno)
            files_ = [f / "ser"] if (f / "ser").exists() else [f / "fid"]
        else:
            f = filename / str(expno) / "pdata" / str(procno)
            if (f / "3rrr").exists():
                files_ = [f / "3rrr"]
            elif (f / "2rr").exists():
                files_ = [f / "2rr"]
            else:
                files_ = [f / "1r"]

    return [item for item in files_ if item.name in _VALID_TOPSPIN_FILENAMES]


def _infer_topspin_filetype_key(filename, **kwargs):
    """Return the TopSpin filetype key for extensionless Bruker data files."""
    if filename.name in _VALID_TOPSPIN_FILENAMES:
        return ".topspin"
    return None


def _topspin_remote_download_target(path, **kwargs):
    """Download the TopSpin experiment directory for component data files."""
    if not _is_topspin_protocol(**kwargs):
        return None
    match = re.match(r"(.*)(/pdata/\d+/\d+[ri]{1,3}|ser|fid)", str(path))
    if match is None:
        return None
    return match[1]


def _ensure_topspin_filetype_registered() -> None:
    """Register the plugin-owned TopSpin key in the legacy importer registry."""
    from spectrochempy.core.readers.filetypes import registry  # noqa: PLC0415

    known = {name for name, _description in registry.filetypes}
    if "topspin" not in known:
        registry.register_filetype(
            "topspin",
            "Bruker TOPSPIN fid, series, or processed data files",
        )


class NMRPlugin(SpectroChemPyPlugin):
    """NMR plugin, currently providing the Bruker TopSpin reader."""

    name = "nmr"
    version = "0.1.3"
    description = "NMR readers and tools for SpectroChemPy"
    spectrochempy_min_version = "0.9.0"
    PLUGIN_API_VERSION = CORE_PLUGIN_API_VERSION
    capabilities = [PluginCapability.READER]

    def register_readers(self) -> list[dict]:
        """Declare the TopSpin file reader."""
        _ensure_topspin_filetype_registered()
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
        from .fft_encodings import _fft_encoding_handler  # noqa: PLC0415
        from .fft_postprocess import _fft_postprocess_result  # noqa: PLC0415

        return {
            "coord.reversed": _nmr_coord_reversed,
            "concatenate.extract_metadata": _nmr_concat_extract_metadata,
            "concatenate.postprocess": _nmr_concat_postprocess,
            "fft.encoding": _fft_encoding_handler,
            "fft.postprocess_result": _fft_postprocess_result,
            "importer.infer_filetype_key": _infer_topspin_filetype_key,
            "importer.remote_download_target": _topspin_remote_download_target,
            "importer.resolve_directory_target": _resolve_topspin_directory_target,
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
