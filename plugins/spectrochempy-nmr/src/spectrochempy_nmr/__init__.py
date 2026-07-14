# ruff: noqa: PLC0415 — defer imports in plugin methods to avoid startup cost
"""NMR readers and tools plugin for SpectroChemPy."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

from spectrochempy.api.plugins import CORE_PLUGIN_API_VERSION
from spectrochempy.api.plugins import PluginCapability
from spectrochempy.api.plugins import SpectroChemPyPlugin
from spectrochempy.plugins.proxies import lazy_proxy


def _resolve_read_topspin():
    """Lazily import and return the real ``read_topspin`` function."""
    from .readers.read_topspin import read_topspin  # noqa: PLC0415

    return read_topspin


def _resolve_read_agilent():
    """Lazily import and return the real ``read_agilent`` function."""
    from .readers.read_agilent import read_agilent  # noqa: PLC0415

    return read_agilent


def _resolve_read_jeol():
    """Lazily import and return the real ``read_jeol`` function."""
    from .readers.read_jeol import read_jeol  # noqa: PLC0415

    return read_jeol


def _resolve_read_tecmag():
    """Lazily import and return the real ``read_tecmag`` function."""
    from .readers.read_tecmag import read_tecmag  # noqa: PLC0415

    return read_tecmag


def _resolve_read_simpson():
    """Lazily import and return the real ``read_simpson`` function."""
    from .readers.read_simpson import read_simpson  # noqa: PLC0415

    return read_simpson


def _ensure_jeol_filetype_registered() -> None:
    """Register the plugin-owned JEOL key in the legacy importer registry."""
    from spectrochempy.core.readers.filetypes import registry  # noqa: PLC0415

    known = {name for name, _description in registry.filetypes}
    if "jeol" not in known:
        registry.register_filetype(
            "jeol",
            "JEOL JDF NMR data files",
            aliases=["jdf"],
        )


def _ensure_tecmag_filetype_registered() -> None:
    """Register the plugin-owned TecMag key in the legacy importer registry."""
    from spectrochempy.core.readers.filetypes import registry  # noqa: PLC0415

    known = {name for name, _description in registry.filetypes}
    if "tecmag" not in known:
        registry.register_filetype(
            "tecmag",
            "TecMag TNT NMR data files",
            aliases=["tnt"],
        )


def _ensure_simpson_filetype_registered() -> None:
    """Register the plugin-owned SIMPSON key in the legacy importer registry."""
    from spectrochempy.core.readers.filetypes import registry  # noqa: PLC0415

    known = {name for name, _description in registry.filetypes}
    if "simpson" not in known:
        registry.register_filetype(
            "simpson",
            "SIMPSON NMR simulation data files",
            aliases=["spe", "fid", "in"],
        )


def _resolve_read():
    """Lazily import and return a generic NMR reader dispatching by file type."""
    from .readers.read_agilent import read_agilent  # noqa: PLC0415
    from .readers.read_jeol import read_jeol  # noqa: PLC0415
    from .readers.read_simpson import read_simpson  # noqa: PLC0415
    from .readers.read_tecmag import read_tecmag  # noqa: PLC0415
    from .readers.read_topspin import read_topspin  # noqa: PLC0415

    def read(*paths, **kwargs):
        """Read NMR data, auto-detecting TopSpin vs Agilent/Varian vs JEOL vs TecMag vs SIMPSON format."""
        protocol = kwargs.get("protocol")
        if protocol == "agilent":
            return read_agilent(*paths, **kwargs)
        if protocol == "topspin":
            return read_topspin(*paths, **kwargs)
        if protocol == "jeol":
            return read_jeol(*paths, **kwargs)
        if protocol == "tecmag":
            return read_tecmag(*paths, **kwargs)
        if protocol == "simpson":
            return read_simpson(*paths, **kwargs)

        # Auto-detect from the first path when no protocol is given.
        if paths:
            first = paths[0]
            try:
                path = Path(first)
                # SIMPSON files use .spe/.fid/.in extensions
                if path.suffix.lower() in {".spe", ".fid", ".in"}:
                    return read_simpson(*paths, **kwargs)
                # TecMag files have .tnt extension
                if path.suffix.lower() == ".tnt":
                    return read_tecmag(*paths, **kwargs)
                # JEOL files have .jdf extension
                if path.suffix.lower() == ".jdf":
                    return read_jeol(*paths, **kwargs)
                if path.is_dir():
                    # An Agilent directory contains fid + procpar.
                    if (path / "fid").exists() and (path / "procpar").exists():
                        return read_agilent(*paths, **kwargs)
                    return read_topspin(*paths, **kwargs)
                if (
                    path.name == "fid"
                    and path.parent.exists()
                    and (path.parent / "procpar").exists()
                ):
                    return read_agilent(*paths, **kwargs)
            except (TypeError, ValueError):
                pass

        # Default to TopSpin for backward compatibility.
        return read_topspin(*paths, **kwargs)

    return read


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


_VALID_AGILENT_FILENAMES = {"fid"}


_VALID_JEOL_EXTENSIONS = {".jdf"}


def _is_jeol_protocol(**kwargs) -> bool:
    protocol = kwargs.get("protocol")
    if protocol is None:
        return False
    if isinstance(protocol, str):
        protocol = [protocol]
    return "jeol" in protocol


def _infer_jeol_filetype_key(filename, **kwargs):
    """Return the JEOL filetype key for .jdf files."""
    if filename.suffix.lower() == ".jdf":
        return ".jeol"
    return None


_VALID_TECMAG_EXTENSIONS = {".tnt"}


def _is_tecmag_protocol(**kwargs) -> bool:
    protocol = kwargs.get("protocol")
    if protocol is None:
        return False
    if isinstance(protocol, str):
        protocol = [protocol]
    return "tecmag" in protocol


def _infer_tecmag_filetype_key(filename, **kwargs):
    """Return the TecMag filetype key for .tnt files."""
    if filename.suffix.lower() == ".tnt":
        return ".tecmag"
    return None


_VALID_SIMPSON_EXTENSIONS = {".spe", ".fid", ".in"}


def _is_simpson_protocol(**kwargs) -> bool:
    protocol = kwargs.get("protocol")
    if protocol is None:
        return False
    if isinstance(protocol, str):
        protocol = [protocol]
    return "simpson" in protocol


def _infer_simpson_filetype_key(filename, **kwargs):
    """Return the SIMPSON filetype key for .spe/.fid/.in files."""
    if filename.suffix.lower() in _VALID_SIMPSON_EXTENSIONS:
        return ".simpson"
    return None


def _is_agilent_protocol(**kwargs) -> bool:
    protocol = kwargs.get("protocol")
    if protocol is None:
        return False
    if isinstance(protocol, str):
        protocol = [protocol]
    return "agilent" in protocol


def _resolve_agilent_directory_target(filename, **kwargs):
    """Resolve an Agilent experiment directory to the fid file."""
    if not _is_agilent_protocol(**kwargs):
        return None
    if filename.name in _VALID_AGILENT_FILENAMES:
        return None

    fid_path = filename / "fid"
    procpar_path = filename / "procpar"
    if fid_path.exists() and procpar_path.exists():
        return [fid_path]
    return None


def _infer_agilent_filetype_key(filename, **kwargs):
    """Return the Agilent filetype key for fid/procpar file pairs."""
    if filename.name == "fid" and filename.parent.exists():
        procpar = filename.parent / "procpar"
        if procpar.exists():
            return ".agilent"
    return None


def _ensure_agilent_filetype_registered() -> None:
    """Register the plugin-owned Agilent key in the legacy importer registry."""
    from spectrochempy.core.readers.filetypes import registry  # noqa: PLC0415

    known = {name for name, _description in registry.filetypes}
    if "agilent" not in known:
        registry.register_filetype(
            "agilent",
            "Agilent/Varian NMR fid and procpar files",
        )


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


def _infer_nmr_filetype_key(filename, **kwargs):
    """Return the filetype key for NMR data files."""
    # Check SIMPSON first: .spe/.fid/.in files are SIMPSON.
    simpson_key = _infer_simpson_filetype_key(filename, **kwargs)
    if simpson_key is not None:
        return simpson_key
    # Check TecMag: .tnt files are TecMag.
    tecmag_key = _infer_tecmag_filetype_key(filename, **kwargs)
    if tecmag_key is not None:
        return tecmag_key
    # Check JEOL: .jdf files are JEOL.
    jeol_key = _infer_jeol_filetype_key(filename, **kwargs)
    if jeol_key is not None:
        return jeol_key
    # Check Agilent first: a fid with a sibling procpar is Agilent, not TopSpin.
    return _infer_agilent_filetype_key(
        filename, **kwargs
    ) or _infer_topspin_filetype_key(filename, **kwargs)


def _resolve_nmr_directory_target(filename, **kwargs):
    """Resolve an NMR experiment directory to concrete data files."""
    return _resolve_topspin_directory_target(
        filename, **kwargs
    ) or _resolve_agilent_directory_target(filename, **kwargs)


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
    """NMR plugin, providing Bruker TopSpin, Agilent/Varian, JEOL, TecMag, and SIMPSON readers."""

    name = "nmr"
    version = "0.11.0"
    description = "NMR readers and tools for SpectroChemPy"
    spectrochempy_min_version = "0.9.0"
    PLUGIN_API_VERSION = CORE_PLUGIN_API_VERSION
    capabilities = [PluginCapability.READER]
    io_namespaces = {
        "topspin": {"read": "nmr.read_topspin"},
        "agilent": {"read": "nmr.read_agilent"},
        "jeol": {"read": "nmr.read_jeol"},
        "tecmag": {"read": "nmr.read_tecmag"},
        "simpson": {"read": "nmr.read_simpson"},
    }

    def register_readers(self) -> list[dict]:
        """Declare the TopSpin, Agilent, JEOL, TecMag, and SIMPSON file readers."""
        _ensure_topspin_filetype_registered()
        _ensure_agilent_filetype_registered()
        _ensure_jeol_filetype_registered()
        _ensure_tecmag_filetype_registered()
        _ensure_simpson_filetype_registered()
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
            {
                "name": "agilent",
                "func": lazy_proxy(
                    _resolve_read_agilent, name="spectrochempy.nmr.read_agilent"
                ),
                "description": "Agilent/Varian NMR fid and procpar files",
                "extensions": ["fid"],
            },
            {
                "name": "jeol",
                "func": lazy_proxy(
                    _resolve_read_jeol, name="spectrochempy.nmr.read_jeol"
                ),
                "description": "JEOL JDF NMR data files",
                "extensions": [".jdf"],
            },
            {
                "name": "tecmag",
                "func": lazy_proxy(
                    _resolve_read_tecmag, name="spectrochempy.nmr.read_tecmag"
                ),
                "description": "TecMag TNT NMR data files",
                "extensions": [".tnt"],
            },
            {
                "name": "simpson",
                "func": lazy_proxy(
                    _resolve_read_simpson, name="spectrochempy.nmr.read_simpson"
                ),
                "description": "SIMPSON NMR simulation data files",
                "extensions": [".spe", ".fid", ".in"],
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
        from .processing.fft_encodings import _fft_encoding_handler  # noqa: PLC0415
        from .processing.fft_postprocess import _fft_postprocess_result  # noqa: PLC0415

        return {
            "coord.reversed": _nmr_coord_reversed,
            "concatenate.extract_metadata": _nmr_concat_extract_metadata,
            "concatenate.postprocess": _nmr_concat_postprocess,
            "fft.encoding": _fft_encoding_handler,
            "fft.postprocess_result": _fft_postprocess_result,
            "importer.infer_filetype_key": _infer_nmr_filetype_key,
            "importer.remote_download_target": _topspin_remote_download_target,
            "importer.resolve_directory_target": _resolve_nmr_directory_target,
        }


# ------------------------------------------------------------------
# Lazy module-level access for public API
# ------------------------------------------------------------------


def __getattr__(name: str):
    if name == "read_topspin":
        from .readers.read_topspin import read_topspin  # noqa: PLC0415

        return read_topspin
    if name == "read_agilent":
        from .readers.read_agilent import read_agilent  # noqa: PLC0415

        return read_agilent
    if name == "read_jeol":
        from .readers.read_jeol import read_jeol  # noqa: PLC0415

        return read_jeol
    if name == "read_tecmag":
        from .readers.read_tecmag import read_tecmag  # noqa: PLC0415

        return read_tecmag
    if name == "read_simpson":
        from .readers.read_simpson import read_simpson  # noqa: PLC0415

        return read_simpson
    if name == "read":
        return _resolve_read()
    if name == "set_nmr_context":
        from .units import set_nmr_context  # noqa: PLC0415

        return set_nmr_context
    if name == "Experiment":
        from .experiment import Experiment  # noqa: PLC0415

        return Experiment
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    return [
        "NMRPlugin",
        "Experiment",
        "read",
        "read_topspin",
        "read_agilent",
        "read_jeol",
        "read_tecmag",
        "read_simpson",
        "set_nmr_context",
    ]
