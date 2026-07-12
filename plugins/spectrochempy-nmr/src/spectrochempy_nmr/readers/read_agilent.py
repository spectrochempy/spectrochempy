# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Agilent/Varian NMR file importers.

This module provides functionality to read Agilent/Varian NMR data files
(FID binary + procpar parameter files).

Functions
---------
- read_agilent : Main entry point for reading Agilent/Varian files

Notes
-----
Supports 1D FID and multidimensional data.  Only reading is implemented;
no write or low-memory variants are provided.

"""

__all__ = ["read_agilent"]

from datetime import datetime

import numpy as np

from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.readers.importer import Importer
from spectrochempy.core.readers.importer import _importer_method
from spectrochempy.core.units import ur
from spectrochempy.utils._logging import warning_
from spectrochempy.utils.meta import Meta

from spectrochempy_nmr.extern.nmrglue import read_varian

# ======================================================================================
# Valid Agilent parameter names for metadata extraction
# ======================================================================================

# procpar parameter names → (unit_string, is_direct)
# Direct dimension parameters: sw, sfrq/dfrq, np, at, dn/tn
# Indirect dimension parameters: ni, ni2, ni3, sw2/sfrq2, etc.
agilent_valid_meta = [
    ("sw", "Hz"),
    ("sfrq", "MHz"),
    ("dfrq", "MHz"),
    ("dn", ""),
    ("tn", ""),
    ("np", ""),
    ("ni", ""),
    ("ni2", ""),
    ("ni3", ""),
    ("nt", ""),
    ("ne", ""),
    ("ns", ""),
    ("sw2", "Hz"),
    ("sfrq2", "MHz"),
    ("dfrq2", "MHz"),
    ("dn2", ""),
    ("tn2", ""),
    ("sw3", "Hz"),
    ("sfrq3", "MHz"),
    ("dfrq3", "MHz"),
    ("dn3", ""),
    ("tn3", ""),
    ("sw4", "Hz"),
    ("sfrq4", "MHz"),
    ("dfrq4", "MHz"),
    ("dn4", ""),
    ("tn4", ""),
    ("at", "s"),
    ("rt", "s"),
    ("t1", "s"),
    ("d1", "s"),
    ("pw", "us"),
    ("p1", "us"),
    ("ro", ""),
    ("tp", ""),
    ("phase", ""),
    ("phase2", ""),
    ("phase3", ""),
    ("array", ""),
    ("pulprog", ""),
    ("seqcon", ""),
    ("filnam", ""),
    ("date", ""),
]


def _extract_procpar_value(pdic, key, default=None):
    """Extract the first value from a procpar parameter, converting to float if possible."""
    if key not in pdic:
        return default
    values = pdic[key].get("values", [])
    if not values:
        return default
    val = values[0]
    try:
        return float(val)
    except (ValueError, TypeError):
        return val


def _extract_procpar_string(pdic, key, default=None):
    """Extract the first string value from a procpar parameter."""
    if key not in pdic:
        return default
    values = pdic[key].get("values", [])
    return values[0] if values else default


def _detect_agilent_encoding(pdic, ndim):
    """
    Detect the quadrature encoding for each dimension from procpar.

    Returns
    -------
    tuple of str
        Encoding per dimension.  One of 'QF', 'TPPI', 'STATES', 'STATES-TPPI'.
    """
    encodings = []

    # Direct dimension: check if data is complex
    np_val = _extract_procpar_value(pdic, "np", 0)
    is_direct_complex = int(np_val) % 2 == 0 and int(np_val) > 0
    encodings.append("QF" if not is_direct_complex else "QSIM")

    # Indirect dimensions
    phase_keys = ["phase", "phase2", "phase3"]
    for i in range(ndim - 1):
        if i < len(phase_keys):
            phase_vals = pdic.get(phase_keys[i], {}).get("values", [])
            if len(phase_vals) >= 2:
                encodings.append("STATES")
            elif len(phase_vals) == 1:
                encodings.append("TPPI")
            else:
                encodings.append("QF")
        else:
            encodings.append("QF")

    return tuple(encodings)


# ======================================================================================
# Public entry point
# ======================================================================================


def read_agilent(*paths, **kwargs):
    r"""
    Open Agilent/Varian NMR spectra.

    Parameters
    ----------
    *paths : `str`, `~pathlib.Path` objects or valid urls, optional
        The data source(s).  Can be:

        - a path to a directory containing ``fid`` and ``procpar`` files
        - a path to the ``fid`` file itself
        - a list of paths (datasets are merged if ``merge=True``)

    **kwargs : keyword parameters, optional
        See Other Parameters.

    Returns
    -------
    object : `NDDataset` or `ScpObjectList` of `NDDataset`
        The returned dataset(s).

    Other Parameters
    ----------------
    content : `bytes`, optional
        Raw bytes content instead of a filename.
    description : `str`, optional
        A custom description.
    directory : `~pathlib.Path`, optional
        Base directory for resolving relative paths.
    merge : `bool`, optional, default: ``False``
        If ``True`` and several filenames are provided, merge into a
        single dataset.
    origin : str, optional
        Override the origin label (default ``'agilent'``).

    See Also
    --------
    read : Generic reader inferring protocol from filename extension.

    Examples
    --------
    Reading a single Agilent FID directory:

    >>> scp.nmr.read('path/to/agilent_1d/')  # doctest: +SKIP

    """
    kwargs["filetypes"] = ["Agilent/Varian files (fid, procpar)"]
    kwargs["protocol"] = ["agilent"]
    importer = Importer()
    return importer(*paths, **kwargs)


# ======================================================================================
# Private reader implementation
# ======================================================================================


def _resolve_agilent_directory_target(filename, **kwargs):
    """Resolve an Agilent directory to the fid file within it."""
    protocol = kwargs.get("protocol")
    if protocol is None:
        return None
    if isinstance(protocol, str):
        protocol = [protocol]
    if "agilent" not in protocol:
        return None

    # If it's already a fid file, no resolution needed
    if filename.name == "fid":
        return None

    # Check if this directory contains a fid file
    fid_path = filename / "fid"
    if fid_path.exists():
        return [fid_path]

    return None


def _infer_agilent_filetype_key(filename, **kwargs):
    """Return the Agilent filetype key for fid files."""
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


@_importer_method
def _read_agilent(*args, **kwargs):
    dataset, path = args

    # Determine the directory containing fid and procpar
    data_dir = path.parent if path.name == "fid" else path

    fid_path = data_dir / "fid"
    procpar_path = data_dir / "procpar"

    if not fid_path.exists():
        warning_(f"No fid file found in {data_dir}")
        return None
    if not procpar_path.exists():
        warning_(f"No procpar file found in {data_dir}")
        return None

    # Read using vendored NMRGlue
    dic, data = read_varian(str(data_dir), fid_file="fid", procpar_file="procpar")
    pdic = dic.get("procpar", {})

    # Determine ndim from data shape
    ndim = data.ndim

    # Build metadata
    meta = Meta()

    # Basic info
    meta.datatype = f"{ndim}D" if ndim >= 2 else "FID"
    meta.pathname = str(path)
    meta.filename = data_dir.parent
    meta.origin = "agilent"

    # File header parameters
    meta.nblocks = dic.get("nblocks", 1)
    meta.ntraces = dic.get("ntraces", 1)
    meta.np_raw = dic.get("np", 0)

    # Procpar parameters — per-dimension lists
    meta.sw = [None] * ndim
    meta.sfrq = [None] * ndim
    meta.dfrq = [None] * ndim
    meta.td = list(data.shape)
    meta.nucleus = [None] * ndim
    meta.encoding = list(_detect_agilent_encoding(pdic, ndim))
    meta.iscomplex = [False] * ndim
    meta.isfreq = [False] * ndim

    # Direct dimension (last axis)
    meta.sw[-1] = _extract_procpar_value(pdic, "sw")
    meta.sfrq[-1] = _extract_procpar_value(pdic, "sfrq")
    meta.dfrq[-1] = _extract_procpar_value(pdic, "dfrq")
    nuc = _extract_procpar_string(pdic, "dn") or _extract_procpar_string(pdic, "tn")
    meta.nucleus[-1] = nuc
    meta.iscomplex[-1] = int(_extract_procpar_value(pdic, "np", 0)) % 2 == 0

    # Spectrometer frequency for direct dim: prefer sfrq, fallback to dfrq
    if meta.sfrq[-1] is None:
        meta.sfrq[-1] = meta.dfrq[-1]

    # Indirect dimensions (first axis(es))
    ni_keys = [
        ("ni", "sw2", "sfrq2", "dfrq2", "dn2", "tn2", "phase"),
        ("ni2", "sw3", "sfrq3", "dfrq3", "dn3", "tn3", "phase2"),
        ("ni3", "sw4", "sfrq4", "dfrq4", "dn4", "tn4", "phase3"),
    ]

    for i, (
        _ni_key,
        sw_key,
        sfrq_key,
        dfrq_key,
        dn_key,
        tn_key,
        phase_key,
    ) in enumerate(ni_keys):
        dim_idx = ndim - 2 - i
        if dim_idx < 0:
            break

        meta.sw[dim_idx] = _extract_procpar_value(pdic, sw_key)
        meta.sfrq[dim_idx] = _extract_procpar_value(pdic, sfrq_key)
        meta.dfrq[dim_idx] = _extract_procpar_value(pdic, dfrq_key)
        if meta.sfrq[dim_idx] is None:
            meta.sfrq[dim_idx] = meta.dfrq[dim_idx]

        nuc = _extract_procpar_string(pdic, dn_key) or _extract_procpar_string(
            pdic, tn_key
        )
        meta.nucleus[dim_idx] = nuc

        # Indirect dimensions are complex if phase array has 2+ values
        phase_vals = pdic.get(phase_key, {}).get("values", [])
        meta.iscomplex[dim_idx] = len(phase_vals) >= 2

    # Compute spectral width in Hz (already in Hz for Agilent)
    meta.sw_h = [None] * ndim
    for i in range(ndim):
        if meta.sw[i] is not None:
            meta.sw_h[i] = meta.sw[i] * ur.Hz

    # ns (number of scans)
    meta.ns = _extract_procpar_value(pdic, "ns", 1.0)
    meta.ne = _extract_procpar_value(pdic, "ne", 1.0)

    # Normalise amplitude to ns=1
    ns = float(meta.ns) if meta.ns else 1.0
    ne = float(meta.ne) if meta.ne else 1.0
    fac = ns * ne
    if fac > 0:
        data = data / fac
        meta.ns_norm = 1.0
    else:
        meta.ns_norm = ns

    # Pulse program
    meta.pulprog = _extract_procpar_string(pdic, "array") or _extract_procpar_string(
        pdic, "pulprog"
    )

    # Date
    date_str = _extract_procpar_string(pdic, "date")
    meta.date_raw = date_str

    # Acquisition time
    meta.at = _extract_procpar_value(pdic, "at")

    # Build coordinates
    coords = []

    def _make_coord(axis, size):
        """Build a single coordinate for the given axis."""
        if not meta.isfreq[axis]:
            sw_hz = meta.sw_h[axis]
            if sw_hz is not None and float(sw_hz.magnitude) > 0:
                dw = (1.0 / sw_hz).to("us")
                coordpoints = np.arange(size)
                coord = Coord(coordpoints * dw, title=f"F{axis + 1} acquisition time")
            else:
                coord = Coord(np.arange(size), title=f"F{axis + 1} acquisition time")

            freq = meta.sfrq[axis] or meta.dfrq[axis]
            if freq is not None:
                coord.meta["acquisition_frequency"] = freq * ur.MHz
            return coord

        # Frequency axis
        sw_hz = meta.sw_h[axis]
        freq_mhz = meta.sfrq[axis] or meta.dfrq[axis]

        if sw_hz is not None and freq_mhz is not None and size > 1:
            sizem = max(size - 1, 1)
            deltaf = -float(sw_hz.magnitude) / sizem
            first = float(freq_mhz) * 1e6 - deltaf * sizem / 2.0

            coordpoints = np.arange(size) * deltaf + first
            coord = Coord(coordpoints)
            coord.meta["acquisition_frequency"] = freq_mhz * ur.MHz
            coord.ito("ppm")

            nuc = meta.nucleus[axis]
            if nuc:
                import re

                regex = r"([^a-zA-Z]+)([a-zA-Z]+)"
                m = re.match(regex, str(nuc))
                nucleus = rf"$^{{{m[1]}}}{m[2]}$" if m else str(nuc)
            else:
                nucleus = ""
            coord.title = rf"$\delta\ {nucleus}$"
        else:
            coord = Coord(np.arange(size), title=f"F{axis + 1}")
        return coord

    for axis in range(ndim):
        coords.append(_make_coord(axis, meta.td[axis]))

    # Set data
    dataset.data = data

    # Apply quaternion encoding for 2D hypercomplex data
    if ndim >= 2 and meta.iscomplex[-2]:
        try:
            dataset.hyper.set_quaternion(inplace=True)
            meta.td[-1] = dataset.data.shape[-1]
            coords[-1] = _make_coord(ndim - 1, meta.td[-1])
        except AttributeError:
            warning_(
                "2D hypercomplex NMR data requires the spectrochempy-hypercomplex "
                "plugin. Install it with: pip install spectrochempy-hypercomplex",
                stacklevel=2,
            )

    # Apply metadata
    dataset.meta.update(meta)
    dataset.meta.readonly = True
    dataset.set_coordset(*tuple(coords))

    dataset.units = "count"
    dataset.title = "intensity"
    dataset.origin = "agilent"
    dataset.name = f"{data_dir.name} ({meta.datatype})"
    dataset.filename = data_dir

    if date_str:
        import contextlib

        with contextlib.suppress(ValueError, TypeError):
            dataset.acquisition_date = datetime.fromisoformat(str(date_str))

    dataset.history = "Imported from Agilent/Varian dataset"

    return dataset
