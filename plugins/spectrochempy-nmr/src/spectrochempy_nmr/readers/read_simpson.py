# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
SIMPSON file importers.

This module provides functionality to read SIMPSON NMR data files
(``.spe``, ``.fid`` and related formats).

Functions
---------
- read_simpson : Main entry point for reading SIMPSON files

Notes
-----
Supports 1D and 2D data only.  3D and higher-dimensional datasets
raise ``NotImplementedError``.

"""

__all__ = ["read_simpson"]

import re
from pathlib import Path

import numpy as np

from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.readers.importer import Importer
from spectrochempy.core.readers.importer import _importer_method
from spectrochempy.core.units import ur
from spectrochempy.utils._logging import warning_
from spectrochempy.utils.meta import Meta
from spectrochempy_nmr.extern.nmrglue._simpson import read as _read_simpson_raw

# ======================================================================================
# Public entry point
# ======================================================================================


def read_simpson(*paths, **kwargs):
    r"""
    Open SIMPSON NMR spectra.

    Parameters
    ----------
    *paths : `str`, `~pathlib.Path` objects or valid urls, optional
        The data source(s).  Can be:

        - a path to a SIMPSON data file (``.spe``, ``.fid``, ``.txt``)
        - a path to a SIMPSON experiment directory containing a ``.in`` file
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
        Override the origin label (default ``'simpson'``).

    See Also
    --------
    read : Generic reader inferring protocol from filename extension.

    Examples
    --------
    Reading a single SIMPSON file:

    >>> scp.nmr.read('path/to/experiment.spe')  # doctest: +SKIP

    """
    kwargs["filetypes"] = ["SIMPSON files (*.spe *.fid)"]
    kwargs["protocol"] = ["simpson"]
    importer = Importer()
    return importer(*paths, **kwargs)


# ======================================================================================
# Private helpers
# ======================================================================================


def _parse_simpson_in(filepath):
    """
    Parse a SIMPSON ``.in`` Tcl input file for simulation metadata.

    Returns a dictionary with keys such as ``channels``, ``nuclei``,
    ``np``, ``ni``, ``sw``, ``sw1`` when present.
    """
    meta = {}
    if not filepath.exists():
        return meta

    try:
        text = filepath.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return meta

    # Extract spinsys block
    spinsys_match = re.search(r"spinsys\s*\{(.*?)\}", text, re.DOTALL | re.IGNORECASE)
    if spinsys_match:
        block = spinsys_match.group(1)
        channels = re.search(r"channels\s+(.*)", block)
        if channels:
            meta["channels"] = channels.group(1).strip().split()
        nuclei = re.search(r"nuclei\s+(.*)", block)
        if nuclei:
            meta["nuclei"] = nuclei.group(1).strip().split()

    # Extract par block
    par_match = re.search(r"par\s*\{(.*?)\}", text, re.DOTALL | re.IGNORECASE)
    if par_match:
        block = par_match.group(1)
        for key in ("np", "ni", "sw", "sw1", "spin_rate", "proton_frequency"):
            m = re.search(rf"{key}\s+([^\n#]+)", block)
            if m:
                val = m.group(1).strip()
                # Skip Tcl expressions that are not simple numbers
                try:
                    meta[key] = (
                        float(val) if "." in val or "e" in val.lower() else int(val)
                    )
                except ValueError:
                    meta[key] = val

    return meta


# ======================================================================================
# Private reader implementation
# ======================================================================================


def _read_simpson_core(dataset, path):
    """Core SIMPSON reading logic shared across file extensions."""
    if not path.exists():
        warning_(f"File not found: {path}")
        return None

    in_meta = {}
    data_path = path

    # If a directory is passed, look for an .in file and a data file
    if path.is_dir():
        in_files = list(path.glob("*.in"))
        if in_files:
            in_meta = _parse_simpson_in(in_files[0])
        # Prefer .spe (frequency domain) then .fid (time domain)
        candidates = list(path.glob("*.spe")) + list(path.glob("*.fid"))
        if not candidates:
            candidates = list(path.glob("*.txt"))
        if not candidates:
            warning_(f"No SIMPSON data file found in directory: {path}")
            return None
        data_path = candidates[0]

    # If a .in file is passed directly, look for sibling data files
    elif path.suffix.lower() == ".in":
        in_meta = _parse_simpson_in(path)
        parent = path.parent
        base = path.stem
        candidates = [
            parent / f"{base}.spe",
            parent / f"{base}.fid",
            parent / f"{base}_text.spe",
            parent / f"{base}_text.fid",
        ]
        candidates.extend(list(parent.glob("*.spe")))
        candidates.extend(list(parent.glob("*.fid")))
        data_path = next((c for c in candidates if c.exists()), None)
        if data_path is None:
            warning_(f"No SIMPSON data file found for input: {path}")
            return None

    # Look for a sibling .in file to enrich metadata
    if not in_meta:
        sibling_in = data_path.with_suffix(".in")
        if not sibling_in.exists():
            sibling_in = data_path.parent / f"{data_path.stem.split('_')[0]}.in"
        if sibling_in.exists():
            in_meta = _parse_simpson_in(sibling_in)

    # Read using the SIMPSON parser
    try:
        dic, data = _read_simpson_raw(str(data_path))
    except Exception as exc:
        warning_(f"Failed to read SIMPSON file {data_path}: {exc}")
        return None

    # Determine ndim and squeeze trailing singletons
    if data.ndim >= 2 and data.shape[0] == 1:
        data = data.reshape(data.shape[1:])
    ndim = data.ndim

    if ndim > 2:
        msg = f"SIMPSON datasets with {ndim} dimensions are not supported (max 2D)."
        raise NotImplementedError(msg)

    # Build metadata
    meta = Meta()

    meta.datatype = (
        f"{ndim}D" if ndim >= 2 else ("SPE" if _is_spe(data_path, dic) else "FID")
    )
    meta.pathname = str(path)
    meta.filename = path.name
    meta.origin = "simpson"

    # Per-dimension lists
    meta.sw = [None] * ndim
    meta.sfrq = [None] * ndim
    meta.td = list(data.shape)
    meta.nucleus = [None] * ndim
    meta.encoding = [None] * ndim
    meta.iscomplex = [True] * ndim
    meta.isfreq = [False] * ndim
    meta.offset = [None] * ndim

    # Spectral widths (Hz)
    sw_keys = ["SW", "SW"] if ndim == 1 else ["SW1", "SW"]
    for i in range(ndim):
        key = sw_keys[i]
        if key in dic:
            meta.sw[i] = float(dic[key])
        elif key.lower() in dic:
            meta.sw[i] = float(dic[key.lower()])
        elif key in in_meta:
            meta.sw[i] = float(in_meta[key])

    # Reference frequencies (MHz)
    proton_freq = in_meta.get("proton_frequency")
    if proton_freq is not None:
        # proton_frequency is in Hz
        ref_mhz = float(proton_freq) / 1e6
        for i in range(ndim):
            meta.sfrq[i] = ref_mhz

    # Nuclei from .in file or defaults
    nuclei = in_meta.get("nuclei", in_meta.get("channels", []))
    for i in range(min(ndim, len(nuclei))):
        meta.nucleus[i] = nuclei[i]

    # Frequency domain flag
    if _is_spe(data_path, dic):
        for i in range(ndim):
            meta.isfreq[i] = True

    # Encoding: SIMPSON simulations are typically QF/STATES
    if ndim == 1:
        meta.encoding[0] = "QF"
    else:
        meta.encoding[0] = "STATES"
        meta.encoding[1] = "QF"

    # Spectral width with units
    meta.sw_h = [None] * ndim
    for i in range(ndim):
        if meta.sw[i] is not None:
            meta.sw_h[i] = meta.sw[i] * ur.Hz

    # Spin rate
    if "spin_rate" in in_meta:
        meta.spin_rate = float(in_meta["spin_rate"]) * ur.Hz

    # Build coordinates
    coords = []

    def _make_coord(axis, size):
        """Build a single coordinate for the given axis."""
        sw_hz = meta.sw_h[axis]
        freq_mhz = meta.sfrq[axis]
        nuc = meta.nucleus[axis]

        if sw_hz is not None and freq_mhz is not None and size > 1:
            sizem = max(size - 1, 1)
            deltaf = -float(sw_hz.magnitude) / sizem
            first = float(freq_mhz) * 1e6 - deltaf * sizem / 2.0
            coordpoints = np.arange(size) * deltaf + first
            coord = Coord(coordpoints)
            coord.meta["acquisition_frequency"] = freq_mhz * ur.MHz
            coord.ito("ppm")

            if nuc:
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

    # Apply metadata
    dataset.meta.update(meta)
    dataset.meta.readonly = True
    dataset.set_coordset(*tuple(coords))

    dataset.units = "count"
    dataset.title = "intensity"
    dataset.origin = "simpson"
    dataset.name = f"{path.stem} ({meta.datatype})"
    dataset.filename = path.parent

    dataset.history = "Imported from SIMPSON dataset"

    return dataset


@_importer_method
def _read_spe(*args, **kwargs):
    """Read a SIMPSON ``.spe`` file."""
    return _read_simpson_core(args[0], args[1])


@_importer_method
def _read_fid(*args, **kwargs):
    """Read a SIMPSON ``.fid`` file."""
    return _read_simpson_core(args[0], args[1])


@_importer_method
def _read_in(*args, **kwargs):
    """Read a SIMPSON ``.in`` input file (data file resolved separately)."""
    return _read_simpson_core(args[0], args[1])


def _is_spe(path, dic):
    """Return True if the file appears to be frequency-domain data."""
    if isinstance(path, Path):
        if ".spe" in path.name.lower():
            return True
        if ".fid" in path.name.lower():
            return False
    # Frequency-domain files often have units information
    return "units" in dic
