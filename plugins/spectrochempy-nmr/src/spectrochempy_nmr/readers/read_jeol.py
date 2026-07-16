# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
JEOL JDF file importers.

This module provides functionality to read JEOL JDF (JEOL Data Format)
NMR data files.

Functions
---------
- read_jeol : Main entry point for reading JEOL JDF files

Notes
-----
Supports 1D and 2D data only.  3D and higher-dimensional datasets
raise ``NotImplementedError``.

"""

__all__ = ["read_jeol"]

import contextlib
import re
from datetime import datetime

import numpy as np

from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.readers.importer import Importer
from spectrochempy.core.readers.importer import _importer_method
from spectrochempy.core.units import ur
from spectrochempy.utils._logging import warning_
from spectrochempy.utils.meta import Meta
from spectrochempy_nmr.extern.nmrglue._jeol import read_jeol as _read_jeol_raw

# ======================================================================================
# Public entry point
# ======================================================================================


def read_jeol(*paths, **kwargs):
    r"""
    Open JEOL JDF NMR spectra.

    Parameters
    ----------
    *paths : `str`, `~pathlib.Path` objects or valid urls, optional
        The data source(s).  Can be:

        - a path to a ``.jdf`` file
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
        Override the origin label (default ``'jeol'``).

    See Also
    --------
    read : Generic reader inferring protocol from filename extension.

    Examples
    --------
    Reading a single JEOL JDF file:

    >>> scp.nmr.read('path/to/experiment.jdf')  # doctest: +SKIP

    """
    kwargs["filetypes"] = ["JEOL JDF files (*.jdf)"]
    kwargs["protocol"] = ["jeol"]
    importer = Importer()
    return importer(*paths, **kwargs)


# ======================================================================================
# Private reader implementation
# ======================================================================================


# Mapping from JEOL axis_type strings to SpectroChemPy encoding + iscomplex
_JEOL_AXIS_TYPE_MAP = {
    "real": ("QF", False),
    "tppi": ("TPPI", True),
    "complex": ("STATES", True),
    "real_complex": ("STATES-TPPI", True),
}


def _jeol_axis_type_to_encoding(axis_type_str):
    """Map a JEOL data_axis_type string to (encoding, iscomplex)."""
    return _JEOL_AXIS_TYPE_MAP.get(axis_type_str, ("unknown", False))


@_importer_method
def _read_jdf(*args, **kwargs):
    dataset, path = args

    if not path.exists():
        warning_(f"File not found: {path}")
        return None

    # Read using vendored NMRGlue
    try:
        dic, data = _read_jeol_raw(str(path))
    except Exception as exc:
        warning_(f"Failed to read JEOL file {path}: {exc}")
        return None

    header = dic["header"]
    params = dic["parameters"]

    # Determine ndim from data shape
    ndim = data.ndim
    if ndim > 2:
        msg = f"JEOL datasets with {ndim} dimensions are not supported (max 2D)."
        raise NotImplementedError(msg)

    # Axis prefixes: x, y, z, ...
    axis_prefixes = ["x", "y", "z", "a", "b", "c", "d", "e"]

    # Build metadata
    meta = Meta()

    # Basic info
    meta.datatype = f"{ndim}D" if ndim >= 2 else "FID"
    meta.pathname = str(path)
    meta.filename = path.name
    meta.origin = "jeol"

    # Header parameters
    meta.endian = header.get("endian")
    meta.data_format = header.get("data_format")
    meta.data_type = header.get("data_type")

    # Per-dimension lists
    meta.sw = [None] * ndim
    meta.sfrq = [None] * ndim
    meta.td = list(data.shape)
    meta.nucleus = [None] * ndim
    meta.encoding = [None] * ndim
    meta.iscomplex = [False] * ndim
    meta.isfreq = [False] * ndim
    meta.offset = [None] * ndim

    for i in range(ndim):
        prefix = axis_prefixes[i]

        # Spectral width in Hz
        sw = params.get(f"{prefix}_sweep")
        meta.sw[i] = sw

        # Spectrometer frequency in MHz (JEOL stores in Hz)
        freq_hz = params.get(f"{prefix}_freq")
        if freq_hz is not None:
            meta.sfrq[i] = freq_hz / 1e6

        # Nucleus
        meta.nucleus[i] = params.get(f"{prefix}_domain")

        # Offset in ppm
        meta.offset[i] = params.get(f"{prefix}_offset")

        # Encoding and iscomplex from axis_type
        axis_type = header.get("data_axis_type", [None] * 8)[i]
        if axis_type is not None:
            enc, ic = _jeol_axis_type_to_encoding(axis_type)
            meta.encoding[i] = enc
            meta.iscomplex[i] = ic

    # Spectral width in Hz with units
    meta.sw_h = [None] * ndim
    for i in range(ndim):
        if meta.sw[i] is not None:
            meta.sw_h[i] = meta.sw[i] * ur.Hz

    # Number of scans
    meta.ns = params.get("total_scans", params.get("scans", 1))

    # Solvent
    meta.solvent = params.get("solvent")

    # Sample
    meta.sample = params.get("sample")

    # Experiment / pulse program
    meta.experiment = params.get("experiment")

    # Acquisition time
    meta.at = params.get("x_acq_time")

    # Relaxation delay
    meta.d1 = params.get("relaxation_delay")

    # 90-degree pulse width
    meta.pw = params.get("x90")

    # Build coordinates
    coords = []

    def _make_coord(axis, size):
        """Build a single coordinate for the given axis."""
        sw_hz = meta.sw_h[axis]
        freq_mhz = meta.sfrq[axis]
        nuc = meta.nucleus[axis]
        offset_ppm = meta.offset[axis]

        if not meta.isfreq[axis]:
            if sw_hz is not None and float(sw_hz.magnitude) > 0:
                dw = (1.0 / sw_hz).to("us")
                coordpoints = np.arange(size)
                coord = Coord(coordpoints * dw, title=f"F{axis + 1} acquisition time")
            else:
                coord = Coord(np.arange(size), title=f"F{axis + 1} acquisition time")

            if freq_mhz is not None:
                coord.meta["acquisition_frequency"] = freq_mhz * ur.MHz
            return coord

        if sw_hz is not None and freq_mhz is not None and size > 1:
            sizem = max(size - 1, 1)
            deltaf = -float(sw_hz.magnitude) / sizem

            # Use offset to set the carrier position
            if offset_ppm is not None:
                first = (
                    float(freq_mhz) * 1e6
                    + float(offset_ppm) * float(freq_mhz) * 1e3
                    - deltaf * sizem / 2.0
                )
            else:
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
    dataset.origin = "jeol"
    dataset.name = f"{path.stem} ({meta.datatype})"
    dataset.filename = path.parent

    # Creation date
    creation = header.get("creation_time")
    if creation:
        with contextlib.suppress(ValueError, TypeError):
            dataset.acquisition_date = datetime(
                creation.get("year", 1970),
                creation.get("month", 1),
                creation.get("day", 1),
            )

    dataset.history = "Imported from JEOL JDF dataset"

    return dataset
