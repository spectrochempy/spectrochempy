# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
TecMag TNT file importers.

This module provides functionality to read TecMag TNT NMR data files.

Functions
---------
- read_tecmag : Main entry point for reading TecMag TNT files

Notes
-----
Supports 1D and 2D data only.  3D and higher-dimensional datasets
raise ``NotImplementedError``.

"""

__all__ = ["read_tecmag"]

import re
from datetime import datetime

import numpy as np

from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.readers.importer import Importer
from spectrochempy.core.readers.importer import _importer_method
from spectrochempy.core.units import ur
from spectrochempy.utils._logging import warning_
from spectrochempy.utils.meta import Meta
from spectrochempy_nmr.extern.nmrglue._tecmag import read as _read_tnt_raw

# ======================================================================================
# Public entry point
# ======================================================================================


def read_tecmag(*paths, **kwargs):
    r"""
    Open TecMag TNT NMR spectra.

    Parameters
    ----------
    *paths : `str`, `~pathlib.Path` objects or valid urls, optional
        The data source(s).  Can be:

        - a path to a ``.tnt`` file
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
        Override the origin label (default ``'tecmag'``).

    See Also
    --------
    read : Generic reader inferring protocol from filename extension.

    Examples
    --------
    Reading a single TecMag TNT file:

    >>> scp.nmr.read('path/to/experiment.tnt')  # doctest: +SKIP

    """
    kwargs["filetypes"] = ["TecMag TNT files (*.tnt)"]
    kwargs["protocol"] = ["tecmag"]
    importer = Importer()
    return importer(*paths, **kwargs)


# ======================================================================================
# Private reader implementation
# ======================================================================================


def _parse_tecmag_date(date_bytes):
    """Parse a TecMag date string (bytes) into a datetime or None."""
    if not isinstance(date_bytes, bytes):
        return None
    s = date_bytes.decode("latin1").rstrip("\x00").strip()
    # Remove trailing junk (e.g. '2222...')
    s = re.sub(r"[^0-9/\-: ]+$", "", s).strip()
    if not s:
        return None
    for fmt in ("%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


@_importer_method
def _read_tnt(*args, **kwargs):
    dataset, path = args

    if not path.exists():
        warning_(f"File not found: {path}")
        return None

    # Read using vendored NMRGlue
    try:
        dic, data = _read_tnt_raw(str(path))
    except Exception as exc:
        warning_(f"Failed to read TecMag file {path}: {exc}")
        return None

    # Determine ndim from actual data dimensions (squeeze trailing size-1 dims)
    actual_npts = list(dic["actual_npts"])
    ndim = 0
    for i, n in enumerate(actual_npts):
        if n > 1:
            ndim = i + 1
    if ndim == 0:
        ndim = 1

    if ndim > 2:
        msg = f"TecMag datasets with {ndim} dimensions are not supported (max 2D)."
        raise NotImplementedError(msg)

    # Squeeze data to ndim
    data = data.reshape(actual_npts[:ndim])

    # Build metadata
    meta = Meta()

    meta.datatype = f"{ndim}D" if ndim >= 2 else "FID"
    meta.pathname = str(path)
    meta.filename = path.name
    meta.origin = "tecmag"

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
        # Spectral width in Hz
        meta.sw[i] = float(dic["sw"][i])

        # Spectrometer frequency in MHz
        meta.sfrq[i] = float(dic["ob_freq"][i])

        # Nucleus name (strip null bytes and padding)
        nuc_raw = dic["nuclei"][i]
        if isinstance(nuc_raw, bytes):
            nuc = nuc_raw.decode("latin1").split("\x00", 1)[0].strip()
            nuc = re.sub(r"[^a-zA-Z0-9]+$", "", nuc)
        else:
            nuc = str(nuc_raw)
        meta.nucleus[i] = nuc if nuc else None

        # Frequency domain flag from fft_flag
        fft_flag = dic.get("fft_flag")
        if fft_flag is not None:
            meta.isfreq[i] = bool(fft_flag[i])

        # Data is always complex in .tnt
        meta.iscomplex[i] = True

        # TecMag 1D data uses a direct complex quadrature acquisition.
        if ndim == 1:
            meta.encoding[i] = "QSIM"

    # Spectral width in Hz with units
    meta.sw_h = [None] * ndim
    for i in range(ndim):
        if meta.sw[i] is not None:
            meta.sw_h[i] = meta.sw[i] * ur.Hz

    # Number of scans
    meta.ns = int(dic.get("actual_scans", dic.get("scans", 1)))

    # Solvent
    lock_solvent = dic.get("lock_solvent")
    if isinstance(lock_solvent, bytes):
        meta.solvent = lock_solvent.decode("latin1").rstrip("\x00").strip()
        meta.solvent = re.sub(r"[^a-zA-Z0-9]+$", "", meta.solvent)
    elif lock_solvent:
        meta.solvent = str(lock_solvent)

    # Temperature
    temp = dic.get("actual_temperature")
    if temp is not None and float(temp) > 0:
        meta.temperature = float(temp) * ur.K

    # Experiment / pulse program
    seq = dic.get("sequence")
    if isinstance(seq, bytes):
        s = seq.decode("latin1").rstrip("\x00").strip()
        meta.experiment = s if s else None
    elif seq:
        meta.experiment = str(seq)

    # Acquisition time
    at = dic.get("acq_time")
    if at is not None:
        meta.at = float(at)

    # Build coordinates
    coords = []

    def _make_coord(axis, size):
        """Build a single coordinate for the given axis."""
        sw_hz = meta.sw_h[axis]
        freq_mhz = meta.sfrq[axis]
        nuc = meta.nucleus[axis]

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
    dataset.origin = "tecmag"
    dataset.name = f"{path.stem} ({meta.datatype})"
    dataset.filename = path.parent

    # Creation date
    date_val = dic.get("date")
    if date_val is not None:
        dt = _parse_tecmag_date(date_val)
        if dt is not None:
            dataset.acquisition_date = dt

    dataset.history = "Imported from TecMag TNT dataset"

    return dataset
