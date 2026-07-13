# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Functions for reading SIMPSON NMR data files.

SIMPSON (SIMulation of SOlid-state NMR) is a simulation program whose output
files can be stored in several ASCII and binary formats.  This module reads
the most common ones.

Supported formats
-----------------
- TEXT : SIMPSON text format with ``SIMP``/``DATA`` header.
- BINARY : SIMPSON binary format with ``SIMP``/``FORMAT``/``DATA`` header.
- XREIM : 1D indexed format, columns ``<unit> <real> <imag>``.
- XYREIM : 2D indexed format, columns ``<ni_unit> <np_unit> <real> <imag>``.
- RAWBIN : Raw float32 binary (requires external shape/domain hints).

References
----------
- nmrglue's ``simpson.py`` provided the original format description and served
  as a reference for the binary decoding logic.
- SIMPSON home page: http://www.bionmr.chem.au.dk/bionmr/software/simpson.php

"""

from __future__ import annotations

import math
from warnings import warn

import numpy as np

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def read(filename, ftype=None, ndim=None, NP=None, NI=None, spe=None):
    """
    Read a SIMPSON file.

    Parameters
    ----------
    filename : str or path
        Name of SIMPSON file to read data from.
    ftype : {None, 'TEXT', 'BINARY', 'XREIM', 'XYREIM', 'RAWBIN'}, optional
        File type.  If ``None``, the type is guessed from the file content.
    ndim : {None, 1, 2}, optional
        Dimensionality, only used for ``RAWBIN``.
    NP : int, optional
        Number of complex points in the direct dimension, only used for 2D
        ``RAWBIN``.
    NI : int, optional
        Number of points in the indirect dimension, only used for 2D
        ``RAWBIN``.
    spe : bool, optional
        Whether the data is in the frequency domain, only used for
        ``RAWBIN``.

    Returns
    -------
    dic : dict
        Dictionary of spectral parameters.
    data : ndarray
        Complex array of spectral data.

    """
    if ftype is None:
        ftype = guess_ftype(filename)

    if ftype == "TEXT":
        return read_text(filename)
    if ftype == "BINARY":
        return read_binary(filename)
    if ftype == "XREIM":
        return read_xreim(filename)
    if ftype == "XYREIM":
        return read_xyreim(filename)
    if ftype == "RAWBIN":
        if spe is None:
            raise ValueError("spe must be True or False for raw_bin data")
        if ndim == 1:
            return read_raw_bin_1d(filename, spe)
        if ndim == 2:
            if NP is None or NI is None:
                raise ValueError("NP and NI must be given for raw_bin data")
            return read_raw_bin_2d(filename, NP, NI, spe)
        raise ValueError("ndim must be 1 or 2 for raw_bin data")

    raise ValueError(f"unknown ftype: {ftype}")


def guess_ftype(filename):
    """Determine a SIMPSON file type from the first few lines."""
    with open(filename, "rb") as f:
        first = f.readline()

    # SIMPSON text/binary files start with "SIMP"
    if first[:4] == b"SIMP":
        with open(filename, "rb") as f:
            for line in f:
                line = line.decode("ascii", "ignore").strip("\n")
                if line[:6] == "FORMAT":
                    return "BINARY"
                if line == "DATA":
                    break
        return "TEXT"

    # Otherwise use column count on first few non-empty lines
    with open(filename, "rb") as f:
        widths = []
        for _ in range(8):
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            widths.append(len(line.split()))
        if not widths:
            raise ValueError("empty SIMPSON file")
        first_width = widths[0]
        if all(w == first_width for w in widths):
            if first_width == 3:
                return "XREIM"
            if first_width == 4:
                return "XYREIM"
        return "RAWBIN"


# ---------------------------------------------------------------------------
# TEXT format
# ---------------------------------------------------------------------------


def read_text(filename):
    """Read a SIMPSON text file."""
    dic = {}

    with open(filename) as f:
        # Parse header
        for line in f:
            line = line.strip("\n").strip()
            if line == "SIMP":
                continue
            if line == "DATA":
                break
            if "=" in line:
                key, value = line.split("=", 1)
                dic[key.strip()] = value.strip()
            else:
                warn(f"Skipping SIMPSON header line: {line!r}", stacklevel=2)

        # Convert known numeric keys
        for key in ("SW", "SW1"):
            if key in dic:
                dic[key] = float(dic[key])
        for key in ("NP", "NI", "NELEM"):
            if key in dic:
                dic[key] = int(dic[key])

        if "NELEM" not in dic:
            dic["NELEM"] = 1

        if "NI" in dic:  # 2D
            shape = (dic["NI"] * dic["NELEM"], dic["NP"])
            data = np.empty(shape, dtype="complex64")
            for iline, line in enumerate(f):
                if line[:3] == "END":
                    break
                vals = line.split()
                if len(vals) < 2:
                    continue
                r_val, i_val = float(vals[0]), float(vals[1])
                ni_idx, np_idx = divmod(iline, dic["NP"])
                if ni_idx >= shape[0]:
                    break
                data.real[ni_idx, np_idx] = r_val
                data.imag[ni_idx, np_idx] = i_val
        else:  # 1D
            npts = dic["NP"] * dic["NELEM"]
            data = np.empty(npts, dtype="complex64")
            for iline, line in enumerate(f):
                if line[:3] == "END":
                    break
                vals = line.split()
                if len(vals) < 2:
                    continue
                data.real[iline] = float(vals[0])
                data.imag[iline] = float(vals[1])
            data = data.reshape(dic["NELEM"], -1)
            if dic["NELEM"] == 1:
                data = data.reshape(-1)

    return dic, data


# ---------------------------------------------------------------------------
# Indexed XREIM / XYREIM formats
# ---------------------------------------------------------------------------


def read_xreim(filename):
    """Read a 1D indexed SIMPSON file (``-xreim``)."""
    units = []
    reals = []
    imags = []

    with open(filename) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3:
                continue
            units.append(float(parts[0]))
            reals.append(float(parts[1]))
            imags.append(float(parts[2]))

    data = np.empty(len(reals), dtype="complex64")
    data.real = np.array(reals, dtype=np.float32)
    data.imag = np.array(imags, dtype=np.float32)
    return {"units": np.array(units, dtype=np.float32)}, data


def read_xyreim(filename):
    """Read a 2D indexed SIMPSON file (``-xyreim``)."""
    lines = []
    with open(filename) as f:
        for line in f:
            if line.strip():
                lines.append([float(x) for x in line.split()])

    lines = np.array(lines)
    # Lines are grouped by indirect point: NI blocks of NP lines
    # Each line: ni_unit, np_unit, real, imag
    ni_values = lines[:, 0]
    np_values = lines[:, 1]

    # Number of unique ni units = NI, unique np units = NP
    ni_unique = np.unique(ni_values)
    np_unique = np.unique(np_values)
    NI = len(ni_unique)
    NP = len(np_unique)

    data = np.empty((NI, NP), dtype="complex64")
    units = np.recarray((NI, NP), dtype=[("ni_unit", "f8"), ("np_unit", "f8")])

    for ni_unit, np_unit, r_val, i_val in lines:
        ni_idx = np.searchsorted(ni_unique, ni_unit)
        np_idx = np.searchsorted(np_unique, np_unit)
        data.real[ni_idx, np_idx] = r_val
        data.imag[ni_idx, np_idx] = i_val
        units[ni_idx, np_idx].ni_unit = ni_unit
        units[ni_idx, np_idx].np_unit = np_unit

    return {"units": units}, data


# ---------------------------------------------------------------------------
# Raw binary format
# ---------------------------------------------------------------------------


def read_raw_bin_1d(filename, spe=False):
    """Read a 1D raw binary SIMPSON file."""
    data = _unappend_data(np.fromfile(filename, dtype="float32"))
    if spe:
        return {}, data[::-1]
    return {}, data


def read_raw_bin_2d(filename, NP, NI, spe=False):
    """Read a 2D raw binary SIMPSON file."""
    data = np.fromfile(filename, dtype="float32").reshape(NI, NP * 2)
    if spe:
        cdata = np.empty((NI, NP), dtype="complex64")
        cdata.real = np.roll(np.roll(data[::-1, NP - 1 :: -1], 1, 1), 1, 0)
        cdata.imag = np.roll(data[::-1, 2 * NP : NP - 1 : -1], 1, 0)
        return {}, cdata
    return {}, _unappend_data(data)


def _unappend_data(data):
    """Return complex data with the last axis unappended (real | imag)."""
    h = data.shape[-1] // 2
    return np.array(data[..., :h] + data[..., h:] * 1.0j, dtype="complex64")


# ---------------------------------------------------------------------------
# BINARY format (ASCII header + encoded binary payload)
# ---------------------------------------------------------------------------


_BASE = 33


def _chars2bytes(chars):
    """Convert four characters from a data block into 3 bytes."""
    c0, c1, c2, c3 = (ord(c) - _BASE for c in chars)
    return [
        (c0 & ~(~0 << 6)) | ((c1 << 2) & (~0 << (8 - 2))),
        (c1 & ~(~0 << 4)) | ((c2 << 2) & (~0 << (8 - 4))),
        (c2 & ~(~0 << 2)) | ((c3 << 2) & (~0 << (8 - 6))),
    ]


def _bytes2float(b):
    """Convert four bytes to a float using SIMPSON's custom encoding."""
    b0, b1, b2, b3 = b
    mantissa = ((b2 % 128) << 16) + (b1 << 8) + b0
    exponent = (b3 % 128) * 2 + (b2 >= 128) * 1
    negative = b3 >= 128

    e = exponent - 0x7F
    m = abs(mantissa) / np.float64(1 << 23)

    if negative:
        return -math.ldexp(m, e)
    return math.ldexp(m, e)


def read_binary(filename):
    """Read a SIMPSON binary file."""
    dic = {}

    with open(filename) as f:
        for line in f:
            line = line.strip("\n").strip()
            if line == "SIMP":
                continue
            if line == "DATA":
                break
            if "=" in line:
                key, value = line.split("=", 1)
                dic[key.strip()] = value.strip()
            else:
                warn(f"Skipping SIMPSON header line: {line!r}", stacklevel=2)

        for key in ("SW", "SW1"):
            if key in dic:
                dic[key] = float(dic[key])
        for key in ("NP", "NI", "NELEM"):
            if key in dic:
                dic[key] = int(dic[key])
        if "NELEM" not in dic:
            dic["NELEM"] = 1

        chardata = "".join(line.strip("\n") for line in f)
    chardata = chardata[:-3] if chardata.endswith("END") else chardata

    nquads, mod = divmod(len(chardata), 4)
    if mod != 0:
        raise ValueError("SIMPSON binary data length is not a multiple of 4")

    raw_bytes = []
    for i in range(nquads):
        raw_bytes.extend(_chars2bytes(chardata[i * 4 : (i + 1) * 4]))

    num_points = len(raw_bytes) // 4
    data = np.empty(num_points, dtype="float32")
    for i in range(num_points):
        data[i] = _bytes2float(raw_bytes[i * 4 : (i + 1) * 4])
    data = data.view("complex64")

    if "NI" in dic:
        return dic, data.reshape(dic["NI"] * dic["NELEM"], dic["NP"])
    return dic, data.reshape(dic["NELEM"], dic["NP"])
