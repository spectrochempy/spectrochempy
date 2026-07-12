# ======================================================================================
# Agilent / Varian file reading
# ======================================================================================
#
# Adapted from nmrglue.fileio.varian (BSD 3-Clause, Jonathan J. Helbus).
# Read-only subset: procpar parsing, binary FID reading, shape/order detection.
# Write functions, low-memory reader, and fid_nd class are omitted.

import contextlib
import os
import struct as _struct

import numpy as np

from ._bruker import _merge_dict


def read_varian(
    dir=".", fid_file="fid", procpar_file="procpar", shape=None, torder=None
):
    """
    Read Agilent/Varian files from a directory.

    Parameters
    ----------
    dir : str
        Directory containing the fid and procpar files.
    fid_file : str
        Filename of the binary FID file.
    procpar_file : str
        Filename of the procpar parameter file.
    shape : tuple of ints, optional
        Data shape.  Determined automatically if ``None``.
    torder : {None, 'r', 'o', 'f'}, optional
        Trace ordering.  Determined automatically if ``None``.

    Returns
    -------
    dic : dict
        Dictionary with file header parameters and ``'procpar'`` key.
    data : ndarray
        Array of NMR data (complex-valued).
    """
    pdic = read_varian_procpar(os.path.join(dir, procpar_file))

    if shape is None:
        shape = find_varian_shape(pdic)
    if torder is None and shape is not None and len(shape) >= 3:
        torder = find_varian_torder(pdic, shape)

    fname = os.path.join(dir, fid_file)
    dic, data = read_varian_fid(fname, shape=shape, torder=torder)

    dic["procpar"] = pdic
    return dic, data


def read_varian_fid(filename, shape=None, torder="flat", read_blockhead=False):
    """
    Read an Agilent/Varian binary fid file.

    Parameters
    ----------
    filename : str
        Path to the binary fid file.
    shape : tuple of ints, optional
        Expected data shape.  If ``None``, data is returned unshaped.
    torder : {'f', 'r', 'o'}, optional
        Trace ordering for 3D+ data.
    read_blockhead : bool, optional
        If ``True``, include block headers in the returned dictionary.

    Returns
    -------
    dic : dict
        File header dictionary.
    data : ndarray
        Complex NMR data.
    """
    with open(filename, "rb") as f:
        dic = _varian_fileheader2dic(_varian_get_fileheader(f))

        if dic["ntraces"] != 1:
            data = _varian_get_nblocks_ntraces(
                f,
                dic["nblocks"],
                dic["ntraces"],
                dic["np"],
                dic["nbheaders"],
                _varian_find_dtype(dic),
                read_blockhead,
            )
        else:
            data = _varian_get_nblocks(
                f,
                dic["nblocks"],
                dic["np"],
                dic["nbheaders"],
                _varian_find_dtype(dic),
                read_blockhead,
            )

    data = _varian_uninterleave_data(data)

    if data.shape[0] == 1:
        data = np.squeeze(data, axis=0)

    if shape is None:
        return dic, data

    if len(shape) >= 3 and data.ndim >= 2:
        try:
            data = _varian_reorder_data(data, shape, torder)
            return dic, data
        except Exception:  # noqa: BLE001
            return dic, data

    with contextlib.suppress(ValueError):
        data = data.reshape(shape)

    return dic, data


def read_varian_procpar(filename):
    """
    Read an Agilent/Varian procpar file.

    Parameters
    ----------
    filename : str
        Path to the procpar file.

    Returns
    -------
    dict
        Dictionary keyed by parameter name.  Each value is a dict with
        ``'name'``, ``'subtype'``, ``'basictype'``, ``'values'``, etc.
    """
    dic = {}
    length = os.stat(filename).st_size
    with open(filename, "rb") as f:
        while f.tell() != length:
            p = _varian_get_parameter(f)
            dic[p["name"]] = p
    return dic


def find_varian_shape(pdic):
    """
    Determine the data shape from a procpar dictionary.

    Parameters
    ----------
    pdic : dict
        Procpar dictionary (as returned by :func:`read_varian_procpar`).

    Returns
    -------
    tuple of ints or None
        Data shape, or ``None`` if it cannot be determined.
    """
    try:
        shape = [int(pdic["np"]["values"][0]) // 2]
    except Exception:  # noqa: BLE001
        return None

    if "array" in pdic:
        array_name = pdic["array"]["values"][0]
        array_name = array_name.split(",")[-1]
        if not array_name.startswith("phase") and array_name in pdic:
            shape.insert(0, len(pdic[array_name]["values"]))

    if "seqcon" in pdic:
        for nv_key in ("nv", "nv2", "nv3"):
            if nv_key in pdic:
                s = max(int(pdic[nv_key]["values"][0]), 1)
                if s > 1:
                    shape.insert(0, s)
        return tuple(shape)

    for ni_key, phase_key in (("ni", "phase"), ("ni2", "phase2"), ("ni3", "phase3")):
        if ni_key in pdic:
            multi = 2
            if phase_key in pdic:
                multi = len(pdic[phase_key]["values"])
            s = max(int(pdic[ni_key]["values"][0]), 1)
            shape.insert(0, s * multi)

    return tuple(shape)


def find_varian_torder(pdic, shape):
    """
    Determine trace ordering from a procpar dictionary.

    Parameters
    ----------
    pdic : dict
        Procpar dictionary.
    shape : tuple of ints
        Data shape.

    Returns
    -------
    str
        ``'f'`` (flat), ``'r'`` (regular), or ``'o'`` (opposite).
    """
    ndim = len(shape)
    if ndim < 3:
        return "f"

    if "array" not in pdic:
        return "r"

    al = pdic["array"]["values"][0].split(",")

    if not all(s.startswith("phase") for s in al):
        ndim -= 1
        if ndim < 3:
            return "f"

    if ndim == 3 and "phase" in al and "phase2" in al:
        return "r" if al.index("phase") > al.index("phase2") else "o"

    if ndim == 4 and all(f"phase{i}" in al for i in ("", "2", "3")):
        idx = [al.index(f"phase{i}") for i in ("", "2", "3")]
        if idx == sorted(idx):
            return "r"
        if idx == sorted(idx, reverse=True):
            return "o"

    return "r"


# ---------------------------------------------------------------------------
# Low-level Agilent binary helpers
# ---------------------------------------------------------------------------


def _varian_get_fileheader(f):
    """Read 32-byte big-endian file header."""
    return _struct.unpack(">6lhhl", f.read(32))


def _varian_fileheader2dic(head):
    """Convert file header tuple to dictionary with unpacked status bits."""
    d = {
        "nblocks": head[0],
        "ntraces": head[1],
        "np": head[2],
        "ebytes": head[3],
        "tbytes": head[4],
        "bbytes": head[5],
        "vers_id": head[6],
        "status": head[7],
        "nbheaders": head[8],
    }
    s = d["status"]
    d["S_DATA"] = (s & 0x1) // 0x1
    d["S_SPEC"] = (s & 0x2) // 0x2
    d["S_32"] = (s & 0x4) // 0x4
    d["S_FLOAT"] = (s & 0x8) // 0x8
    d["S_COMPLEX"] = (s & 0x10) // 0x10
    d["S_HYPERCOMPLEX"] = (s & 0x20) // 0x20
    d["S_ACQPAR"] = (s & 0x80) // 0x80
    d["S_SECND"] = (s & 0x100) // 0x100
    d["S_TRANSF"] = (s & 0x200) // 0x200
    d["S_NP"] = (s & 0x800) // 0x800
    d["S_NF"] = (s & 0x1000) // 0x1000
    d["S_NI"] = (s & 0x2000) // 0x2000
    d["S_NI2"] = (s & 0x4000) // 0x4000
    return d


def _varian_find_dtype(dic):
    """Determine the real dtype from a file header dictionary."""
    if dic["S_FLOAT"] == 1:
        return np.dtype(">f4")
    if dic["S_32"] == 1:
        return np.dtype(">i4")
    return np.dtype(">i2")


def _varian_uninterleave_data(data):
    """Convert interleaved R,I data to complex."""
    rdt = data.dtype.name
    if rdt in ("int16", "float32"):
        cdt = "complex64"
    elif rdt == "int32":
        cdt = "complex128"
    else:
        cdt = data.dtype
    return data[..., ::2] + np.array(data[..., 1::2] * 1.0j, dtype=cdt)


def _varian_get_trace(f, pts, dt):
    """Read one trace of ``pts`` elements of dtype ``dt``."""
    return np.frombuffer(f.read(pts * dt.itemsize), dt)


def _varian_get_blockheader(f):
    """Read 28-byte block header."""
    return _struct.unpack(">4hl4f", f.read(28))


def _varian_skip_blockheader(f):
    """Skip one 28-byte block header."""
    f.read(28)


def _varian_get_block(f, pts, nbheaders, dt, read_blockhead=False):
    """Read a single block (skip headers, read trace)."""
    if not read_blockhead:
        for _ in range(nbheaders):
            _varian_skip_blockheader(f)
        return _varian_get_trace(f, pts, dt)
    for i in range(nbheaders):
        if i == 0:
            bh = _varian_get_blockheader(f)
        else:
            _varian_skip_blockheader(f)
    return bh, _varian_get_trace(f, pts, dt)


def _varian_get_nblocks(f, nblocks, pts, nbheaders, dt, read_blockhead):
    """Read multiple blocks (one trace per block)."""
    data = np.empty((nblocks, pts), dtype=dt)
    for i in range(nblocks):
        if read_blockhead:
            bh, block_data = _varian_get_block(f, pts, nbheaders, dt, True)
            data[i] = block_data
        else:
            data[i] = _varian_get_block(f, pts, nbheaders, dt, False)
    return data


def _varian_get_nblocks_ntraces(
    f, nblocks, ntraces, pts, nbheaders, dt, read_blockhead
):
    """Read blocks with multiple traces per block."""
    data = np.empty((nblocks * ntraces, pts), dtype=dt)
    for i in range(nblocks):
        _varian_skip_blockheader(f)
        if nbheaders >= 2:
            _varian_skip_blockheader(f)
        for _ in range(max(nbheaders - 2, 0)):
            _varian_skip_blockheader(f)
        block_data = _varian_get_trace(f, pts * ntraces, dt)
        data[i * ntraces : (i + 1) * ntraces] = block_data.reshape(ntraces, pts)
    return data


def _varian_reorder_data(data, shape, torder):
    """Reorder raw 2D data to NMR matrix shape for 3D+ data."""
    if torder in ("flat", "f"):
        try:
            return data.reshape(shape)
        except ValueError:
            return data

    ndata = np.empty(shape, dtype=data.dtype)
    indirect = shape[:-1]

    if torder in ("regular", "r"):
        i2t = index2trace_reg
    elif torder in ("opposite", "o"):
        i2t = index2trace_opp
    else:
        i2t = index2trace_flat

    for tup in np.ndindex(indirect):
        ntrace = i2t(indirect, tup)
        ndata[tup] = data[ntrace]
    return ndata


def _varian_get_parameter(f):
    """Read one procpar parameter from a file object."""
    d = {}
    line = f.readline().decode().split()
    d["name"] = line[0]
    d["subtype"] = line[1]
    d["basictype"] = line[2]
    d["maxvalue"] = line[3]
    d["minvalue"] = line[4]
    d["stepsize"] = line[5]
    d["Ggroup"] = line[6]
    d["Dgroup"] = line[7]
    d["protection"] = line[8]
    d["active"] = line[9]
    d["intptr"] = line[10]

    line = f.readline().decode()
    num = int(line.split()[0])
    values = []

    if d["basictype"] == "1":
        values = line.split()[1:]
    elif d["basictype"] == "2":
        values.append(line.split('"')[1])
        for _ in range(num - 1):
            values.append(f.readline().decode().split('"')[1])

    d["values"] = values
    line = f.readline().decode()
    d["enumerable"] = line.split()[0]
    if d["enumerable"] != "0":
        if d["basictype"] == "1":
            d["enumerables"] = line.split()[1:]
        elif d["basictype"] == "2":
            d["enumerables"] = line.split('"')[1::2]
    return d
