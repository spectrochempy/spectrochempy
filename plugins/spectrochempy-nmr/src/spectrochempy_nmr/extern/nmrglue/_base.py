# ======================================================================================
# Shared utilities for vendored nmrglue readers.
#
# Extracted from _bruker.py to avoid duplication when multiple readers
# (Bruker, JEOL, Varian, etc.) need the same low-level helpers.
#
# Original: https://github.com/jjhelmus/nmrglue
# License: see NMRGLUE_LICENSE.rst in this plugin directory.
# ======================================================================================

from __future__ import annotations

import itertools
import string
from functools import reduce

import numpy as np


def create_blank_udic(ndim):
    """Create a blank universal dictionary for a spectrum of dimension ndim."""
    udic = {}
    udic["ndim"] = ndim

    for i in range(ndim):
        d = {}
        d["sw"] = 999.99
        d["complex"] = True
        d["obs"] = 999.99
        d["car"] = 999.99
        d["size"] = 1
        d["label"] = ["X", "Y", "Z", "A"][i]

        if i == ndim - 1:
            d["encoding"] = "direct"
        else:
            d["encoding"] = "states"

        d["time"] = True
        d["freq"] = False

        udic[i] = d

    return udic


class unit_conversion:
    """
    Provides methods to convert between common NMR units.

    Parameters
    ----------
    size : int
        Number of points in dimension (R|I).
    cplex : bool
        True if dimension is complex, False is real.
    sw : float
        Spectral width in Hz.
    obs : float
        Observation frequency in MHz.
    car : float
        Carrier frequency in Hz.

    """

    def __init__(self, size, cplx, sw, obs, car):
        """Create and set up a unit_conversion object."""
        self._size = int(size)
        self._cplx = bool(cplx)
        self._sw = float(sw)
        self._obs = float(obs)
        self._car = float(car)

        self._delta = -self._sw / (self._size * self._obs)
        self._first = self._car / self._obs - self._delta * self._size / 2.0

    def __percent2pts(self, percent):
        return percent * (self._size - 1) / 100.0

    def __pts2percent(self, pts):
        return pts * 100 / (self._size - 1.0)

    def __hz2pts(self, hz):
        return ((hz / self._obs) - self._first) / self._delta

    def __pts2hz(self, pts):
        return (pts * self._delta + self._first) * self._obs

    def __ppm2pts(self, ppm):
        return (ppm - self._first) / self._delta

    def __pts2ppm(self, pts):
        return (pts * self._delta) + self._first

    def __sec2pts(self, sec):
        return sec * self._sw

    def __pts2sec(self, pts):
        return pts * 1.0 / self._sw

    def __ms2pts(self, ms):
        return ms * self._sw / 1.0e3

    def __pts2ms(self, pts):
        return pts * 1.0e3 / self._sw

    def __us2pts(self, us):
        return us * self._sw / 1.0e6

    def __pts2us(self, pts):
        return pts * 1.0e6 / self._sw

    def __unit2pnt(self, val, units):
        units = units.upper()
        if units == "PPM":
            pts = self.__ppm2pts(val)
        elif units == "HZ":
            pts = self.__hz2pts(val)
        elif units in ("%", "PERCENT"):
            pts = self.__percent2pts(val)
        elif units in ("SEC", "SECOND", "S"):
            pts = self.__sec2pts(val)
        elif units == "MS":
            pts = self.__ms2pts(val)
        elif units == "US":
            pts = self.__us2pts(val)
        else:
            raise ValueError("invalid unit type")
        return pts

    def __pnt2unit(self, val, units):
        units = units.upper()
        if units == "PPM":
            k = self.__pts2ppm(val)
        elif units == "HZ":
            k = self.__pts2hz(val)
        elif units in ("%", "PERCENT"):
            k = self.__pts2percent(val)
        elif units in ("SEC", "SECOND", "S"):
            k = self.__pts2sec(val)
        elif units == "MS":
            k = self.__pts2ms(val)
        elif units == "US":
            k = self.__pts2us(val)
        else:
            raise ValueError("invalid units")
        return k

    def __str2pnt(self, s):
        units = s.strip(string.digits + string.whitespace + "." + "-").upper()
        val = float(s.strip(string.ascii_letters + string.whitespace + "%"))
        return self.__unit2pnt(val, units)

    def __convert(self, val, unit=None):
        if isinstance(val, str):
            return self.__str2pnt(val)
        if unit is None:
            raise ValueError("invalid unit type")
        return self.__unit2pnt(val, unit)

    def f(self, val, unit=None):
        """Convert string or value/unit pair to float."""
        return self.__convert(val, unit)

    def i(self, val, unit=None):
        """Convert string or value/unit pair to integer."""
        return int(round(self.__convert(val, unit)))

    def ppm(self, val):
        """Convert to ppm."""
        return self.__pnt2unit(val, "PPM")

    def hz(self, val):
        """Convert to Hz."""
        return self.__pnt2unit(val, "HZ")

    def percent(self, val):
        """Convert to percent."""
        return self.__pnt2unit(val, "PERCENT")

    def seconds(self, val):
        """Convert to seconds."""
        return self.__pnt2unit(val, "SEC")

    def sec(self, val):
        """Convert to seconds."""
        return self.__pnt2unit(val, "SEC")

    def ms(self, val):
        """Convert to milliseconds (ms)."""
        return self.__pnt2unit(val, "MS")

    def us(self, val):
        """Convert to microseconds (us)."""
        return self.__pnt2unit(val, "US")

    def unit(self, val, unit):
        """Convert val points to unit."""
        return self.__pnt2unit(val, unit)

    def percent_limits(self):
        return 0.0, 100.0

    def percent_scale(self):
        return np.linspace(0.0, 100.0, self._size)

    def ppm_limits(self):
        return self.ppm(0), self.ppm(self._size - 1)

    def ppm_scale(self):
        x0, x1 = self.ppm_limits()
        return np.linspace(x0, x1, self._size)

    def hz_limits(self):
        return self.hz(0), self.hz(self._size - 1)

    def hz_scale(self):
        x0, x1 = self.hz_limits()
        return np.linspace(x0, x1, self._size)

    def sec_limits(self):
        return self.sec(0), self.sec(self._size - 1)

    def sec_scale(self):
        x0, x1 = self.sec_limits()
        return np.linspace(x0, x1, self._size)

    def ms_limits(self):
        return self.ms(0), self.ms(self._size - 1)

    def ms_scale(self):
        x0, x1 = self.ms_limits()
        return np.linspace(x0, x1, self._size)

    def us_limits(self):
        return self.us(0), self.us(self._size - 1)

    def us_scale(self):
        x0, x1 = self.us_limits()
        return np.linspace(x0, x1, self._size)

    __call__ = i


def uc_from_udic(udic, dim=-1):
    """
    Create a unit conversion object from a Universal dictionary.

    Parameters
    ----------
    udic : dic
        Universal dictionary of spectral parameters.
    dim : int. optional
        Dimension number to create unit conversion object for.  Default is for
        last dimension.

    Returns
    -------
    uc : unit conversion object.
        Unit conversion object for given dimension.

    """
    if dim == -1:
        dim = udic["ndim"] - 1
    adic = udic[dim]
    return unit_conversion(
        adic["size"], adic["complex"], adic["sw"], adic["obs"], adic["car"]
    )


def reorder_submatrix(data, shape, submatrix_shape, reverse=False):
    """
    Reorder processed binary data using submatrix reordering.

    Parameters
    ----------
    data : array
    shape : tuple
        Shape of final data.
    submatrix_shape : tuple
        Shape of submatrix.
    reverse : Bool
        True to reverse the reordering of a submatrix.

    Returns
    -------
    rdata : array
        Array in which data has been reordered and correctly shaped.

    """
    if submatrix_shape is None or shape is None:
        return data

    if len(submatrix_shape) == 1 or len(shape) == 1:
        return data

    sub_per_dim = [int(i / j) for i, j in zip(shape, submatrix_shape, strict=False)]
    nsubs = np.prod(sub_per_dim)

    if reverse:
        rdata = np.empty([nsubs] + list(submatrix_shape))
    else:
        data = data.reshape([nsubs] + list(submatrix_shape))
        rdata = np.empty(shape, dtype=data.dtype)

    for sub_num, sub_idx in enumerate(np.ndindex(tuple(sub_per_dim))):
        sub_slices = [
            slice(i * j, (i + 1) * j)
            for i, j in zip(sub_idx, submatrix_shape, strict=False)
        ]
        if reverse:
            rdata[sub_num] = data[tuple(sub_slices)]
        else:
            rdata[tuple(sub_slices)] = data[sub_num]
    return rdata.reshape(shape)


# -------------------------------------------------------------------
# Unit conversion from frequency scale
# -------------------------------------------------------------------


def uc_from_freqscale(scale, obs, unit="ppm"):
    """
    Create a unit conversion object from a spectrum frequency scale axis.

    Parameters
    ----------
    scale : array like
        array of spectrum axis
    obs : float
        Observation frequency in MHz.
    unit: {'ppm', 'hz', 'khz'}
        The unit of the scale axis.

    Returns
    -------
    uc : unit conversion object.
        Unit conversion object for given axis.

    """
    scale = np.array(scale)
    size = len(scale)

    if unit in ["ppm", "hz", "khz"]:
        cplx = False

        lo = scale.min()
        hi = scale.max()

        dx = abs(scale[1] - scale[0])

        if unit == "ppm":
            sw = ((hi + dx / 2.0) - (lo - dx / 2.0)) * obs
            car = (lo - dx / 2.0 + (hi - lo) / 2.0) * obs
        elif unit == "hz":
            sw = (hi + dx / 2.0) - (lo - dx / 2.0)
            car = lo - dx / 2.0 + (hi - lo) / 2.0
        else:
            # unit is 'kHz':
            sw = ((hi + dx / 2.0) - (lo - dx / 2.0)) / 1.0e3
            car = (lo - dx / 2.0 + (hi - lo) / 2.0) / 1.0e3

    else:
        mesg = f"{unit} is not a supported unit."
        raise ValueError(mesg)

    return unit_conversion(size, cplx, sw, obs, car)


# -------------------------------------------------------------------
# ND array index / trace converters (used by Bruker and Varian)
# -------------------------------------------------------------------


def ndfrom_iter(shape, slices):
    ch = [range(lenx)[sX] for lenx, sX in zip(shape, slices, strict=False)]
    return itertools.product(*ch)


def ndto_iter(shape, slices):
    ich = [range(len(range(lenx)[sX])) for lenx, sX in zip(shape, slices, strict=False)]
    return itertools.product(*ich)


def ndtofrom_iter(shape, slices):
    ch = [range(lenx)[sX] for lenx, sX in zip(shape, slices, strict=False)]
    ich = [range(len(i)) for i in ch]
    return zip(itertools.product(*ich), itertools.product(*ch), strict=False)


def size_and_ndtofrom_iter(shape, slices):
    ch = [range(lenx)[sX] for lenx, sX in zip(shape, slices, strict=False)]
    s = [len(i) for i in ch]
    ich = [range(i) for i in s]
    return s, zip(itertools.product(*ich), itertools.product(*ch), strict=False)


# -------------------------------------------------------------------
# index2trace and trace2index functions
# -------------------------------------------------------------------


def index2trace_flat(shape, index):
    """Calculate trace number from shape and index assuming a flat structure."""
    a = index[-1]
    for i, v in enumerate(index[:-1]):
        mult = reduce(lambda x, y: x * y, shape[i + 1 :])
        a = a + mult * v
    return a


def trace2index_flat(shape, ntrace):
    """Calculate the index of a trace assuming a flat structure."""
    q = ntrace
    index = []
    for s in shape[:0:-1]:
        q, r = divmod(q, s)
        index.insert(0, r)
    index.insert(0, q)
    return tuple(index)


def index2trace_opp(shape, index):
    """Calculate trace number assuming phase ordering opposite the time increments."""
    n = len(shape)
    phases = [v % 2 for v in index]
    nphase = index2trace_flat([2] * n, phases[::-1])
    pindex = [v // 2 for v in index]
    pshape = [i // 2 for i in shape]
    nbase = index2trace_flat(pshape, pindex)
    return nbase * 2**n + nphase


def trace2index_opp(shape, ntrace):
    """Calculate the index of a trace assuming opposite phase/time increment ordering."""
    n = len(shape)
    q, r = divmod(ntrace, 2**n)
    to_add = list(trace2index_flat([2] * n, r))[::-1]
    pshape = [i // 2 for i in shape]
    base = list(trace2index_flat(pshape, q))
    total = [b * 2 + a for b, a in zip(base, to_add, strict=False)]
    return tuple(total)


def index2trace_reg(shape, index):
    """Calculate trace number assuming the same phase and time ordering."""
    n = len(shape)
    phases = [v % 2 for v in index]
    nphase = index2trace_flat([2] * n, phases)
    pindex = [v // 2 for v in index]
    pshape = [i // 2 for i in shape]
    nbase = index2trace_flat(pshape, pindex)
    return nbase * 2**n + nphase


def trace2index_reg(shape, ntrace):
    """Calculate the index of a trace assuming the same phase/time increment ordering."""
    n = len(shape)
    q, r = divmod(ntrace, 2**n)
    to_add = list(trace2index_flat([2] * n, r))
    pshape = [i // 2 for i in shape]
    base = list(trace2index_flat(pshape, q))
    total = [b * 2 + a for b, a in zip(base, to_add, strict=False)]
    return tuple(total)


# -------------------------------------------------------------------
# Data packing helpers
# -------------------------------------------------------------------


def complexify_data(data):
    """Complexify data packed real, imag."""
    return data[..., ::2] + data[..., 1::2] * 1.0j


def uncomplexify_data(data_in, isfloat):
    """Uncomplexify data (pack real,imag) into a int32 or float64 array."""
    size = list(data_in.shape)
    size[-1] = size[-1] * 2
    if isfloat:
        data_out = np.empty(size, dtype="float64")
    else:
        data_out = np.empty(size, dtype="int32")
    data_out[..., ::2] = data_in.real
    data_out[..., 1::2] = data_in.imag
    return data_out
