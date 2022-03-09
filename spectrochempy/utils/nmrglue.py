"""
Functions for reading and writing Bruker binary (ser/fid) files, Bruker
JCAMP-DX parameter (acqus) files, and Bruker pulse program (pulseprogram)
files.

Copied and adapted from NMRGLUE (See NMRGLUE_LICENCES in root folder LICENSES)
"""
import locale
import io

__developer_info__ = """
Bruker file format information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Bruker binary files (ser/fid) store data as an array of numbers whose
endianness is determined by the parameter BYTORDA (1 = big endian, 0 = little
endian), and whose data type is determined by the parameter DTYPA (0 = int32,
2 = float64). Typically the direct dimension is digitally filtered. The exact
method of removing this filter is unknown but an approximation is available.

Bruker JCAMP-DX files (acqus, etc) are text file which are described by the
`JCAMP-DX standard <http://www.jcamp-dx.org/>`_.  Bruker parameters are
prefixed with a '$'.

Bruker pulseprogram files are text files described in various Bruker manuals.
Of special important are lines which describe external variable assignments
(surrounded by "'s), loops (begin with lo), phases (contain ip of dp) or
increments (contain id, dd, ipu or dpu).  These lines are parsed when reading
the file with nmrglue.

"""

from functools import reduce
import os
from warnings import warn
import string
import itertools

import numpy as np


def create_blank_udic(ndim):
    """
    Create a blank universal dictionary for a spectrum of dimension ndim.
    """
    udic = dict()
    udic["ndim"] = ndim

    for i in range(ndim):
        d = dict()
        d["sw"] = 999.99  # spectral width in Hz
        d["complex"] = True  # Quadrature, True when dimension is complex
        d["obs"] = 999.99  # Observation frequency in MHz
        d["car"] = 999.99  # Carrier frequency in Hz
        # Number of points in dimension based on the shape of the data array.
        # As such the direct dimension (-1) size is R|I, all indirect
        # dimensions are R+I
        d["size"] = 1

        d["label"] = ["X", "Y", "Z", "A"][i]  # name of dimension

        # encoding of dimension, ie states, tppi, etc.  The direct dimension
        # should be listed as direct.
        if i == ndim - 1:
            d["encoding"] = "direct"
        else:
            d["encoding"] = "states"

        # time and freq flags for domain of dimension
        d["time"] = True
        d["freq"] = False

        udic[i] = d

    return udic


class unit_conversion:
    """
    Provides methods to convert between common NMR units

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
        """
        create and set up a unit_conversion object
        """
        # fundamental units
        self._size = int(size)
        self._cplx = bool(cplx)
        self._sw = float(sw)
        self._obs = float(obs)
        self._car = float(car)

        # derived units (these are in ppm)
        self._delta = -self._sw / (self._size * self._obs)
        self._first = self._car / self._obs - self._delta * self._size / 2.0

    # individual unit conversion functions
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

    # times based units: seconds, ms, and us
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

    # routers
    def __unit2pnt(self, val, units):
        """
        Convert units to points
        """
        units = units.upper()
        if units == "PPM":
            pts = self.__ppm2pts(val)
        elif units == "HZ":
            pts = self.__hz2pts(val)
        elif units == "%" or units == "PERCENT":
            pts = self.__percent2pts(val)
        elif units == "SEC" or units == "SECOND" or units == "S":
            pts = self.__sec2pts(val)
        elif units == "MS":
            pts = self.__ms2pts(val)
        elif units == "US":
            pts = self.__us2pts(val)
        else:
            raise ValueError("invalid unit type")
        # if self._cplx:
        #     return pts+round(pts)
        # else:
        return pts

    def __pnt2unit(self, val, units):
        """
        Convert points to units
        """
        units = units.upper()
        # if self._cplx:
        #     val = val-round(val)
        if units == "PPM":
            k = self.__pts2ppm(val)
        elif units == "HZ":
            k = self.__pts2hz(val)
        elif units == "%" or units == "PERCENT":
            k = self.__pts2percent(val)
        elif units == "SEC" or units == "SECOND" or units == "S":
            k = self.__pts2sec(val)
        elif units == "MS":
            k = self.__pts2ms(val)
        elif units == "US":
            k = self.__pts2us(val)
        else:
            raise ValueError("invalid units")
        return k

    def __str2pnt(self, s):
        """
        Convert string with units to points
        """
        units = s.strip(string.digits + string.whitespace + "." + "-").upper()
        val = float(s.strip(string.ascii_letters + string.whitespace + "%"))
        return self.__unit2pnt(val, units)

    def __convert(self, val, unit=None):
        """
        Convert string or value/unit pair
        """
        if isinstance(val, str):
            return self.__str2pnt(val)
        else:
            if unit is None:
                raise ValueError("invalid unit type")
            return self.__unit2pnt(val, unit)

    # User functions
    def f(self, val, unit=None):
        """
        Convert string or value/unit pair to float
        """
        return self.__convert(val, unit)

    def i(self, val, unit=None):
        """
        Convert string or value/unit pair to integer
        """
        return int(round(self.__convert(val, unit)))

    def ppm(self, val):
        """
        Convert to ppm
        """
        return self.__pnt2unit(val, "PPM")

    def hz(self, val):
        """
        Convert to Hz
        """
        return self.__pnt2unit(val, "HZ")

    def percent(self, val):
        """
        Convert to percent
        """
        return self.__pnt2unit(val, "PERCENT")

    def seconds(self, val):
        """
        Convert to seconds
        """
        return self.__pnt2unit(val, "SEC")

    def sec(self, val):
        """
        Convert to seconds
        """
        return self.__pnt2unit(val, "SEC")

    def ms(self, val):
        """
        Convert to milliseconds (ms)
        """
        return self.__pnt2unit(val, "MS")

    def us(self, val):
        """
        Convert to microseconds (us)
        """
        return self.__pnt2unit(val, "US")

    def unit(self, val, unit):
        """
        Convert val points to unit
        """
        return self.__pnt2unit(val, unit)

    # limits and scales
    def percent_limits(self):
        """
        Return tuple of left and right edges in percent
        """
        return 0.0, 100.0

    def percent_scale(self):
        """
        Return array of percent values
        """
        return np.linspace(0.0, 100.0, self._size)

    def ppm_limits(self):
        """
        Return tuple of left and right edges in ppm
        """
        return self.ppm(0), self.ppm(self._size - 1)

    def ppm_scale(self):
        """
        Return array of ppm values
        """
        x0, x1 = self.ppm_limits()
        return np.linspace(x0, x1, self._size)

    def hz_limits(self):
        """
        Return tuple of left and right edges in Hz
        """
        return self.hz(0), self.hz(self._size - 1)

    def hz_scale(self):
        """
        Return array of Hz values
        """
        x0, x1 = self.hz_limits()
        return np.linspace(x0, x1, self._size)

    def sec_limits(self):
        """
        Return tuple of left and right edges in seconds
        """
        return self.sec(0), self.sec(self._size - 1)

    def sec_scale(self):
        """
        Return array of seconds values
        """
        x0, x1 = self.sec_limits()
        return np.linspace(x0, x1, self._size)

    def ms_limits(self):
        """
        Return tuple of left and right edges in milliseconds
        """
        return self.ms(0), self.ms(self._size - 1)

    def ms_scale(self):
        """
        Return array of seconds values
        """
        x0, x1 = self.ms_limits()
        return np.linspace(x0, x1, self._size)

    def us_limits(self):
        """
        Return tuple of left and right edges in milliseconds
        """
        return self.us(0), self.us(self._size - 1)

    def us_scale(self):
        """
        Return array of seconds values
        """
        x0, x1 = self.us_limits()
        return np.linspace(x0, x1, self._size)

    __call__ = i  # calling the object x is the same as x.i


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
        dim = udic["ndim"] - 1  # last dimension
    adic = udic[dim]
    return unit_conversion(
        adic["size"], adic["complex"], adic["sw"], adic["obs"], adic["car"]
    )


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
        complex = False

        min = scale.min()
        max = scale.max()

        # The scale needs be corrected by extending each extremum by half the
        # bin width (to convert from centers to edges).
        dx = abs(scale[1] - scale[0])

        if unit == "ppm":
            sw = ((max + dx / 2.0) - (min - dx / 2.0)) * obs
            car = (min - dx / 2.0 + (max - min) / 2.0) * obs
        elif unit == "hz":
            sw = (max + dx / 2.0) - (min - dx / 2.0)
            car = min - dx / 2.0 + (max - min) / 2.0
        else:
            # unit is 'kHz':
            sw = ((max + dx / 2.0) - (min - dx / 2.0)) / 1.0e3
            car = (min - dx / 2.0 + (max - min) / 2.0) / 1.0e3

    else:
        mesg = "{} is not a supported unit.".format(unit)
        raise ValueError(mesg)

    return unit_conversion(size, complex, sw, obs, car)


def open_towrite(filename, overwrite=False, mode="wb"):
    """
    Open filename for writing and return file object

    Function checks if file exists (and raises IOError if overwrite=False) and
    creates necessary directories as needed.
    """
    # check if file exists and overwrite if False
    if os.path.exists(filename) and (overwrite is False):
        raise IOError("File exists, recall with overwrite=True")

    p, fn = os.path.split(filename)  # split into filename and path
    # create directories if needed
    if p != "" and os.path.exists(p) is False:
        os.makedirs(p)

    return open(filename, mode)


################################################
# numpy ndarray emulation and helper functions #
################################################

# iterators for ND array


def ndfrom_iter(shape, slices):
    ch = [range(lenx)[sX] for lenx, sX in zip(shape, slices)]
    return itertools.product(*ch)


def ndto_iter(shape, slices):
    ich = [range(len(range(lenx)[sX])) for lenx, sX in zip(shape, slices)]
    return itertools.product(*ich)


def ndtofrom_iter(shape, slices):
    ch = [range(lenx)[sX] for lenx, sX in zip(shape, slices)]
    ich = [range(len(i)) for i in ch]
    return zip(itertools.product(*ich), itertools.product(*ch))


def size_and_ndtofrom_iter(shape, slices):
    ch = [range(lenx)[sX] for lenx, sX in zip(shape, slices)]
    s = [len(i) for i in ch]
    ich = [range(i) for i in s]
    return s, zip(itertools.product(*ich), itertools.product(*ch))


# index2trace and trace2index functions


def index2trace_flat(shape, index):
    """
    Calculate trace number from shape and index of all indirect dimensions
    assuming a flat structure
    """
    # We need to perform:
    # index[0]*shape[1]*...shape[-1] + index[1]*shape[2]*...shape[-1] + ...
    # + index[-1]*shape[-1] + index[-1]
    # To do this we calculate the product of shape[X] elements and multiple
    # by the corresponding index element, index[-1] as added at the beginning
    a = index[-1]
    for i, v in enumerate(index[:-1]):
        mult = reduce(lambda x, y: x * y, shape[i + 1 :])
        a = a + mult * v
    return a


def trace2index_flat(shape, ntrace):
    """
    Calculate the index of a trace assuming a flat structure
    """
    # algorithm is to take quotient/remainers of sizes in reverse
    q = ntrace  # seed quotient with remained
    index = []
    for s in shape[:0:-1]:  # loop from last size to 2nd size
        q, r = divmod(q, s)
        index.insert(0, r)
    index.insert(0, q)
    return tuple(index)


def index2trace_opp(shape, index):
    """
    Calculate trace number from shape and index of all indirect dimensions
    assuming a phase ordering opposite the time increments.
    """
    n = len(shape)
    # deal with the phase component
    phases = [v % 2 for v in index]
    nphase = index2trace_flat([2] * n, phases[::-1])
    # deal with the remainder
    pindex = [v // 2 for v in index]
    pshape = [i // 2 for i in shape]
    nbase = index2trace_flat(pshape, pindex)
    return nbase * 2 ** n + nphase


def trace2index_opp(shape, ntrace):
    """
    Calculate the index of a trace assuming opposite phase/time increment
    ordering
    """
    n = len(shape)
    q, r = divmod(ntrace, 2 ** n)
    to_add = list(trace2index_flat([2] * n, r))[::-1]
    pshape = [i // 2 for i in shape]
    base = list(trace2index_flat(pshape, q))
    total = [b * 2 + a for b, a in zip(base, to_add)]
    return tuple(total)


def index2trace_reg(shape, index):
    """
    Calculate trace number from shape and index of all indirect dimensions
    assuming the same  phase and time ordering.
    """
    n = len(shape)
    # deal with the phase component
    phases = [v % 2 for v in index]
    nphase = index2trace_flat([2] * n, phases)
    # deal with the remainder
    pindex = [v // 2 for v in index]
    pshape = [i // 2 for i in shape]
    nbase = index2trace_flat(pshape, pindex)
    return nbase * 2 ** n + nphase


def trace2index_reg(shape, ntrace):
    """
    Calculate the index of a trace assuming the same phase/time increment
    ordering
    """
    n = len(shape)
    q, r = divmod(ntrace, 2 ** n)
    to_add = list(trace2index_flat([2] * n, r))
    pshape = [i // 2 for i in shape]
    base = list(trace2index_flat(pshape, q))
    total = [b * 2 + a for b, a in zip(base, to_add)]
    return tuple(total)


def fft_positive(data):
    """
    Fourier transform with positive exponential, NMR ordering of results

    This is similar to the transform performed by NMRPipe's FFT function.

    See :py:func:`fft` for documentation of the transformation applied by this
    function.

    Parameters
    ----------
    data : ndarray
        Array of NMR data.

    Returns
    -------
    ndata : ndarray
        Fourier transform of NMR data in 'NMR order'.

    See Also
    --------
    ifft_positive : Inversion Fourier transform.
    fft_norm : Norm (power) conserving Fourier transform.
    fft_positive : Forier transform with a positive exponential.

    """
    # a positive exponential is the same as a IFFT, but we need to undo
    # the 1/N scaling
    s = float(data.shape[-1])
    return np.fft.fftshift(np.fft.ifft(data, axis=-1).astype(data.dtype), -1) * s


def ifft_positive(data):
    """
    Inverse fourier transform with positive exponential, NMR ordered results.

    This is similar to the transform performed by NMRPipe's FFT function with
    the -inv flag

    Parameters
    ----------
    data : ndarray
        Array of NMR data.

    Returns
    -------
    ndata : ndarray
        Inverse fourier transform of NMR data in 'NMR order'.

    See Also
    --------
    fft_postive : Fourier transform with a positive exponential.
    ifft_norm : Norm (power) conserving inverse Fourier transform.
    ifft : Inverse Forier transform with a negative exponential.

    """
    # a inverse fft with positive exponential in the FFT definition is the
    # same as a FFT with negative exponentials, but with a 1/N scaling factor
    s = 1.0 / float(data.shape[-1])
    return np.fft.fft(np.fft.ifftshift(data, -1), axis=-1).astype(data.dtype) * s


def fsh2(data, pts):
    """
    Frequency Shift by Fourier transform. Positive signed phase correction.

    Parameters
    ----------
    data : ndarray
        Array of NMR data.
    pts : float
        Number of points to frequency shift the data.  Positive value will
        shift the spectrum to the right, negative values to the left.

    Returns
    -------
    ndata : ndarray
        Array of NMR data with last axis rolled.

    """
    s = float(data.shape[-1])
    return fft_positive(
        np.exp(2.0j * np.pi * pts * np.arange(s) / s).astype(data.dtype)
        * ifft_positive(data)
    )


# data creation


def create_data(data):
    """
    Create a bruker data array (recast into a complex128 or int32)
    """
    if np.iscomplexobj(data):
        return np.array(data, dtype="complex128")
    else:
        return np.array(data, dtype="int32")


# universal dictionary functions


def guess_udic(dic, data, strip_fake=False):
    """
    Guess parameters of universal dictionary from dic, data pair.

    Parameters
    ----------
    dic : dict
        Dictionary of Bruker parameters.
    data : ndarray
        Array of NMR data.
    strip_fake: bool
        If data is proceed (i.e. read using `bruker.read_pdata`) and the Bruker
        processing parameters STSI and/or STSR are set, the returned sweep
        width and carrier frequencies is changed to values that are incorrect
        but instead can are intended to trick the normal unit_conversion object
        into producing the correct result.

    Returns
    -------
    udic : dict
        Universal dictionary of spectral parameters.

    """
    # TODO if pprog, acqus, procs are in dic use them better.

    # create an empty universal dictionary
    udic = create_blank_udic(data.ndim)

    # update default values
    for b_dim in range(data.ndim):
        udic[b_dim]["size"] = data.shape[b_dim]

        # try to add additional parameter from acqus dictionary keys
        try:
            add_axis_to_udic(udic, dic, b_dim, strip_fake)
        except Exception:
            warn("Failed to determine udic parameters for dim: %i" % (b_dim))
    return udic


def add_axis_to_udic(udic, dic, udim, strip_fake):
    """
    Add axis parameters to a udic.

    Parameters
    ----------
    udic : dict
        Universal dictionary to update, modified in place.
    dic : dict
        Bruker dictionary used to determine axes parameters.
    dim : int
        Universal dictionary dimension to update.
    strip_fake: bool
        See `bruker.guess_udic`

    """
    # This could still use some work
    b_dim = udic["ndim"] - udim - 1  # last dim
    acq_file = "acqu" + str(b_dim + 1) + "s"
    pro_file = "proc" + str(b_dim + 1) + "s"

    # Because they're inconsistent,..
    if acq_file == "acqu1s":
        acq_file = "acqus"

    if pro_file == "proc1s":
        pro_file = "procs"

    if acq_file in dic:
        if b_dim == 0:
            sw = dic[acq_file]["SW_h"]
        else:
            sw = dic[acq_file]["SW"] * dic[acq_file]["SFO1"]
    elif pro_file in dic:
        sw = dic[pro_file]["SW_p"]
        # procNs files store sw (in Hz) with the 'SW_p' key instead of 'SW_h'.
        # this is a bug in TopSpin (TopSpin3.5pl7)

    if acq_file in dic:
        udic[udim]["label"] = dic[acq_file]["NUC1"]
    elif pro_file in dic:
        udic[udim]["label"] = dic[pro_file]["AXNUC"]

    try:
        obs = dic[pro_file]["SF"]
        if acq_file in dic:
            car = (dic[acq_file]["SFO1"] - obs) * 1e6
        else:
            # we should be able to use the 'OFFSET' parameter in procNs to
            # calculate 'car'. But this is slightly off (~ 5E-3 Hz)
            # most likely because the procs file does not store the OFFSET
            # to a high precision. Hence the value in acquNs is given priority
            car = dic[pro_file]["OFFSET"] * obs - sw / 2

    except KeyError:
        warn('The chemical shift referencing was not corrected for "sr".')
        obs = dic[acq_file]["SFO1"]
        car = dic[acq_file]["O1"]

    if strip_fake:
        try:
            # Temporary parameters
            w = sw / float(dic[pro_file]["FTSIZE"])
            d = (w * dic[pro_file]["STSR"]) + (w * dic[pro_file]["STSI"] / 2.0)

            # Fake car frequency
            car -= d - (sw / 2.0)

            # Fake sw frequency
            sw = w * dic[pro_file]["STSI"]
        except KeyError:
            pass

    udic[udim]["sw"] = sw
    udic[udim]["car"] = car
    udic[udim]["obs"] = obs

    if acq_file in dic:
        if acq_file == "acqus":
            if dic["acqus"]["AQ_mod"] == 0:  # qf
                udic[udim]["complex"] = False
            else:
                udic[udim]["complex"] = True
        else:
            aq_mod = dic[acq_file]["FnMODE"]
            if aq_mod == 0:
                udic[udim]["encoding"] = "undefined"
            elif aq_mod == 1:
                udic[udim]["encoding"] = "magnitude"  # qf
            elif aq_mod == 2:
                udic[udim]["encoding"] = "magnitude"  # qsec
            elif aq_mod == 3:
                udic[udim]["encoding"] = "tppi"
            elif aq_mod == 4:
                udic[udim]["encoding"] = "states"
            elif aq_mod == 5:
                udic[udim]["encoding"] = "states-tppi"  # states-tppi
            elif aq_mod == 6:
                udic[udim]["encoding"] = "echo-antiecho"  # echo-antiecho
    else:
        if pro_file == "procs":
            # this seems to have the 'MC2' parameter always set to 0
            # irrespective of what the actual data is
            udic[udim]["complex"] = "undefined"
        else:
            # these are only used when params in acquNs are 'undefined'
            # but in absence of acqus, this is the best that can be done
            aq_mod = dic[pro_file]["MC2"]
            if aq_mod == 0:
                udic[udim]["encoding"] = "magnitude"  # qf
            elif aq_mod == 1:
                udic[udim]["encoding"] = "magnitude"  # qsec
            elif aq_mod == 2:
                udic[udim]["encoding"] = "tppi"
            elif aq_mod == 3:
                udic[udim]["encoding"] = "states"
            elif aq_mod == 4:
                udic[udim]["encoding"] = "states-tppi"
            elif aq_mod == 5:
                udic[udim]["encoding"] = "echo-antiecho"

    return udic


def create_dic(udic):
    """
    Create a Bruker parameter dictionary from a universal dictionary.

    Parameters
    ----------
    udic : dict
        Universal dictionary of spectral parameters.

    Returns
    -------
    dic : dict
        Dictionary of Bruker parameters.

    """
    ndim = udic["ndim"]

    # determine the size in bytes
    if udic[ndim - 1]["complex"]:
        bytes = 8
    else:
        bytes = 4

    for k in range(ndim):
        bytes *= udic[k]["size"]

    dic = {"FILE_SIZE": bytes}

    # create the pprog dictionary parameter
    dic["pprog"] = {
        "incr": [[], [1]] * (ndim * 2 - 2),
        "loop": [2] * (ndim * 2 - 2),
        "ph_extra": [[]] * (ndim * 2 - 2),
        "phase": [[]] * (ndim * 2 - 2),
        "var": {},
    }

    # create acqus dictionary parameters and fill in loop sizes
    dic["acqus"] = create_acqus_dic(udic[ndim - 1], direct=True)
    if ndim >= 2:
        dic["acqu2s"] = create_acqus_dic(udic[ndim - 2])
        dic["pprog"]["loop"][1] = udic[ndim - 2]["size"] // 2
    if ndim >= 3:
        dic["acqu3s"] = create_acqus_dic(udic[ndim - 3])
        dic["pprog"]["loop"][3] = udic[ndim - 3]["size"] // 2
    if ndim >= 4:
        dic["acqu4s"] = create_acqus_dic(udic[ndim - 4])
        dic["pprog"]["loop"][5] = udic[ndim - 4]["size"] // 2

    return dic


def create_acqus_dic(adic, direct=False):
    """
    Create a Bruker acqus dictionary from an Universal axis dictionary.
    Set direct=True for direct dimension.
    """
    if adic["complex"]:
        AQ_mod = 3
        if direct:
            TD = int(np.ceil(adic["size"] / 256.0) * 256) * 2
        else:
            TD = adic["size"]
    else:
        AQ_mod = 1
        if direct:
            TD = int(np.ceil(adic["size"] / 256.0) * 256)
        else:
            TD = adic["size"]

    s = "##NMRGLUE automatically created parameter file"
    return {"_comments": [], "_coreheader": [s], "AQ_mod": AQ_mod, "TD": TD}


# Global read/write function and related utilities


def read_fid(
    dir=".",
    bin_file=None,
    acqus_files=None,
    pprog_file=None,
    shape=None,
    cplex=None,
    big=None,
    isfloat=None,
    read_pulseprogram=True,
    read_acqus=True,
    procs_files=None,
    read_procs=True,
):
    """
    Read Bruker files from a directory.

    Parameters
    ----------
    dir : str
        Directory to read from.
    bin_file : str, optional
        Filename of binary file in directory. None uses standard files.
    acqus_files : list, optional
        List of filename(s) of acqus parameter files in directory. None uses
        standard files.
    pprog_file : str, optional
        Filename of pulse program in directory. None uses standard files.
    shape : tuple, optional
        Shape of resulting data.  None will guess the shape from the spectral
        parameters.
    cplex : bool, optional
        True is direct dimension is complex, False otherwise. None will guess
        quadrature from spectral parameters.
    big : bool or None, optional
        Endianness of binary file. True for big-endian, False for
        little-endian, None to determine endianness from acqus file(s).
    isfloat : bool or None, optional
        Data type of binary file. True for float64, False for int32. None to
        determine data type from acqus file(s).
    read_pulseprogram : bool, optional
        True to read pulse program, False prevents reading.
    read_acqus : bool, optional
        True to read acqus files(s), False prevents reading.
    procs_files : list, optional
        List of filename(s) of procs parameter files in directory. None uses
        standard files.
    read_procs : bool, optional
        True to read procs files(s), False prevents reading.

    Returns
    -------
    dic : dict
        Dictionary of Bruker parameters.
    data : ndarray
        Array of NMR data.

    See Also
    --------
    read_pdata : Read Bruker processed files.
    read_lowmem : Low memory reading of Bruker files.
    write : Write Bruker files.

    """
    if os.path.isdir(dir) is not True:
        raise IOError("directory %s does not exist" % (dir))

    # Take a shot at reading the procs file
    if read_procs:
        dic = read_procs_file(dir, procs_files)
    else:
        # create an empty dictionary
        dic = dict()

    # determine parameter automatically
    if bin_file is None:
        if os.path.isfile(os.path.join(dir, "fid")):
            bin_file = "fid"
        elif os.path.isfile(os.path.join(dir, "ser")):
            bin_file = "ser"

        # Look two directory levels lower.
        elif os.path.isdir(os.path.dirname(os.path.dirname(dir))):

            # ! change the dir
            dir = os.path.dirname(os.path.dirname(dir))

            if os.path.isfile(os.path.join(dir, "fid")):
                bin_file = "fid"
            elif os.path.isfile(os.path.join(dir, "ser")):
                bin_file = "ser"
            else:
                mesg = "No Bruker binary file could be found in %s"
                raise IOError(mesg % (dir))
        else:
            mesg = "No Bruker binary file could be found in %s"
            raise IOError(mesg % (dir))

    if read_acqus:
        # read the acqus_files and add to the dictionary
        acqus_dic = read_acqus_file(dir, acqus_files)
        dic = _merge_dict(dic, acqus_dic)

    if pprog_file is None:
        pprog_file = "pulseprogram"

    # read the pulse program and add to the dictionary
    if read_pulseprogram:
        try:
            dic["pprog"] = read_pprog(os.path.join(dir, pprog_file))
        except Exception:
            warn("Error reading the pulse program")

    # determine file size and add to the dictionary
    dic["FILE_SIZE"] = os.stat(os.path.join(dir, bin_file)).st_size

    # determine shape and complexity for direct dim if needed
    if shape is None or cplex is None:
        gshape, gcplex = guess_shape(dic)
        if gcplex is True:  # divide last dim by 2 if complex
            t = list(gshape)
            t[-1] = t[-1] // 2
            gshape = tuple(t)
    if shape is None:
        shape = gshape
    if cplex is None:
        cplex = gcplex

    # determine endianness (assume little-endian unless BYTORDA is 1)
    if big is None:
        big = False  # default value
        if "acqus" in dic and "BYTORDA" in dic["acqus"]:
            if dic["acqus"]["BYTORDA"] == 1:
                big = True
            else:
                big = False

    # determine data type (assume int32 unless DTYPA is 2)
    if isfloat is None:
        isfloat = False  # default value
        if "acqus" in dic and "DTYPA" in dic["acqus"]:
            if dic["acqus"]["DTYPA"] == 2:
                isfloat = True
            else:
                isfloat = False

    # read the binary file
    f = os.path.join(dir, bin_file)
    null, data = read_binary(f, shape=shape, cplex=cplex, big=big, isfloat=isfloat)
    return dic, data


def read_acqus_file(dir=".", acqus_files=None):
    """
    Read Bruker acquisition files from a directory.

    Parameters
    ----------
    dir : str
        Directory to read from.
    acqus_files : list, optional
        List of filename(s) of acqus parameter files in directory. None uses
        standard files. If filename(s) contains a full absolute path, dir is not used.

    Returns
    -------
    dic : dict
        Dictionary of Bruker parameters.
    """
    if acqus_files is None:
        acqus_files = []
        for f in ["acqus", "acqu2s", "acqu3s", "acqu4s"]:
            fp = os.path.join(dir, f)
            if os.path.isfile(fp):
                acqus_files.append(fp)

    # create an empty dictionary
    dic = dict()

    # read the acqus_files and add to the dictionary
    for f in acqus_files:
        if not os.path.isfile(f):
            f = os.path.join(dir, f)
        acqu = os.path.basename(f)
        dic[acqu] = read_jcamp(f)

    return dic


def read_procs_file(dir=".", procs_files=None):
    """
    Read Bruker processing files from a directory.

    Parameters
    ----------
    dir : str
        Directory to read from.
    procs_files : list, optional
        List of filename(s) of procs parameter files in directory. None uses
        standard files. If filename(s) contains a full absolute path, dir is not used.

    Returns
    -------
    dic : dict
        Dictionary of Bruker parameters.
    """

    if procs_files is None:

        # Reading standard procs files
        procs_files = []

        pdata_path = dir
        for f in ["procs", "proc2s", "proc3s", "proc4s"]:
            pf = os.path.join(pdata_path, f)
            if os.path.isfile(pf):
                procs_files.append(pf)

        if not procs_files:
            # procs not found in the given dir, try look adding pdata to the dir path

            if os.path.isdir(os.path.join(dir, "pdata")):
                pdata_folders = [
                    folder for folder in os.walk(os.path.join(dir, "pdata"))
                ][0][1]
                if "1" in pdata_folders:
                    pdata_path = os.path.join(dir, "pdata", "1")
                else:
                    pdata_path = os.path.join(dir, "pdata", pdata_folders[0])

            for f in ["procs", "proc2s", "proc3s", "proc4s"]:
                pf = os.path.join(pdata_path, f)
                if os.path.isfile(pf):
                    procs_files.append(pf)

    else:
        # proc paths were explicitly given
        # just check if they exists

        for i, f in enumerate(procs_files):
            pdata_path, f = os.path.split(f)
            if not pdata_path:
                pdata_path = dir

            pf = os.path.join(pdata_path, f)
            if not os.path.isfile(pf):
                mesg = "The file `%s` could not be found "
                warn(mesg % pf)
            else:
                procs_files[i] = pf

    # create an empty dictionary
    dic = dict()

    # read the acqus_files and add to the dictionary
    for f in procs_files:
        pdata_path = os.path.basename(f)
        dic[pdata_path] = read_jcamp(f)
    return dic


def guess_shape(dic):
    """
    Determine data shape and complexity from Bruker dictionary.

    Returns
    -------
    shape : tuple
        Shape of data in Bruker binary file (R+I for all dimensions).
    cplex : bool
        True for complex data in last (direct) dimension, False otherwise.

    """
    # determine complexity of last (direct) dimension
    try:
        aq_mod = dic["acqus"]["AQ_mod"]
    except KeyError:
        aq_mod = 0

    if aq_mod == 0 or aq_mod == 2:
        cplex = False
    elif aq_mod == 1 or aq_mod == 3:
        cplex = True
    else:
        raise ValueError("Unknown Acquisition Mode")

    # file size
    try:
        fsize = dic["FILE_SIZE"]
    except KeyError:
        warn("cannot determine shape do to missing FILE_SIZE key")
        return (1,), True

    # extract td0,td1,td2,td3 from dictionaries
    try:
        td0 = float(dic["acqus"]["TD"])
    except KeyError:
        td0 = 1024  # default value

    try:
        td2 = int(dic["acqu2s"]["TD"])
    except KeyError:
        td2 = 0  # default value

    try:
        td1 = float(dic["acqu3s"]["TD"])
    except KeyError:
        td1 = int(td2)  # default value

    try:
        td3 = int(dic["acqu4s"]["TD"])
    except KeyError:
        td3 = int(td1)  # default value

    # From the acquisition reference manual (section on parameter NBL):
    #     ---
    #     If TD is not a multiple of 256 (1024 bytes), successive FIDs will
    #     still begin at 1024 byte memory boundaries. This is so for the FIDs
    #     in the acquisition memory as well as on disk. The size of the raw
    #     data file (ser) is therefore always a multiple of 1024 times NBL.
    #     ---
    # This seems to hold for 1D data sets as well. However, this paragraph
    # assumes that each data point is 4 bytes, hence the "multiple of 256".
    # For data in DTYPA=2 (float64), each point is 8 bytes, so while it always
    # allocates the fids in 1024-byte blocks, for float64 data it pads the data
    # (by points) out to multiples of 128, not 256. So we need to get the
    # data type before we guess the shape of the last dimension.

    # extract data type from dictionary
    try:
        dtypa = int(dic["acqus"]["DTYPA"])
    except KeyError:
        dtypa = 0  # default value, int32 data

    # last (direct) dimension is given by "TD" parameter in acqus file
    # rounded up to nearest (1024/(bytes per point))
    # next-to-last dimension may be given by "TD" in acqu2s. In 3D+ data
    # this is often the sum of the indirect dimensions
    if dtypa == 2:
        shape = [0, 0, td2, int(np.ceil(td0 / 128.0) * 128.0)]
    else:
        shape = [0, 0, td2, int(np.ceil(td0 / 256.0) * 256.0)]

    # additional dimension given by data size
    if shape[2] != 0 and shape[3] != 0:
        shape[1] = fsize // (shape[3] * shape[2] * 4)
        shape[0] = fsize // (shape[3] * shape[2] * shape[1] * 4)

    # if there in no pulse program parameters in dictionary return current
    # shape after removing zeros
    if "pprog" not in dic or "loop" not in dic["pprog"]:
        return tuple([int(i) for i in shape if i > 1]), cplex

    # if pulseprogram dictionary is missing loop or incr return current shape
    pprog = dic["pprog"]
    if "loop" not in pprog or "incr" not in pprog:
        return tuple([int(i) for i in shape if i > 1]), cplex

    # determine indirect dimension sizes from pulseprogram parameters
    loop = pprog["loop"]
    loopn = len(loop)  # number of loops
    li = [len(i) for i in pprog["incr"]]  # length of incr lists

    # replace td0,td1,td2,td3 in loop list
    rep = {"td0": td0, "td1": td1, "td2": td2, "td3": td3}
    for i, v in enumerate(loop):
        if v in rep.keys():
            loop[i] = rep[v]

    # if the loop variables contains strings, return current shape
    # these variables could be resolved from the var key in the pprog dict
    # but this would require executing unknown code to perform the
    # arithmetic present in the string.
    if str in [type(e) for e in loop]:
        return tuple([int(i) for i in shape if i > 1]), cplex

    # size of indirect dimensions based on number of loops in pulse program
    # there are two kinds of loops, active and passive.
    # active loops are from indirect dimension increments, the corresponding
    # incr lists should have non-zero length and the size of the dimension
    # is twice that of the active loop size.
    # passive loops are from phase cycles and similar elements, these should
    # have zero length incr lists and should be of length 2.

    # The following checks for these and updates the indirect dimension
    # if the above is found.
    if loopn == 1:  # 2D with no leading passive loops
        if li[0] != 0:
            shape[2] = loop[0]
            shape = shape[-2:]

    elif loopn == 2:  # 2D with one leading passive loop
        if loop[0] == 2 and li[0] == 0 and li[1] != 0:
            shape[2] = 2 * loop[1]
            shape = shape[-2:]

    elif loopn == 3:  # 2D with two leading passive loops
        if loop[0] == 2 and loop[1] == 2 and li[0] == 0 and li[1] == 0 and li[2] != 0:
            shape[2] = 2 * loop[2]
            shape = shape[-2:]

    elif loopn == 4:  # 3D with one leading passive loop for each indirect dim
        if loop[0] == 2 and li[0] == 0 and li[1] != 0:
            shape[2] = 2 * loop[1]
        if loop[2] == 2 and li[2] == 0 and li[3] != 0:
            shape[1] = 2 * loop[3]
            shape = shape[-3:]

    elif loopn == 5:  # 3D with two/one leading passive loops
        if loop[1] == 2 and li[0] == 0 and li[1] == 0 and li[2] != 0:
            shape[2] = 2 * loop[2]
        if loop[3] == 2 and li[0] == 0 and li[3] == 0 and li[4] != 0:
            shape[1] = 2 * loop[4]
            shape = shape[-3:]

    elif loopn == 6:  # 4D with one leading passive loop for each indirect dim
        if loop[0] == 2 and li[0] == 0 and li[1] != 0:
            shape[2] = 2 * loop[1]
        if loop[2] == 2 and li[2] == 0 and li[3] != 0:
            shape[1] = 2 * loop[3]
        if loop[4] == 2 and li[4] == 0 and li[5] != 0:
            shape[0] = 2 * loop[5]

    elif loopn == 7:
        if loop[1] == 2 and li[0] == 0 and li[1] == 0 and li[2] != 0:
            shape[2] = 2 * loop[2]
        if loop[3] == 2 and li[0] == 0 and li[3] == 0 and li[4] != 0:
            shape[1] = 2 * loop[4]
        if loop[5] == 2 and li[0] == 0 and li[5] == 0 and li[6] != 0:
            shape[0] = 2 * loop[6]

    return tuple([int(i) for i in shape if i >= 2]), cplex


# Bruker processed binary (1r, 1i, 2rr, 2ri, etc) reading


def read_pdata(
    dir=".",
    bin_files=None,
    procs_files=None,
    read_procs=True,
    acqus_files=None,
    read_acqus=True,
    scale_data=True,
    shape=None,
    submatrix_shape=None,
    all_components=False,
    big=None,
    isfloat=None,
):
    """
    Read processed Bruker files from a directory.

    In Topspin and other programs this data is typically scaled by dividing by
    2 ** -NC_proc where NC_proc is defined in the procs file.  This scaling
    can be accomplished by setting the scale_data parameter to True.

    Parameters
    ----------
    dir : str
        Directory to read from.
    bin_files : list of str, optional
        List of filename of binary file in directory. None uses standard
        files.
    procs_files : list, optional
        List of filename(s) of procs parameter files in directory. None uses
        standard files.
    read_procs : bool, optional
        True to read procs files(s), False prevents reading.
    acqus_files : list, optional
        List of filename(s) of acqus parameter files in directory. None uses
        standard files.
    read_acqus : bool, optional
        True to read acqus files(s), False prevents reading.
    scale_data : bool, optional
        True, the default, to apply scaling defined in the procs file.  The
        data should almost always be scaled. False, returns the
        data as it appears in the file.
    shape : tuple, optional
        Shape of resulting data.  None will guess the shape from the
        parameters in the procs file(s).
    submatrix_shape : tuple, optional
        Shape of submatrix for 2D+ data.  None will guess the shape from
        the metadata in the procs file(s).
    all_components : bool
        True to return a list of all components, False returns just the
        all real component (1r, 2rr, 3rrr, etc).
    big : bool or None, optional
        Endianness of binary file. True for big-endian, False for
        little-endian, None to determine endianness from procs file(s).
    isfloat : bool or None, optional
        Data type of binary file. True for float64, False for int32. None to
        determine data type from procs file(s).

    Returns
    -------
    dic : dict
        Dictionary of Bruker parameters.
    data : ndarray or list
        Array of NMR data.  If all_components is True this is a list of array
        with each quadrature component.

    Notes
    -----
    There is currently no support for writing Bruker processed files or
    reading processed files using minimal memory.

    """
    # TODO read_pdata_lowmem, write_pdata

    if os.path.isdir(dir) is not True:
        raise IOError("directory %s does not exist" % (dir))

    # find binary files
    if bin_files is None:
        if os.path.isfile(os.path.join(dir, "1r")):
            if all_components:
                bin_files = ["1r", "1i"]
            else:
                bin_files = ["1r"]
        elif os.path.isfile(os.path.join(dir, "2rr")):
            if all_components:
                bin_files = ["2rr", "2ri", "2ir", "2ii"]
            else:
                bin_files = ["2rr"]
        elif os.path.isfile(os.path.join(dir, "3rrr")):
            if all_components:
                bin_files = [
                    "3rrr",
                    "3rri",
                    "3rir",
                    "3rii",
                    "3irr",
                    "3iri",
                    "3iir",
                    "3iii",
                ]
            else:
                bin_files = ["3rrr"]
        else:
            raise IOError("No Bruker binary file could be found in %s" % (dir))

    for f in bin_files.copy():
        if not os.path.isfile(os.path.join(dir, f)):
            bin_files.remove(f)

    if read_procs:
        # read the procs_files and add to the dictionary
        dic = read_procs_file(dir, procs_files)
    else:
        # create an empty dictionary
        dic = dict()

    if read_acqus:
        # If acqus files were not listed check in the usual place.
        acqus_dir = os.path.dirname(os.path.dirname(dir))
        if acqus_files is not None:
            acqus_dic = read_acqus_file(dir, acqus_files)
            # Merge the two dicts.
            dic = _merge_dict(dic, acqus_dic)

        elif os.path.isdir(acqus_dir):
            acqus_dic = read_acqus_file(acqus_dir)
            # Merge the two dicts.
            dic = _merge_dict(dic, acqus_dic)

    # determine shape and complexity for direct dim if needed
    if submatrix_shape is None or shape is None:
        g_shape, g_submatrix_shape = guess_shape_and_submatrix_shape(dic)
        if shape is None:
            shape = g_shape
        if submatrix_shape is None:
            submatrix_shape = g_submatrix_shape

    # issue a warning is submatrix_shape or shape are still None
    if submatrix_shape is None:
        warn("Submatrix shape not defined, returning 1D data")
    if shape is None:
        warn("Data shape not defined, returning 1D data")

    # determine endianness (assume little-endian unless BYTORDA is 1)
    if big is None:
        big = False  # default value
        if "procs" in dic and "BYTORDP" in dic["procs"]:
            if dic["procs"]["BYTORDP"] == 1:
                big = True
            else:
                big = False

    # determine data type (assume int32 unless DTYPA is 2)
    if isfloat is None:
        isfloat = False  # default value
        if "procs" in dic and "DTYPP" in dic["procs"]:
            if dic["procs"]["DTYPP"] == 2:
                isfloat = True
            else:
                isfloat = False

    # read the binary file
    data = [
        read_pdata_binary(os.path.join(dir, f), shape, submatrix_shape, big, isfloat)[1]
        for f in bin_files
    ]

    # scale data if requested
    if scale_data:
        data = [scale_pdata(dic, d) for d in data]

    if len(data) == 1:
        return dic, data[0]
    else:
        return dic, data


def scale_pdata(dic, data, reverse=False):
    """
    Scale Bruker processed data using parameters from the procs file.

    Parameters
    ----------
    dic : dict
        Dictionary of Bruker parameters.
    data : ndarray
        Array of NMR data.
    reverse : Bool
        True to reverse the scaling, i.e. multiply by the
        scaling factor rather than divide

    Returns
    -------
    sdata : array
        Scaled data.
    """
    try:
        scale = np.power(2.0, -float(dic["procs"]["NC_proc"]))
    except KeyError:
        warn("Unable to scale data, returning unscaled data")
        scale = 1

    if reverse:
        return data * scale
    else:
        return data / scale


def array_to_int(data):
    """
    Cast bruker (processed) data into int32 and normalise to have
    the absolute maximum intensity in the range [2**28, 2**29]

    Parameters
    ----------
    data : ndarray
        Array of NMR data (float64 or int32).
    reverse : Bool
        True to reverse the scaling, i.e. multiply by the
        scaling factor rather than divide

    Returns
    -------
    intdata : array
        Real valued data scaled to have the maximum intensity between
        2**28 and 2**29, converted to type int32
    """

    for _ in range(30):
        if np.max(abs(data)) < 2 ** 28:
            data *= 2
        else:
            break
    intdata = data.real.astype("int32")
    return intdata


def guess_shape_and_submatrix_shape(dic):
    """
    Guess the data shape and the shape of the processed data submatrix.
    """
    if "procs" not in dic:  # unknown dimensionality and shapes
        return None, None

    procs = dic["procs"]
    if "SI" not in procs or "XDIM" not in procs:
        return None, None  # cannot determine shape

    si_0 = procs["SI"]
    xdim_0 = procs["XDIM"]

    if "proc2s" not in dic:  # 1D data
        return (si_0,), (xdim_0,)

    proc2s = dic["proc2s"]
    if "SI" not in proc2s or "XDIM" not in proc2s:
        return None, None  # cannot determine shape

    si_1 = proc2s["SI"]
    xdim_1 = proc2s["XDIM"]

    if "proc3s" not in dic:  # 2D data
        return (si_1, si_0), (xdim_1, xdim_0)

    proc3s = dic["proc3s"]
    if "SI" not in proc3s or "XDIM" not in proc3s:
        return None, None  # cannot determine shape

    si_2 = proc3s["SI"]
    xdim_2 = proc3s["XDIM"]

    if "proc4s" not in dic:  # 3D data
        return (si_2, si_1, si_0), (xdim_2, xdim_1, xdim_0)

    proc4s = dic["proc4s"]
    if "SI" not in proc4s or "XDIM" not in proc4s:
        return None, None  # cannot determine shape

    si_3 = proc4s["SI"]
    xdim_3 = proc4s["XDIM"]

    # assume 4D data
    return (si_3, si_2, si_1, si_0), (xdim_3, xdim_2, xdim_1, xdim_0)


def read_pdata_binary(
    filename, shape=None, submatrix_shape=None, big=True, isfloat=False
):
    """
    Read a processed Bruker binary file and return dic, data pair.

    If data cannot be reshaped as described a 1D representation of the data
    will be returned after printing a warning message.

    Parameters
    ----------
    filename : str
        Filename of Bruker binary file.
    shape : tuple
        Shape of resulting data.  None will return 1D data.
    submatrix_shape : tuple
        Tuple describing shape of resulting data.  None will return 1D data.
    big : bool
        Endianness of binary file, True for big-endian, False for
        little-endian.
    isfloat : bool
        Data type of binary file. True for float64, False for int32.

    Returns
    -------
    dic : dict
        Dictionary containing "FILE_SIZE" key and value.
    data : ndarray
        Array of raw NMR data.

    """
    # open the file and get the data
    with open(filename, "rb") as f:
        data = get_data(f, big=big, isfloat=isfloat)

    # create dictionary
    dic = {"FILE_SIZE": os.stat(filename).st_size}

    # submatrix reordering
    if submatrix_shape is None or shape is None:
        return dic, data
    else:
        try:
            data = reorder_submatrix(data, shape, submatrix_shape)
            return dic, data
        except Exception:
            warn("unable to reorder data")
            return dic, data


def reorder_submatrix(data, shape, submatrix_shape, reverse=False):
    """
    Reorder processed binary Bruker data.

    Parameters
    ----------
    data : array

    shape : tuple
        Shape of final data.
    submatrix_shape : tuple
        Shape of submatrix.
    reverse : Bool
        True to reverse the reordering of a submatrix.
        This options is used to reorder a numpy matrix that is
        ordered correctly into the Bruker format using submatrix_shape

    Returns
    -------
    rdata : array
        Array in which data has been reordered and correctly shaped.

    """
    if submatrix_shape is None or shape is None:
        return data

    # do nothing to 1D data
    if len(submatrix_shape) == 1 or len(shape) == 1:
        return data

    sub_per_dim = [int(i / j) for i, j in zip(shape, submatrix_shape)]
    nsubs = np.product(sub_per_dim)

    if reverse:
        rdata = np.empty([nsubs] + list(submatrix_shape))
    else:
        data = data.reshape([nsubs] + list(submatrix_shape))
        rdata = np.empty(shape, dtype=data.dtype)

    for sub_num, sub_idx in enumerate(np.ndindex(tuple(sub_per_dim))):
        sub_slices = [
            slice(i * j, (i + 1) * j) for i, j in zip(sub_idx, submatrix_shape)
        ]
        if reverse:
            rdata[sub_num] = data[tuple(sub_slices)]
        else:
            rdata[tuple(sub_slices)] = data[sub_num]
    return rdata.reshape(shape)


# Bruker binary (fid/ser) reading and writing


def read_binary(filename, shape=(1), cplex=True, big=True, isfloat=False):
    """
    Read Bruker binary data from file and return dic,data pair.

    If data cannot be reshaped as described a 1D representation of the data
    will be returned after printing a warning message.

    Parameters
    ----------
    filename : str
        Filename of Bruker binary file.
    shape : tuple
        Tuple describing shape of resulting data.
    cplex : bool
        Flag indicating if direct dimension is complex.
    big : bool
        Endianness of binary file, True for big-endian, False for
        little-endian.
    isfloat : bool
        Data type of binary file. True for float64, False for int32.

    Returns
    -------
    dic : dict
        Dictionary containing "FILE_SIZE" key and value.
    data : ndarray
        Array of raw NMR data.

    See Also
    --------
    read_binary_lowmem : Read Bruker binary file using minimal memory.

    """
    # open the file and get the data
    with open(filename, "rb") as f:
        data = get_data(f, big=big, isfloat=isfloat)

    # complexify if needed
    if cplex:
        data = complexify_data(data)

    # create dictionary
    dic = {"FILE_SIZE": os.stat(filename).st_size}

    # reshape if possible
    try:
        return dic, data.reshape(shape)

    except ValueError:
        warn(str(data.shape) + "cannot be shaped into" + str(shape))
        return dic, data


# binary get/put functions


def get_data(f, big, isfloat):
    """
    Get binary data from file object with given endianness and data type.
    """
    if isfloat:
        if big:
            return np.frombuffer(f.read(), dtype=">f8")
        else:
            return np.frombuffer(f.read(), dtype="<f8")
    else:
        if big:
            return np.frombuffer(f.read(), dtype=">i4")
        else:
            return np.frombuffer(f.read(), dtype="<i4")


def get_trace(f, num_points, big, isfloat):
    """
    Get trace of num_points from file with given endianness and data type.
    """
    if isfloat:
        if big:
            bsize = num_points * np.dtype(">f8").itemsize
            return np.frombuffer(f.read(bsize), dtype=">f8")
        else:
            bsize = num_points * np.dtype("<f8").itemsize
            return np.frombuffer(f.read(bsize), dtype="<f8")
    else:
        if big:
            bsize = num_points * np.dtype(">i4").itemsize
            return np.frombuffer(f.read(bsize), dtype=">i4")
        else:
            bsize = num_points * np.dtype("<i4").itemsize
            return np.frombuffer(f.read(bsize), dtype="<i4")


# data manipulation functions


def complexify_data(data):
    """
    Complexify data packed real, imag.
    """
    return data[..., ::2] + data[..., 1::2] * 1.0j


def uncomplexify_data(data_in, isfloat):
    """
    Uncomplexify data (pack real,imag) into a int32 or float64 array,
    depending on isfloat.
    """
    size = list(data_in.shape)
    size[-1] = size[-1] * 2
    if isfloat:
        data_out = np.empty(size, dtype="float64")
    else:
        data_out = np.empty(size, dtype="int32")
    data_out[..., ::2] = data_in.real
    data_out[..., 1::2] = data_in.imag
    return data_out


# JCAMP-DX functions


def read_jcamp(filename, encoding=locale.getpreferredencoding()):
    """
    Read a Bruker JCAMP-DX file into a dictionary.

    Creates two special dictionary keys _coreheader and _comments Bruker
    parameter "$FOO" are extracted into strings, floats or lists and assigned
    to dic["FOO"]

    Parameters
    ----------
    filename : str
        Filename of Bruker JCAMP-DX file.
    encoding : str
        Encoding of Bruker JCAMP-DX file. Defaults to the system default locale

    Returns
    -------
    dic : dict
        Dictionary of parameters in file.

    See Also
    --------
    write_jcamp : Write a Bruker JCAMP-DX file.

    Notes
    -----
    This is not a fully functional JCAMP-DX reader, it is only intended
    to read Bruker acqus (and similar) files.

    """
    dic = {"_coreheader": [], "_comments": []}  # create empty dictionary

    with io.open(filename, "r", encoding=encoding) as f:
        while True:  # loop until end of file is found

            line = f.readline().rstrip()  # read a line
            if line == "":  # end of file found
                break

            if line[:6] == "##END=":
                # print("End of file")
                break
            elif line[:2] == "$$":
                dic["_comments"].append(line)
            elif line[:2] == "##" and line[2] != "$":
                dic["_coreheader"].append(line)
            elif line[:3] == "##$":
                try:
                    key, value = parse_jcamp_line(line, f)
                    dic[key] = value
                except Exception:
                    warn("Unable to correctly parse line:" + line)
            else:
                warn("Extraneous line:" + line)

    return dic


def parse_jcamp_line(line, f):
    """
    Parse a single JCAMP-DX line.

    Extract the Bruker parameter name and value from a line from a JCAMP-DX
    file.  This may entail reading additional lines from the fileobj f if the
    parameter value extends over multiple lines.

    """

    # extract key= text from line
    key = line[3 : line.index("=")]
    text = line[line.index("=") + 1 :].lstrip()

    if "<" in text:  # string
        while ">" not in text:  # grab additional text until ">" in string
            text = text + "\n" + f.readline().rstrip()
        value = text[1:-1]  # remove < and >

    elif "(" in text:  # array
        num = int(line[line.index("..") + 2 : line.index(")")]) + 1
        value = []
        rline = line[line.index(")") + 1 :]

        # extract value from remainder of line
        for t in rline.split():
            value.append(parse_jcamp_value(t))

        # parse additional lines as necessary
        while len(value) < num:
            nline = f.readline().rstrip()
            for t in nline.split():
                value.append(parse_jcamp_value(t))

    elif text == "yes":
        value = True

    elif text == "no":
        value = False

    else:  # simple value
        value = parse_jcamp_value(text)

    return key, value


def parse_jcamp_value(text):
    """
    Parse value text from Bruker JCAMP-DX file returning the value.
    """
    if text == "":
        return None
    elif text.startswith("<") and text.endswith(">"):
        return text[1:-1]  # remove < and >
    else:
        if "." in text or "e" in text or "inf" in text:
            try:
                return float(text)
            except ValueError:
                return text
        else:
            try:
                return int(text)
            except ValueError:
                return text


def read_pprog(filename):
    """
    Read a Bruker pulse program (pulseprogram) file.

    Resultsing dictionary contains the following keys:

    ========    ===========================================================
    key         description
    ========    ===========================================================
    var         dictionary of variables assigned in pulseprogram
    incr        list of lists containing increment times
    loop        list of loop multipliers
    phase       list of lists containing phase elements
    ph_extra    list of lists containing comments at the end of phase lines
    ========    ===========================================================

    The incr,phase and ph_extra lists match up with loop list.  For example
    incr[0],phase[0] and ph_extra[0] are all increment and phase commands
    with comments which occur during loop 0 which has loop[0] steps.

    Parameters
    ----------
    filename : str
        Filename of pulseprogram file to read from,

    Returns
    -------
    dic : dict
        A dictionary with keys described above.

    See Also
    --------
    write_pprog : Write a Bruker pulse program to file.

    """

    # open the file
    f = open(filename, "r")

    # initialize lists and dictionaries
    var = dict()
    loop = []
    incr = [[]]
    phase = [[]]
    ph_extra = [[]]

    # loop over lines in pulseprogram looking for loops, increment,
    # assignments and phase commands
    for line in f:

        # split line into comment and text and strip leading/trailing spaces
        if ";" in line:
            text = line[: line.index(";")].strip()
        else:
            text = line.strip()

        # remove label from text when first word is all digits or
        # has "," as the last element
        if len(text.split()) != 0:
            s = text.split()[0]
            if s.isdigit() or s[-1] == ",":
                text = text[len(s) :].strip()

        # skip blank lines and include lines
        if text == "" or text[0] == "#":
            # print(line,"--Blank, Comment or Include")
            continue

        # see if we have quotes and have an assignment
        # syntax "foo=bar"
        # add foo:bar to var dictionary
        if '"' in text:
            if "=" in line:
                # strip quotes, split on = and add to var dictionary
                text = text.strip('"')
                t = text.split("=")
                if len(t) >= 2:
                    key, value = t[0], t[1]
                    var[key] = value
                    # print(line,"--Assignment")
                else:
                    pass
                    # print(line,"--Statement")
                continue
            else:
                # print(line,"--Statement")
                continue

        # loops begin with lo
        # syntax is: lo to N time M
        # add M to loop list
        if text[0:2] == "lo":
            loop.append(text.split()[4])
            incr.append([])
            phase.append([])
            ph_extra.append([])
            # print(line,"--Loop")
            continue

        tokens = text.split()
        if len(tokens) >= 2:
            token2 = tokens[1]
            # increment statement have id, dd, ipu or dpu
            # syntax foo {id/dd/ipu/dpu}N
            # store N to incr list
            if token2.startswith("id") or token2.startswith("dd"):
                incr[len(loop)].append(int(token2[2:]))
                # print(line,"--Increment")
                continue

            if token2.startswith("ipu") or token2.startswith("dpu"):
                incr[len(loop)].append(int(token2[3:]))
                # print(line,"--Increment")
                continue

            # phase statement have ip or dp
            # syntax fpp {ip/dp}N extra
            # store N to phase list and extra to ph_extra list
            if token2.startswith("ip") or token2.startswith("dp"):
                phase[len(loop)].append(int(token2[2:]))

                # find the first space after "ip" and read past there
                last = text.find(" ", text.index("ip"))
                if last == -1:
                    ph_extra[len(loop)].append("")
                else:
                    ph_extra[len(loop)].append(text[last:].strip())
                # print(line,"--Phase")
                continue

            # print(line,"--Unimportant")

    f.close()

    # remove the last empty incr, phase and ph_extra lists
    incr.pop()
    phase.pop()
    ph_extra.pop()

    # convert loop to numbers if possible
    for i, t in enumerate(loop):
        if t.isdigit():
            loop[i] = int(t)
        else:
            if (t in var) and var[t].isdigit():
                loop[i] = int(var[t])

    # create the output dictionary
    dic = {"var": var, "incr": incr, "loop": loop, "phase": phase, "ph_extra": ph_extra}
    return dic


def _merge_dict(a, b):
    c = a.copy()
    c.update(b)
    return c
