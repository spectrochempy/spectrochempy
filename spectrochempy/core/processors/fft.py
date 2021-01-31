# -*- coding: utf-8 -*-

# ======================================================================================================================
# Copyright (©) 2015-2016 Christian Fernandez
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

__all__ = ["fft", "ifft"]

__dataset_methods__ = __all__

# ======================================================================================================================
# Standard python imports
# ======================================================================================================================
import re
import matplotlib.pyplot as plt

# ======================================================================================================================
# Third party imports
# ======================================================================================================================
import numpy as np
from scipy.interpolate import interp1d

# ======================================================================================================================
# Local imports
# ======================================================================================================================

from spectrochempy.core import error_
from spectrochempy.units import ur
from spectrochempy.core.dataset.ndmath import zeros_like
from spectrochempy.core.processors.apodization import hamming
from spectrochempy.core.processors.concatenate import concatenate
from spectrochempy.utils import largest_power_of_2

_fft = lambda data: np.fft.fftshift(np.fft.fft(data), -1)
_ifft = lambda data : np.fft.ifft(np.fft.ifftshift(data, -1))
_fft_positive = lambda data : np.fft.fftshift(np.fft.ifft(data).astype(data.dtype)) * data.shape[-1]
_ifft_positive = lambda data : np.fft.fft(np.fft.ifftshift(data, -1)) * data.shape[-1]


def _get_zpd(dataset, dim=-1, mode='max'):
    """
    Find the zero path difference (zpd) positions.

    For multidimensional NDDataset the search is by default performed along the last dimension.

    Parameters
    ----------
    dataset : |NDDataset|
        The dataset on which to search for zpd
    dim: int or str, optional
        Dimension along which to make the search. Default=-1.
    mode : enum('max','abs'), optional
        Mode of selection. Default = 'max'.

    Returns
    -------
    index
        zero path difference index
    """
    # On which axis do we want to work (get axis from arguments)
    axis, dim = dataset.get_axis(dim, negative_axis=True)

    if mode == 'max':
        return np.argmax(dataset.data, axis=axis)
    elif mode == 'abs':
        return np.argmax(np.abs(dataset.data), axis=axis)


def ifft(dataset, size=None, **kwargs):
    """
    Apply a inverse fast fourier transform.

    For multidimensional NDDataset,
    the apodization is by default performed on the last dimension.

    The data in the last dimension MUST be in frequency (or without dimension)
    or an error is raised.

    To make direct Fourier transform, i.e., from frequency to time domain, use the `fft` transform.

    Parameters
    ----------
    dataset : |NDDataset|
        The dataset on which to apply the fft transformation.
    size : int, optional
        Size of the transformed dataset dimension - a shorter parameter is `si`. by default, the size is the closest
        power of two greater than the data size.
    **kwargs : dict
        Other parameters (see other parameters).

    Returns
    -------
    out
        Transformed |NDDataset|.

    Other Parameters
    ----------------
    dim : str or int, optional, default='x'.
        Specify on which dimension to apply this method. If `dim` is specified as an integer it is equivalent
        to the usual `axis` numpy parameter.
    inplace : bool, optional, default=False.
        True if we make the transform inplace.  If False, the function return a new object

    See Also
    --------
    fft : Direct Fourier transform.
    """
    return fft(dataset, size=size, inv=True, **kwargs)


def fft(dataset, size=None, sizeff=None, inv=False, ppm=True, **kwargs):
    """
    Apply a complex fast fourier transform.

    For multidimensional NDDataset,
    the apodization is by default performed on the last dimension.

    The data in the last dimension MUST be in time-domain (or without dimension)
    or an error is raised.

    To make reverse Fourier transform, i.e., from frequency to time domain, use the `ifft` transform
    (or equivalently, the `inv=True` parameters.

    Parameters
    ----------
    dataset : |NDDataset|
        The dataset on which to apply the fft transformation.
    size : int, optional
        Size of the transformed dataset dimension - a shorter parameter is `si`. by default, the size is the closest
        power of two greater than the data size.
    sizeff : int, optional
        The number of effective data point to take into account for the transformation. By default it is equal to the
        data size, but may be smaller.
    inv : bool, optional, default=False
        If True, an inverse Fourier transform is performed - size parameter is not taken into account.
    ppm : bool, optional, default=True
        If True, and data are from NMR, then a ppm scale is calculated instead of frequency.
    **kwargs : dict
        Other parameters (see other parameters).

    Returns
    -------
    out
        Transformed |NDDataset|.

    Other Parameters
    ----------------
    dim : str or int, optional, default='x'.
        Specify on which dimension to apply this method. If `dim` is specified as an integer it is equivalent
        to the usual `axis` numpy parameter.
    inplace : bool, optional, default=False.
        True if we make the transform inplace.  If False, the function return a new object
    tdeff : int, optional
        Alias of sizeff (specific to NMR). If both sizeff and tdeff are passed, sizeff has the priority.

    See Also
    --------
    ifft : Inverse Fourier transform.
    """
    # datatype
    is_nmr = dataset.origin.lower() in ["topspin", ]
    is_ir = dataset.origin.lower() in ["omnic", "opus"]

    # On which axis do we want to apply transform (get axis from arguments)
    dim = kwargs.pop('dim', kwargs.pop('axis', -1))
    axis, dim = dataset.get_axis(dim, negative_axis=True)

    # output dataset inplace or not
    inplace = kwargs.pop('inplace', False)
    if not inplace:  # default
        new = dataset.copy()  # copy to be sure not to modify this dataset
    else:
        new = dataset

    # The last dimension is always the dimension on which we apply the fourier transform.
    # If needed, we swap the dimensions to be sure to be in this situation
    swaped = False
    if axis != -1:
        new.swapdims(axis, -1, inplace=True)  # must be done in  place
        swaped = True

    # select the last coordinates
    lastcoord = new.coordset[dim]

    if not inv and not lastcoord.dimensionless \
            and lastcoord.units.dimensionality != '[time]':
        error_('fft apply only to dimensions with [time] dimensionality or dimensionless coords\n'
               'fft processing was thus cancelled')
        return dataset

    elif inv and lastcoord.units.dimensionality != '1/[time]' and not lastcoord.dimensionless:
        error_('ifft apply only to dimensions with [frequency] dimensionality or with ppm units '
               'or dimensionless coords.\n ifft processing was thus cancelled')
        return dataset

    elif new.is_masked:
        error_('current fft or ifft processing does not support masked data as input.\n processing was thus cancelled')

    # TODO: other tests data spacing and so on.

    # Can we use some metadata as for NMR spectra
    if is_nmr and not inv:
        td = new.meta.td[-1]
    else:
        td = lastcoord.size

    # if no size (or si) parameter then use the size of the data (size not used for inverse transform
    if size is None or inv:
        size = kwargs.get('si', td)

    # we default to the closest power of two larger of the data size
    size = largest_power_of_2(size)

    # do we have an effective td to apply
    tdeff = sizeff
    if tdeff is None:
        tdeff = kwargs.get("tdeff", td)

    if tdeff is None or tdeff < 5 or tdeff > size:
        tdeff = size

    # Eventually apply the effective size
    new[..., tdeff:] = 0.

    # should we work on complex data
    iscomplex = new.is_complex

    # if we are in NMR we have an additional complication due to the mode
    # of acquisition (sequential mode when ['QSEQ','TPPI','STATES-TPPI'])
    encoding = None
    if is_nmr and not inv:
        encoding = new.meta.encoding[-1]

    # perform the fft
    if iscomplex and encoding in ['QSIM', 'DQD']:
        data = zf_size(new.data, size)
        data = _fft(data)

    elif inv:
        # we assume no special encoding for inverse fft transform
        data = _ifft(new.data)

    elif is_ir and not inv:

        # TODO: revise this

        # subtract  DC
        new -= new.mean()
        # determine phase correction (Mertz)
        zpd = _get_zpd(new)
        if not np.all(zpd[0] == zpd):
            raise ValueError("zpd should be at the same index")
        zpd = zpd[0]
        narrowed = hamming(new[:, 0: 2 * zpd])
        mirrored = concatenate(narrowed[:, zpd:], narrowed[:, :zpd])
        spectrum = np.fft.rfft(mirrored.data)
        phase_angle = np.arctan(spectrum.imag, spectrum.real)
        initx = np.arange(phase_angle.shape[1])
        interpolate_phase_angle = interp1d(initx, phase_angle)

        zeroed = concatenate(zeros_like(new[:, zpd + 1:]), new)
        apodized = hamming(zeroed)  # mertz(new, zpd)
        zpd = len(apodized.x) // 2
        mirrored = concatenate(apodized[:, zpd:], apodized[:, 0:zpd])

        wavenumbers = np.fft.rfftfreq(mirrored.shape[1], 3.165090310992977e-05 * 2)

        spectrum = np.fft.rfft(mirrored.data)
        #plt.plot(wavenumbers, spectrum[0])
        #plt.show()
        newx = np.arange(spectrum.shape[1]) * max(initx) / max(np.arange(spectrum.shape[1]))
        phase_angle = interpolate_phase_angle(newx)
        spectrum = spectrum.real * np.cos(phase_angle) + spectrum.imag * np.sin(phase_angle)

        #plt.plot(wavenumbers, spectrum[0])
        #plt.show()

    else:
        raise NotImplementedError(encoding)

    # We need here to create a new dataset with new shape and axis
    new._data = data
    new.mask = False  # TODO: make a test on mask - should be none before fft!

    # create new coordinates for the transformed data

    if is_nmr:
        sfo1 = new.meta.sfo1[-1]
        bf1 = new.meta.bf1[-1]
        sf = new.meta.sf[-1]
        sw = new.meta.sw_h[-1]

    else:
        sfo1 = 1.0 * ur.Hz
        bf1 = sfo1
        sf = 0 * ur.Hz
        dw = lastcoord.spacing
        sw = 1. / dw

    if not inv:
        # time to frequency
        sizem = max(size - 1, 1)
        deltaf = -sw / sizem
        first = sfo1 - sf - deltaf * sizem / 2.

        newcoord = type(lastcoord)(np.arange(size) * deltaf + first)
        newcoord.name = lastcoord.name
        newcoord.title = 'frequency'
        newcoord.ito("Hz")

    else:
        # frequency or ppm to time
        sw = abs(lastcoord.data[-1] - lastcoord.data[0])
        if lastcoord.units == 'ppm':
            sw = bf1.to("Hz") * sw / 1.0e6
        deltat = 1. / sw

        newcoord = type(lastcoord)(np.arange(size) * deltat)
        newcoord.name = lastcoord.name
        newcoord.title = 'time'
        newcoord.ito("s")

    if is_nmr and not inv:
        newcoord.meta.larmor = bf1  # needed for ppm transformation
        ppm = kwargs.get('ppm', True)
        if ppm:
            newcoord.ito('ppm')
            if new.meta.nuc1 is not None:
                nuc1 = new.meta.nuc1[-1]
                regex = r"([^a-zA-Z]+)([a-zA-Z]+)"
                m = re.match(regex, nuc1)
                if m is not None:
                    mass = m[1]
                    name = m[2]
                    nucleus = '^{' + mass + '}' + name
                else:
                    nucleus = ""
            else:
                nucleus = ""
            newcoord.title = fr"$\delta\ {nucleus}$"

    new.coordset[-1] = newcoord

    if not inv:
        # phase frequency domain

        # if some phase related metadata do not exist yet, initialize them
        new.meta.readonly = False

        if not new.meta.phased:
            new.meta.phased = [False] * new.ndim

        new.meta.pivot = [abs(new).coordset[i].max() for i in range(new.ndim)]  # create pivot metadata

        # applied the stored phases
        new.pk(inplace=True)

        new.meta.readonly = True

    # restore original data order if it was swaped
    if swaped:
        new.swapdims(axis, -1, inplace=True)  # must be done inplace

    s = 'ifft' if inv else 'fft'
    new.history = f'{s} applied on dimension {dim}'
    return new

ft = fft
ift = ifft

def zf_inter(data, pts=1):
    """
    Zero fill between points.

    Parameters
    ----------
    data : ndarray
        Array of NMR data.
    pts : int
        Number zeros to add between points.

    Returns
    -------
    ndata : ndarray
        Array of NMR data to which `pts` zero have been added between all
        points.

    """
    size = list(data.shape)
    size[-1] = (pts + 1) * size[-1]
    z = np.zeros(size, dtype=data.dtype)
    z[..., ::pts + 1] = data[..., :]
    return z


def zf_pad(data, pad=0, mid=False):
    """
    Zero fill by padding with zeros.

    Parameters
    ----------
    data : ndarray
        Array of NMR data.
    pad : int
        Number of zeros to pad data with.
    mid : bool
        True to zero fill in middle of data.

    Returns
    -------
    ndata : ndarray
        Array of NMR data to which `pad` zeros have been appended to the end or
        middle of the data.

    """
    size = list(data.shape)
    size[-1] = int(pad)
    z = np.zeros(size, dtype=data.dtype)

    if mid:
        h = int(data.shape[-1] / 2.0)
        return np.concatenate((data[..., :h], z, data[..., h:]), axis=-1)
    else:
        return np.concatenate((data, z), axis=-1)


zf = zf_pad


def zf_double(data, n, mid=False):
    """
    Zero fill by doubling original data size once or multiple times.

    Parameters
    ----------
    data : ndarray
        Array of NMR data.
    n : int
        Number of times to double the size of the data.
    mid : bool
        True to zero fill in the middle of data.

    Returns
    -------
    ndata : ndarray
        Zero filled array of NMR data.

    """
    return zf_pad(data, int((data.shape[-1] * 2 ** n) - data.shape[-1]), mid)


def zf_size(data, size, mid=False):
    """
    Zero fill to given size.

    Parameters
    ----------
    data : ndarray
        Array of NMR data.
    size : int
        Size of data after zero filling.
    mid : bool
        True to zero fill in the middle of data.

    Returns
    -------
    ndata : ndarray
        Zero filled array of NMR data.

    """
    return zf_pad(data, int(size - data.shape[-1]), mid)


def zf_auto(data, mid=False):
    """
    Zero fill to next largest power of two.

    Parameters
    ----------
    data : ndarray
        Array of NMR data.
    mid : bool
        True to zero fill in the middle of data.

    Returns
    -------
    ndata : ndarray
        Zero filled array of NMR data.

    """
    return zf_size(data, largest_power_of_2(data.shape[-1]), mid)


# Modulus Calculation
def mc(data):
    """
    Modulus calculation.

    Calculates sqrt(real^2 + imag^2)
    """
    return np.sqrt(data.real ** 2 + data.imag ** 2)


def mc_pow(data):
    """
    Modulus calculation. Squared version.

    Calculated real^2+imag^2
    """
    return data.real ** 2 + data.imag ** 2


def hadamard(data):
    """
    Hadamard Transform

    Copied for NMRGlue - ha
    Parameters
    ----------
    data : ndarray
        Array of NMR data.

    Returns
    -------
    ndata : ndarray
        Hadamard transform of NMR data.

    Notes
    -----
    This function is very slow.  Implement a Fast Walsh-Hadamard Transform
    with sequency/Walsh ordering (FWHT_w) will result in much faster tranforms.

    http://en.wikipedia.org/wiki/Walsh_matrix
    http://en.wikipedia.org/wiki/Fast_Hadamard_transform

    """
    # implementation is a proof of concept and EXTEMEMLY SLOW

    # Hadamard Transform functions
    def _int2bin(n, digits=8):
        """
        Integer to binary string
        """
        return "".join([str((n >> y) & 1) for y in range(digits - 1, -1, -1)])

    def _bin2int(s):
        """
        binary string to integer
        """
        o = 0
        k = len(s) - 1
        for i, v in enumerate(s):
            o = o + int(v) * 2 ** (k - i)
        return o

    def _gray(n):
        """
        Calculate n-bit gray code
        """
        g = [0, 1]
        for i in range(1, int(n)):
            mg = g + g[::-1]  # mirror the current code
            # first bit 0/2**u for mirror
            first = [0] * 2 ** (i) + [2 ** (i)] * 2 ** (i)
            g = [mg[j] + first[j] for j in range(2 ** (i + 1))]
        return g


    # determind the order and final size of input vectors
    ord = int(np.ceil(np.log2(data.shape[-1])))  # Walsh/Hadamard order
    max = 2 ** ord

    # zero fill to power of 2
    pad = max - data.shape[-1]
    zdata = zf(data, pad)

    # Multiple each vector by the hadamard matrix
    nat = np.zeros(zdata.shape, dtype=zdata.dtype)
    H = scipy.linalg.hadamard(max)
    nat = np.dot(zdata, H)
    nat = np.array(nat, dtype=data.dtype)

    # Bit-Reversal Permutation
    s = [_int2bin(x, digits=ord)[::-1] for x in range(max)]
    brp = [_bin2int(x) for x in s]
    brp_data = np.take(nat, brp, axis=-1)

    # Gray code permutation (bit-inverse)
    gp = _gray(ord)
    gp_data = np.take(brp_data, gp, axis=-1)

    return gp_data


def ht(data, N=None):
    """
    Hilbert transform.

    Reconstruct imaginary data via hilbert transform.

    Parameters
    ----------
    data : ndarrat
        Array of NMR data.
    N : int or None
        Number of Fourier components.

    Returns
    -------
    ndata : ndarray
        NMR data which has been Hilvert transformed.

    """
    # XXX come back and fix this when a sane version of scipy.signal.hilbert
    # is included with scipy 0.8

    # create an empty output array
    fac = N / data.shape[-1]
    z = np.empty(data.shape, dtype=(data.flat[0] + data.flat[1] * 1.j).dtype)
    if data.ndim == 1:
        z[:] = scipy.signal.hilbert(data.real, N)[:data.shape[-1]] * fac
    else:
        for i, vec in enumerate(data):
            z[i] = scipy.signal.hilbert(vec.real, N)[:data.shape[-1]] * fac

    # correct the real data as sometimes it changes
    z.real = data.real
    return z

# ======================================================================================================================
if __name__ == '__main__':  # pragma: no cover
    pass
