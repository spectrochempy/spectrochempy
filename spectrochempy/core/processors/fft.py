# -*- coding: utf-8 -*-

# ======================================================================================================================
# Copyright (©) 2015-2016 Christian Fernandez
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

__all__ = ["fft", "ifft", "mc", "ps", "ht"]

__dataset_methods__ = __all__

import re
import numpy as np
from scipy.signal import hilbert
from quaternion import as_float_array

from spectrochempy.core import error_
from spectrochempy.units import ur
from spectrochempy.core.dataset.coord import LinearCoord
from spectrochempy.utils import (
    largest_power_of_2,
    get_component,
    typequaternion,
    as_quaternion,
)
from spectrochempy.core.processors.utils import _units_agnostic_method
from spectrochempy.core.processors.zero_filling import zf_size


# ======================================================================================================================
# Private methods
# ======================================================================================================================


def _fft(data):
    if data.dtype == typequaternion:

        dr = get_component(data, "R")
        fr = np.fft.fftshift(np.fft.fft(dr), -1)
        di = get_component(data, "I")
        fi = np.fft.fftshift(np.fft.fft(di), -1)

        # rebuild the quaternion
        data = as_quaternion(fr, fi)

    else:
        data = np.fft.fftshift(np.fft.fft(data), -1)

    return data


def _ifft(data):
    if data.dtype == typequaternion:

        fr = get_component(data, "R")
        dr = np.fft.ifft(np.fft.ifftshift(fr, -1))
        fi = get_component(data, "I")
        di = np.fft.ifft(np.fft.ifftshift(fi, -1))

        # rebuild the quaternion
        data = as_quaternion(dr, di)

    else:
        data = np.fft.ifft(np.fft.ifftshift(data, -1))

    return data


def _fft_positive(data):
    if data.dtype == typequaternion:

        dr = get_component(data, "R")
        fr = np.fft.fftshift(np.fft.ifft(dr).astype(data.dtype)) * data.shape[-1]
        di = get_component(data, "I")
        fi = np.fft.fftshift(np.fft.ifft(di).astype(data.dtype)) * data.shape[-1]

        # rebuild the quaternion
        data = as_quaternion(fr, fi)

    else:
        data = np.fft.fftshift(np.fft.ifft(data).astype(data.dtype)) * data.shape[-1]

    return data


def _ifft_positive(data):
    if data.dtype == typequaternion:

        fr = get_component(data, "R")
        dr = np.fft.fft(np.fft.ifftshift(fr, -1)) * data.shape[-1]
        fi = get_component(data, "I")
        di = np.fft.fft(np.fft.ifftshift(fi, -1)) * data.shape[-1]

        # rebuild the quaternion
        data = as_quaternion(dr, di)

    else:
        data = np.fft.fft(np.fft.ifftshift(data, -1)) * data.shape[-1]

    return data


def _states_fft(data, tppi=False):
    # FFT transform according to STATES encoding

    # warning: at this point, data must have been swaped so the last dimension is the one used for FFT
    wt, yt, xt, zt = as_float_array(
        data
    ).T  # x and y are exchanged due to swaping of dims
    w, y, x, z = wt.T, yt.T, xt.T, zt.T

    sr = (w - 1j * y) / 2.0
    si = (x - 1j * z) / 2.0

    if tppi:
        sr[..., 1::2] = -sr[..., 1::2]
        si[..., 1::2] = -si[..., 1::2]

    fr = np.fft.fftshift(np.fft.fft(sr), -1)
    fi = np.fft.fftshift(np.fft.fft(si), -1)

    # rebuild the quaternion
    data = as_quaternion(fr, fi)

    return data


def _echoanti_fft(data):
    # FFT transform according to ECHO-ANTIECHO encoding

    # warning: at this point, data must have been swaped so the last dimension is the one used for FFT
    wt, yt, xt, zt = as_float_array(
        data
    ).T  # x and y are exchanged due to swaping of dims
    w, y, x, z = wt.T, xt.T, yt.T, zt.T

    c = (w + y) + 1j * (w - y)
    s = (x + z) - 1j * (x - z)
    fc = np.fft.fftshift(np.fft.fft(c / 2.0), -1)
    fs = np.fft.fftshift(np.fft.fft(s / 2.0), -1)
    data = as_quaternion(fc, fs)

    return data


def _tppi_fft(data):
    # FFT transform according to TPPI encoding

    # warning: at this point, data must have been swaped so the last dimension is the one used for FFT
    wt, yt, xt, zt = as_float_array(
        data
    ).T  # x and y are exchanged due to swaping of dims
    w, y, x, z = wt.T, xt.T, yt.T, zt.T

    sx = w + 1j * y
    sy = x + 1j * z

    sx[..., 1::2] = -sx[..., 1::2]
    sy[..., 1::2] = -sy[..., 1::2]

    fx = np.fft.fftshift(np.fft.fft(sx), -1)  # reverse
    fy = np.fft.fftshift(np.fft.fft(sy), -1)

    # rebuild the quaternion
    data = as_quaternion(fx, fy)

    return data


def _qf_fft(data):
    # FFT transform according to QF encoding

    data = np.fft.fftshift(np.fft.fft(np.conjugate(data)), -1)

    return data


def _interferogram_fft(data):
    """
    FFT transform for rapid-scan interferograms. Phase corrected using the Mertz method.
    """

    def _get_zpd(data, mode="max"):
        if mode == "max":
            return np.argmax(data, -1)
        elif mode == "abs":
            return int(np.argmax(np.abs(data), -1))

    zpd = _get_zpd(data, mode="abs")
    size = data.shape[-1]

    # Compute Mertz phase correction
    w = np.arange(0, zpd) / zpd
    ma = np.concatenate((w, w[::-1]))
    dma = np.zeros_like(data)
    dma[..., 0 : 2 * zpd] = data[..., 0 : 2 * zpd] * ma[0 : 2 * zpd]
    dma = np.roll(dma, -zpd)
    dma[0] = dma[0] / 2.0
    dma[-1] = dma[-1] / 2.0
    dma = np.fft.rfft(dma)[..., 0 : size // 2]
    phase = np.arctan(dma.imag / dma.real)

    # Make final phase corrected spectrum
    w = np.arange(0, 2 * zpd) / (2 * zpd)

    mapod = np.ones_like(data)
    mapod[..., 0 : 2 * zpd] = w
    data = np.roll(data * mapod, int(-zpd))
    data = np.fft.rfft(data)[..., 0 : size // 2] * np.exp(-1j * phase)

    # The imaginary part can be now discarder
    return data.real[..., ::-1] / 2.0


# ======================================================================================================================
# Public methods
# ======================================================================================================================


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
    is_nmr = dataset.origin.lower() in [
        "topspin",
    ]
    is_ir = dataset.origin.lower() in ["omnic", "opus"]

    # On which axis do we want to apply transform (get axis from arguments)
    dim = kwargs.pop("dim", kwargs.pop("axis", -1))
    axis, dim = dataset.get_axis(dim, negative_axis=True)

    # output dataset inplace or not
    inplace = kwargs.pop("inplace", False)
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

    # Select the last coordinates
    x = new.coordset[dim]

    # Performs some dimentionality checking
    error = False
    if (
        not inv
        and not x.unitless
        and not x.dimensionless
        and x.units.dimensionality != "[time]"
    ):
        error_(
            "fft apply only to dimensions with [time] dimensionality or dimensionless coords\n"
            "fft processing was thus cancelled"
        )
        error = True

    elif (
        inv
        and not x.unitless
        and x.units.dimensionality != "1/[time]"
        and not x.dimensionless
    ):
        error_(
            "ifft apply only to dimensions with [frequency] dimensionality or with ppm units "
            "or dimensionless coords.\n ifft processing was thus cancelled"
        )
        error = True

    # Should not be masked
    elif new.is_masked:
        error_(
            "current fft or ifft processing does not support masked data as input.\n processing was thus cancelled"
        )
        error = True

    # Coordinates should be uniformly spaced (linear coordinate)
    if not x.linear:
        # try to linearize it
        x.linear = True
        if not x.linear:
            # linearization failed
            error = True

    if hasattr(x, "_use_time_axis"):
        x._use_time_axis = True  # we need to havze dimentionless or time units

    if not error:
        # OK we can proceed

        # time domain size
        td = None
        if not inv:
            td = x.size

        # if no size (or si) parameter then use the size of the data (size not used for inverse transform
        if size is None or inv:
            size = kwargs.get("si", x.size)

        # we default to the closest power of two larger of the data size
        if is_nmr:
            size = largest_power_of_2(size)

        # do we have an effective td to apply
        tdeff = sizeff
        if tdeff is None:
            tdeff = kwargs.get("tdeff", td)

        if tdeff is None or tdeff < 5 or tdeff > size:
            tdeff = size

        # Eventually apply the effective size
        new[..., tdeff:] = 0.0

        # Should we work on complex or hypercomplex data
        # interleaved is in case of >2D data  ( # TODO: >D not yet implemented in ndcomplex.py
        iscomplex = False
        if axis == -1:
            iscomplex = new.is_complex
        if new.is_quaternion or new.is_interleaved:
            iscomplex = True

        # If we are in NMR we have an additional complication due to the mode
        # of acquisition (sequential mode when ['QSEQ','TPPI','STATES-TPPI'])
        encoding = "undefined"
        if not inv and "encoding" in new.meta:
            encoding = new.meta.encoding[-1]

        qsim = encoding in ["QSIM", "DQD"]
        qseq = "QSEQ" in encoding
        states = "STATES" in encoding
        echoanti = "ECHO-ANTIECHO" in encoding
        tppi = "TPPI" in encoding
        qf = "QF" in encoding

        zf_size(new, size=size, inplace=True)

        # Perform the fft
        if qsim:  # F2 fourier transform
            data = _fft(new.data)

        elif qseq:
            raise NotImplementedError("QSEQ not yet implemented")

        elif states:
            data = _states_fft(new.data, tppi)

        elif tppi:
            data = _tppi_fft(new.data)

        elif echoanti:
            data = _echoanti_fft(new.data)

        elif qf:
            # we must perform a real fourier transform of a time domain dataset
            data = _qf_fft(new.data)

        elif iscomplex and inv:
            # We assume no special encoding for inverse complex fft transform
            data = _ifft(new.data)

        elif not iscomplex and not inv and is_ir:
            # transform interferogram
            data = _interferogram_fft(new.data)

        elif not iscomplex and inv:
            raise NotImplementedError("Inverse FFT for real dimension")

        else:
            raise NotImplementedError(
                f"{encoding} not yet implemented. We recommend you to put an issue on "
                f"Github, so we will not forget to work on this!."
            )

        # We need here to create a new dataset with new shape and axis
        new._data = data
        new.mask = False

        # create new coordinates for the transformed data

        if is_nmr:
            sfo1 = new.meta.sfo1[-1]
            bf1 = new.meta.bf1[-1]
            sf = new.meta.sf[-1]
            sw = new.meta.sw_h[-1]
            if new.meta.nuc1 is not None:
                nuc1 = new.meta.nuc1[-1]
                regex = r"([^a-zA-Z]+)([a-zA-Z]+)"
                m = re.match(regex, nuc1)
                if m is not None:
                    mass = m[1]
                    name = m[2]
                    nucleus = "^{" + mass + "}" + name
                else:
                    nucleus = ""
            else:
                nucleus = ""
        else:
            sfo1 = 0 * ur.Hz
            bf1 = sfo1
            dw = x.spacing
            if isinstance(dw, list):
                print()
            sw = 1 / 2 / dw
            sf = -sw / 2
            size = size // 2

        if not inv:
            # time to frequency
            sizem = max(size - 1, 1)
            deltaf = -sw / sizem
            first = sfo1 - sf - deltaf * sizem / 2.0

            # newcoord = type(x)(np.arange(size) * deltaf + first)
            newcoord = LinearCoord.arange(size) * deltaf + first
            newcoord.show_datapoints = False
            newcoord.name = x.name
            new.title = "intensity"
            if is_nmr:
                newcoord.title = f"${nucleus}$ frequency"
                newcoord.ito("Hz")
            elif is_ir:
                new._units = None
                newcoord.title = "wavenumbers"
                newcoord.ito("cm^-1")
            else:
                newcoord.title = "frequency"
                newcoord.ito("Hz")

        else:
            # frequency or ppm to time
            sw = abs(x.data[-1] - x.data[0])
            if x.units == "ppm":
                sw = bf1.to("Hz") * sw / 1.0e6
            deltat = (1.0 / sw).to("us")

            newcoord = LinearCoord.arange(size) * deltat
            newcoord.name = x.name
            newcoord.title = "time"
            newcoord.ito("us")

        if is_nmr and not inv:
            newcoord.meta.larmor = bf1  # needed for ppm transformation
            ppm = kwargs.get("ppm", True)
            if ppm:
                newcoord.ito("ppm")
                newcoord.title = fr"$\delta\ {nucleus}$"

        new.coordset[dim] = newcoord

        # update history
        s = "ifft" if inv else "fft"
        new.history = f"{s} applied on dimension {dim}"

        # PHASE ?
        iscomplex = new.is_complex or new.is_quaternion
        if iscomplex and not inv:
            # phase frequency domain

            # if some phase related metadata do not exist yet, initialize them
            new.meta.readonly = False

            if not new.meta.phased:
                new.meta.phased = [False] * new.ndim

            if not new.meta.phc0:
                new.meta.phc0 = [0] * new.ndim

            if not new.meta.phc1:
                new.meta.phc1 = [0] * new.ndim

            if not new.meta.exptc:
                new.meta.exptc = [0] * new.ndim

            if not new.meta.pivot:
                new.meta.pivot = [0] * new.ndim

            # applied the stored phases
            new.pk(inplace=True)

            new.meta.pivot[-1] = abs(new).coordmax(dim=dim)
            new.meta.readonly = True

    # restore original data order if it was swaped
    if swaped:
        new.swapdims(axis, -1, inplace=True)  # must be done inplace

    return new


ft = fft
ift = ifft


# Modulus Calculation
@_units_agnostic_method
def mc(dataset):
    """
    Modulus calculation.

    Calculates sqrt(real^2 + imag^2)
    """
    return np.sqrt(dataset.real ** 2 + dataset.imag ** 2)


@_units_agnostic_method
def ps(dataset):
    """
    Power spectrum. Squared version.

    Calculated real^2+imag^2
    """
    return dataset.real ** 2 + dataset.imag ** 2


@_units_agnostic_method
def ht(dataset, N=None):
    """
    Hilbert transform.

    Reconstruct imaginary data via hilbert transform.
    Copied from NMRGlue (BSD3 licence).

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
    # create an empty output array
    fac = N / dataset.shape[-1]
    z = np.empty(dataset.shape, dtype=(dataset.flat[0] + dataset.flat[1] * 1.0j).dtype)
    if dataset.ndim == 1:
        z[:] = hilbert(dataset.real, N)[: dataset.shape[-1]] * fac
    else:
        for i, vec in enumerate(dataset):
            z[i] = hilbert(vec.real, N)[: dataset.shape[-1]] * fac

    # correct the real data as sometimes it changes
    z.real = dataset.real
    return z


# ======================================================================================================================
if __name__ == "__main__":  # pragma: no cover
    pass
