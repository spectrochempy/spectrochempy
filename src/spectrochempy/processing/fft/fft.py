# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
__all__ = ["fft", "ifft", "mc", "ps", "ht"]

__dataset_methods__ = __all__

import numpy as np
from scipy.signal import hilbert

from spectrochempy.application.application import error_
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.units import ur
from spectrochempy.processing.fft.zero_filling import zf_size
from spectrochempy.utils.decorators import _units_agnostic_method


# ======================================================================================
# Private methods
# ======================================================================================
def _fft(data):
    return np.fft.fftshift(np.fft.fft(data), -1)


def _ifft(data):
    return np.fft.ifft(np.fft.ifftshift(data, -1))


def _qf_fft(data):
    # FFT transform according to QF encoding
    return np.fft.fftshift(np.fft.fft(np.conjugate(data)), -1)


def _interferogram_fft(data):
    """FFT transform for rapid-scan interferograms. Phase corrected using the Mertz method."""

    def _get_zpd(data, mode="max"):
        if mode == "max":
            return np.argmax(data, -1).item()
        if mode == "abs":
            return int(np.argmax(np.abs(data), -1).item())
        return None

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


# ======================================================================================
# Public methods
# ======================================================================================
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
    dataset : `NDDataset`
        The dataset on which to apply the fft transformation.
    size : int, optional
        Size of the transformed dataset dimension - a shorter parameter is `si` . by default, the size is the closest
        power of two greater than the data size.
    **kwargs
        Optional keyword parameters (see Other Parameters).

    Returns
    -------
    out
        Transformed `NDDataset` .

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


def fft(dataset, size=None, sizeff=None, inv=False, **kwargs):
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
    dataset : `NDDataset`
        The dataset on which to apply the fft transformation.
    size : int, optional
        Size of the transformed dataset dimension - a shorter parameter is `si` .
        By default, the size is the data size.
    sizeff : int, optional
        The number of effective data point to take into account for the transformation. By default it is equal to the
        data size, but may be smaller.
    inv : bool, optional, default=False
        If True, an inverse Fourier transform is performed - size parameter is not taken into account.
    **kwargs
        Optional keyword parameters (see Other Parameters).

    Returns
    -------
    out
        Transformed `NDDataset` .

    Other Parameters
    ----------------
    dim : str or int, optional, default='x'.
        Specify on which dimension to apply this method. If `dim` is specified as an integer it is equivalent
        to the usual `axis` numpy parameter.
    inplace : bool, optional, default=False.
        True if we make the transform inplace.  If False, the function return a new object
    tdeff : int, optional
        Alias of sizeff. If both sizeff and tdeff are passed, sizeff has the priority.

    See Also
    --------
    ifft : Inverse Fourier transform.

    """
    is_ir = dataset.meta.interferogram

    # On which axis do we want to apply transform (get axis from arguments)
    dim = kwargs.pop("dim", kwargs.pop("axis", -1))
    axis, dim = dataset.get_axis(dim, negative_axis=True)

    # output dataset inplace or not
    inplace = kwargs.pop("inplace", False)
    new = dataset.copy() if not inplace else dataset

    # Capture the encoding for the target axis BEFORE any swap.
    # swapdims reorders meta.encoding, so encoding[0] would point to the
    # wrong encoding after a swap.  For quaternion (hypercomplex) data the
    # encoding at index 0 always describes the indirect dimension and knows
    # how to decompose quaternion → complex subspectra.
    encoding = "undefined"
    if not inv and "encoding" in new.meta:
        encoding = new.meta.encoding[axis]

    # The last dimension is always the dimension on which we apply the fourier transform.
    # If needed, we swap the dimensions to be sure to be in this situation
    swapped = False
    if axis != -1:
        new.swapdims(axis, -1, inplace=True)  # must be done in  place
        swapped = True

    # Select the last coordinates
    x = new.coordset[dim]

    # Performs some dimensionality checking
    error = False
    if (
        not inv
        and not x.unitless
        and not x.dimensionless
        and x.units.dimensionality != "[time]"
    ):
        error_(
            Exception,
            "fft apply only to dimensions with [time] dimensionality or dimensionless coords\n"
            "fft processing was thus cancelled",
        )
        error = True

    elif (
        inv
        and not x.unitless
        and x.units.dimensionality != "1/[time]"
        and not x.dimensionless
    ):
        error_(
            Exception,
            "ifft apply only to dimensions with [frequency] dimensionality or with ppm units "
            "or dimensionless coords.\n ifft processing was thus cancelled",
        )
        error = True

    # Should not be masked
    elif new.is_masked:
        error_(
            Exception,
            "current fft or ifft processing does not support masked data as input.\n processing was thus cancelled",
        )
        error = True

    # Coordinates should be uniformly spaced (linear coordinate)
    if not x.linear:
        error_(
            "fft or ifft processing only support linear coordinates.\n"
            "Processing was thus cancelled",
        )

        error = True

    if hasattr(x, "_use_time_axis"):
        x._use_time_axis = True  # we need to have dimentionless or time units

    if not error:
        # OK we can proceed

        # time domain size
        td = None
        if not inv:
            td = x.size

        # if no size (or si) parameter then use the size of the data
        # (size not used for inverse transform
        if size is None or inv:
            size = kwargs.get("si", x.size)

        # do we have an effective td to apply
        tdeff = sizeff
        if tdeff is None:
            tdeff = kwargs.get("tdeff", td)

        if tdeff is None or tdeff < 5 or tdeff > size:
            tdeff = size

        # Eventually apply the effective size
        new[..., tdeff:] = 0.0

        # Determine whether the data are complex (or plugin-specific interleaved)
        # interleaved is in case of >2D data  ( # TODO: >D not yet implemented in ndcomplex.py
        iscomplex = False
        if axis == -1:
            iscomplex = new.is_complex
        if new.is_interleaved:
            iscomplex = True

        zf_size(new, size=size, inplace=True)

        # Perform the fft
        if encoding != "undefined":
            try:
                from spectrochempy.plugins import (
                    manager as manager_module,  # noqa: PLC0415
                )

                handler = manager_module.plugin_manager.registry.get_handler(
                    "fft.encoding"
                )
            except Exception:  # noqa: BLE001
                handler = None
            if handler is None:
                raise NotImplementedError(
                    f"FFT encoding {encoding!r} requires a plugin. "
                    "Install the relevant plugin (e.g. spectrochempy-nmr)."
                )
            data = handler(new.data, encoding, original_axis=axis, **kwargs)

        elif iscomplex and inv:
            # We assume no special encoding for inverse complex fft transform
            data = _ifft(new.data)

        elif not iscomplex and not inv and is_ir:
            # transform interferogram
            data = _interferogram_fft(new.data)

        elif not iscomplex and inv:
            raise NotImplementedError("Inverse FFT for real dimension")

        else:
            data = _fft(new.data)

        # We need here to create a new dataset with new shape and axis
        new._data = data
        new.mask = False

        # Determine the coordinate size for the output
        coord_size = size
        if not inv and is_ir:
            # interferogram FFT yields half the number of points
            coord_size = size // 2

        # create new coordinates for the transformed data
        if not inv:
            # time to frequency
            dw = x.spacing
            if isinstance(dw, list):
                pass  # print()
            sw = 1 / 2 / dw
            sf = -sw / 2

            sizem = max(coord_size - 1, 1)
            deltaf = -sw / sizem
            first = sf - deltaf * sizem / 2.0

            newcoord = Coord.arange(coord_size) * deltaf + first
            newcoord.show_datapoints = False
            newcoord.name = x.name
            new.title = "intensity"
            if is_ir:
                new._units = None
                newcoord.title = "wavenumbers"
                newcoord.ito("cm^-1")
            else:
                newcoord.title = "frequency"
                newcoord.ito("Hz")

        else:
            # frequency to time
            sw = abs(x.data[-1] - x.data[0])
            # sw is a plain float here (x.data is an ndarray).  Multiply by the
            # original coordinate unit so that 1/sw has time dimensionality.
            if x.units is not None and x.units.dimensionality == "1/[time]":
                deltat = (1.0 / (sw * x.units)).to("us")
            else:
                # For ppm or dimensionless coordinates we cannot determine the
                # correct time step without extra context.  Use a placeholder
                # so that plugins (e.g. NMR) can replace the coordinate.
                deltat = (1.0 / sw) * ur.us

            newcoord = Coord.arange(coord_size) * deltat
            newcoord.name = x.name
            newcoord.title = "time"
            newcoord.ito("us")

        new.coordset[dim] = newcoord

        # Allow plugins to post-process the result (e.g. NMR axis labels, ppm conversion)
        try:
            from spectrochempy.plugins import manager as manager_module  # noqa: PLC0415

            post_handler = manager_module.plugin_manager.registry.get_handler(
                "fft.postprocess_result"
            )
        except Exception:  # noqa: BLE001
            post_handler = None
        if post_handler is not None:
            new = post_handler(new, dim=dim, inv=inv, **kwargs)

        if getattr(new.meta, "isfreq", None) is not None:
            meta_dim_index = new.dims.index(dim) if isinstance(dim, str) else axis
            new.meta.isfreq[meta_dim_index] = not inv

        # update history
        s = "ifft" if inv else "fft"
        new.history = f"{s} applied on dimension {dim}"

        # PHASE ?
        iscomplex = new.is_complex or new.is_interleaved
        # Quaternion/hypercomplex data from the encoding handler is in the
        # frequency domain but is_complex is False.  Treat it as phaseable
        # when the encoding handler ran (encoding != "undefined").
        if not iscomplex and encoding != "undefined":
            iscomplex = True
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

            # Apply auto-phasing for complex and quaternion data.
            # Quaternion data is handled by the plugin-provided pk.execute
            # handler which decomposes → phases subspectra → rebuilds.
            new.pk(inplace=True)
            if new.is_complex:
                new.meta.pivot[-1] = abs(new).coordmax(dim=dim)
            else:
                # Quaternion: compute pivot from quaternion modulus
                import quaternion as _quat  # noqa: PLC0415

                _qarr = _quat.as_float_array(new.data)
                _mod = np.sqrt(np.sum(_qarr**2, axis=-1))
                _mod_ds = new.copy()
                _mod_ds.data = _mod
                new.meta.pivot[-1] = _mod_ds.coordmax(dim=dim)

            new.meta.readonly = True

    # restore original data order if it was swapped
    if swapped:
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
    return np.sqrt(dataset.real**2 + dataset.imag**2)


@_units_agnostic_method
def ps(dataset):
    """
    Power spectrum. Squared version.

    Calculated real^2+imag^2
    """
    return dataset.real**2 + dataset.imag**2


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
