# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
__all__ = ["fft", "ifft", "rfft", "irfft", "mc", "ps", "ht"]

__dataset_methods__ = __all__

FFT_PREVALIDATION_API = 1

import numpy as np
from scipy.signal import hilbert

from spectrochempy.application.application import error_
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.coordset import CoordSet
from spectrochempy.core.units import ur
from spectrochempy.processing.fft.zero_filling import zf_size
from spectrochempy.utils.decorators import _units_agnostic_method
from spectrochempy.utils.exceptions import SpectroChemPyError


# ======================================================================================
# Private methods
# ======================================================================================
def _fft(data):
    return np.fft.fftshift(np.fft.fft(data), -1)


def _ifft(data, size=None):
    return np.fft.ifft(np.fft.ifftshift(data, -1), n=size)


def _qf_fft(data):
    # FFT transform according to QF encoding
    return np.fft.fftshift(np.fft.fft(np.conjugate(data)), -1)


def _single_interferogram_fft(data):
    """Apply the historical Mertz transform to one interferogram."""

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


def _interferogram_fft(data):
    """Apply the Mertz transform independently to each interferogram."""
    data = np.asarray(data)
    trace_count = int(np.prod(data.shape[:-1])) if data.ndim > 1 else 1

    # Preserve the exact historical numerical path for strict 1D inputs and
    # singleton leading geometry.
    if trace_count == 1:
        return _single_interferogram_fft(data)

    size = data.shape[-1]
    traces = data.reshape(trace_count, size)
    transformed = np.stack(
        [_single_interferogram_fft(trace[np.newaxis, :])[0] for trace in traces]
    )
    return transformed.reshape(*data.shape[:-1], transformed.shape[-1])


def _has_calibrated_interferogram_axis(coord):
    """Return whether *coord* can define an interferogram spectral scale."""
    if coord.units is None:
        return False
    if coord.units.dimensionality == "[time]":
        return True
    return coord.units.dimensionality == "[length]" and "laser_frequency" in coord.meta


# ======================================================================================
# Public methods
# ======================================================================================
def _real_fft_axis(dataset, dim):
    if dataset.ndim == 0 or dataset.is_masked or not np.isfinite(dataset.data).all():
        raise ValueError("Real FFT requires finite, unmasked data.")
    axis, dim = dataset.get_axis(dim, allow_multiple=False)
    if dataset.shape[axis] < 2:
        raise ValueError("Real FFT requires at least two points on the selected axis.")
    coords = dataset.coordset
    coordinate = coords._resolved_axis_coord(dim) if coords is not None else None
    if coordinate is None or coordinate.default is None or coordinate.default.is_empty:
        coordinate = Coord(np.arange(dataset.shape[axis]), title="sample")
    else:
        coordinate = coordinate.default

    values = coordinate._data
    if (
        values is None
        or values.dtype.kind not in "fiu"
        or not np.isfinite(values).all()
    ):
        raise ValueError("Real FFT requires a finite numeric coordinate.")
    if coordinate.meta.get("split_irregular_sampling", False):
        raise ValueError(
            "Within-block samples are not uniformly spaced. Supply an explicit regular coordinate to use a uniform approximation."
        )
    with np.errstate(over="ignore"):
        steps = np.diff(values.astype(float))
    if (
        not np.isfinite(steps).all()
        or steps[0] <= 0
        or not np.allclose(steps, steps[0], rtol=1e-6, atol=0.0)
    ):
        raise ValueError("Real FFT requires a uniformly spaced increasing coordinate.")
    return axis, dim, coordinate, float(steps[0])


def _real_fft_result(dataset, dim, data, coordinate, operation, **parameters):
    result = dataset.copy()
    result._data = data
    coords = dataset.coordset
    result_coords = {dim: coordinate}
    for other in dataset.dims:
        if other == dim:
            continue
        target = coords.references.get(other) if coords is not None else None
        if target in dataset.dims and target != dim:
            result_coords[other] = target
        else:
            original = (
                coords._resolved_axis_coord(other) if coords is not None else None
            )
            result_coords[other] = original.copy() if original is not None else Coord()
    references = {
        key: value for key, value in result_coords.items() if isinstance(value, str)
    }
    dims = [key for key in result_coords if key not in references]
    result._coordset = CoordSet(
        *(result_coords[key] for key in dims), dims=dims, **references
    )
    result._append_history_entry(
        operation=operation,
        parameters={"dim": dim, **parameters},
        message=f"Applied {operation.upper()} on dimension {dim}",
    )
    return result


def rfft(dataset, *, dim=None, norm="backward"):
    """
    Compute the one-sided discrete Fourier transform along a dataset dimension.

    The selected axis must have at least two uniformly spaced, increasing
    samples. An axis without coordinates uses unit sample spacing. The
    first sample defines the phase origin. The output coordinate uses the
    inverse of the sample coordinate unit. The other axes are preserved.
    After a split reshape, the original within-block sampling is checked
    even if the mean coordinate is regular. To use a uniform approximation,
    assign a regular coordinate explicitly before transforming.

    Parameters
    ----------
    dataset : NDDataset
        Real, finite, unmasked samples.
    dim : str or int, optional
        Axis to transform. Defaults to the last dimension.
    norm : {"backward", "forward", "ortho"}, optional
        NumPy FFT normalization. With "forward", divide by the sample count.

    Returns
    -------
    NDDataset
        Complex coefficients from DC up to the highest nonnegative frequency.
    """
    if dataset.is_empty or dataset.data.dtype.kind not in "fiub":
        raise ValueError("rfft requires nonempty real input.")
    axis, dim, source, step = _real_fft_axis(dataset, dim)
    size = dataset.shape[axis]
    data = np.fft.rfft(dataset.data, axis=axis, norm=norm)
    with np.errstate(over="ignore"):
        frequencies = np.fft.rfftfreq(size) / step
    if not np.isfinite(frequencies).all() or not np.all(np.diff(frequencies) > 0):
        raise ValueError(
            "The frequency grid cannot be represented in the coordinate units."
        )
    frequency = Coord(
        frequencies,
        title=f"frequency of {source.title}",
        units=None if source.units is None else source.units**-1,
        rounding=False,
        sigdigits=15,
    )
    frequency.meta["rfft_source"] = {
        "size": int(size),
        "origin": float(source._data[0]),
        "units": None if source.units is None else str(source.units),
        "title": source.title,
        "norm": norm,
    }
    return _real_fft_result(dataset, dim, data, frequency, "rfft", norm=norm)


def irfft(dataset, *, dim=None, n=None, norm=None):
    """
    Invert a one-sided real Fourier transform along a dataset dimension.

    Parameters
    ----------
    dataset : NDDataset
        Finite, unmasked coefficients on an increasing frequency grid from DC.
    dim : str or int, optional
        Axis to invert. Defaults to the last dimension.
    n : int, optional
        Output length. Uses the original length recorded by rfft when
        available, otherwise assumes an even length. For externally supplied
        coefficients from an odd-length signal, give the original length.
    norm : {"backward", "forward", "ortho"}, optional
        Uses the normalization recorded by rfft, or "backward" if absent.

    Returns
    -------
    NDDataset
        Reconstructed real samples with coordinates derived from the frequency
        spacing and the original origin, if recorded.
    """
    if dataset.is_empty or dataset.data.dtype.kind not in "cfiub":
        raise ValueError(
            "irfft requires nonempty real or complex Fourier coefficients."
        )
    axis, dim, frequency, step = _real_fft_axis(dataset, dim)
    if frequency._data[0] != 0:
        raise ValueError("irfft requires a frequency coordinate starting at zero.")
    source = frequency.meta.get("rfft_source", {})
    bins = dataset.shape[axis]
    if n is None:
        n = source.get("size", 2 * (bins - 1))
    if not isinstance(n, (int, np.integer)) or n < 2 or n // 2 + 1 != bins:
        raise ValueError(
            "n must be an original sample count compatible with the spectrum."
        )
    if norm is None:
        norm = source.get("norm", "backward")
    data = np.fft.irfft(dataset.data, n=n, axis=axis, norm=norm)
    units = None if frequency.units is None else frequency.units**-1
    origin = source.get("origin", 0.0)
    if source.get("units") is not None and units is not None:
        origin = ur.Quantity(origin, source["units"]).to(units).magnitude
    with np.errstate(over="ignore"):
        values = origin + (np.arange(n) / n) / step
    if not np.isfinite(values).all() or not np.all(np.diff(values) > 0):
        raise ValueError(
            "The sample grid cannot be represented in the coordinate units."
        )
    coordinate = Coord(
        values,
        title=source.get("title", "sample"),
        units=units,
        rounding=False,
        sigdigits=15,
    )
    return _real_fft_result(
        dataset, dim, data, coordinate, "irfft", n=int(n), norm=norm
    )


def ifft(dataset, size=None, **kwargs):
    """
    Apply a inverse fast fourier transform.

    For a multidimensional dataset, transform the last dimension by default.

    The data in the last dimension MUST be in frequency (or without dimension)
    or an error is raised.

    To make direct Fourier transform, i.e., from frequency to time domain, use the `fft` transform.

    Parameters
    ----------
    dataset : `NDDataset`
        The dataset on which to apply the fft transformation.
    size : int, optional
        Size of the transformed dataset dimension; the shorter alias is `si`.
        By default, the input size is used. If `size` is larger or smaller than
        the input, the unshifted frequency data are padded with zeros or
        truncated, respectively, before the inverse transform.
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
    Apply a complex fast Fourier transform.

    Apply apodization before this operation and phase correction with
    ``pk()`` afterwards when needed.

    For a multidimensional dataset, transform the last dimension by default.

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
        If True, an inverse Fourier transform is performed. The `size`
        parameter controls its output size as described for `ifft`.
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
    dim = str(dim)

    domain_state = getattr(dataset.meta, "isfreq", None)
    if domain_state is not None:
        meta_axis = dataset.dims.index(dim)
        if meta_axis < len(domain_state):
            is_frequency = domain_state[meta_axis]
            if is_frequency is not None and not inv and bool(is_frequency):
                raise SpectroChemPyError(
                    f"Cannot apply FFT on dimension {dim!r} because metadata "
                    "marks it as already in the frequency domain. Use ifft() "
                    "for a frequency-to-time transform, or choose a time-domain "
                    "dimension."
                )
            if is_frequency is not None and inv and not bool(is_frequency):
                raise SpectroChemPyError(
                    f"Cannot apply IFFT on dimension {dim!r} because metadata "
                    "marks it as still in the time domain. Use fft() for a "
                    "time-to-frequency transform, or choose a frequency-domain "
                    "dimension."
                )

    from spectrochempy.plugins import manager as manager_module  # noqa: PLC0415

    validate = manager_module.plugin_manager.registry.get_handler("fft.validate_input")
    if validate is not None:
        validate(dataset, dim=dim, inv=inv, size=size, sizeff=sizeff, **kwargs)

    # Validate the original target coordinate before an in-place swap can
    # mutate the caller. Interferogram FFTs build a physical wavenumber axis,
    # so indices and dimensionless coordinates are insufficient; calibrated
    # time and optical path difference coordinates are both valid.
    target_coord = dataset.coordset[dim]
    if is_ir and not inv and not _has_calibrated_interferogram_axis(target_coord):
        raise SpectroChemPyError(
            "Fourier transforming an interferogram requires a calibrated "
            "time or optical path difference axis, from which a wavenumber "
            f"axis can be built. The interferogram axis here is "
            f"{target_coord.title!r} with units {target_coord.units}. Call "
            "Coord.set_laser_frequency() with the value of your instrument "
            "first, or convert the axis yourself. The dataset was left "
            "untouched."
        )

    # output dataset inplace or not
    inplace = kwargs.pop("inplace", False)
    new = dataset.copy() if not inplace else dataset

    encoding = "undefined"
    if not inv and "encoding" in new.meta:
        encoding = new.meta.encoding[axis]

    # The last dimension is always the dimension on which we apply the fourier transform.
    # If needed, we swap the dimensions to be sure to be in this situation
    swapped = False
    if axis != -1:
        new._swapdims_without_history(axis, -1, inplace=True)
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
        and not (is_ir and x.units.dimensionality == "[length]")
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

    is_ir_opd = (
        is_ir
        and not inv
        and x.units is not None
        and x.units.dimensionality == "[length]"
    )
    if hasattr(x, "_use_time_axis") and not is_ir_opd:
        x._use_time_axis = True  # we need to have dimensionless or time units

    if not error:
        # OK we can proceed

        # if no size (or si) parameter then use the size of the data
        if size is None:
            size = kwargs.get("si", x.size)

        if not inv:
            # do we have an effective td to apply
            tdeff = sizeff
            if tdeff is None:
                tdeff = kwargs.get("tdeff", x.size)

            if tdeff is None or tdeff < 5 or tdeff > size:
                tdeff = size

            # Eventually apply the effective size
            new[..., tdeff:] = 0.0

        # Determine whether the data are complex (or plugin-specific interleaved)
        # interleaved is in case of >2D data  ( # TODO: >D not yet implemented in ndcomplex.py
        iscomplex = new.is_complex
        if new.is_interleaved:
            iscomplex = True

        if not inv:
            zf_size(new, size=size, inplace=True)
            if is_ir_opd:
                # zf_size currently normalizes interferogram coordinates to
                # time. Restore the explicitly calibrated OPD grid so the
                # reciprocal-length FFT scale uses its physical spacing.
                x._use_time_axis = False

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
            data = _ifft(new.data, size=size)

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
            if is_ir:
                kept_frequencies = np.fft.rfftfreq(size)[:coord_size][::-1]
                newcoord = Coord((kept_frequencies / abs(dw)).to("cm^-1"))
            else:
                shifted_frequencies = np.fft.fftshift(np.fft.fftfreq(coord_size))
                frequencies = shifted_frequencies / dw
                if x.unitless or x.dimensionless:
                    newcoord = Coord(frequencies)
                    newcoord.ito("Hz")
                else:
                    newcoord = Coord(frequencies.to("Hz"))

            newcoord.show_datapoints = False
            newcoord.name = x.name
            new.title = "intensity"
            if is_ir:
                new._units = None
                newcoord.title = "wavenumbers"
            else:
                newcoord.title = "frequency"

        else:
            # frequency to time
            if x.units is not None and x.units.dimensionality == "1/[time]":
                # With a preserved frequency-bin spacing, the reciprocal time
                # step is 1 / (size * df) for the requested output size.
                deltat = (1.0 / (size * abs(x.spacing))).to("us")
            else:
                # For ppm or dimensionless coordinates we cannot determine the
                # correct time step without extra context.  Use a placeholder
                # so that plugins (e.g. NMR) can replace the coordinate.
                sw = abs(x.data[-1] - x.data[0])
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
        new.history = f"Applied {s.upper()} on dimension {dim}"

        if (new.is_complex or new.is_interleaved) and not inv:
            # Start explicit phasing from zero on the transformed axis.
            new.meta.readonly = False
            for name, value in (
                ("phased", True),
                ("phc0", 0 * ur.degree),
                ("phc1", 0 * ur.degree),
                ("exptc", 0),
                ("pivot", 0),
                ("physical_pivots", False),
            ):
                values = list(new.meta.get(name) or [0] * new.ndim)
                values[-1] = value
                new.meta[name] = values

            new.meta.readonly = True

    # restore original data order if it was swapped
    if swapped:
        new._swapdims_without_history(axis, -1, inplace=True)

    return new


ft = fft
ift = ifft


def _normalize_ht_parameters(kwargs, *, axis_size):
    """Resolve and validate Hilbert-transform parameters before data mutation."""
    parameters = dict(kwargs)
    n = parameters.get("N")
    if n is None:
        n = axis_size
    elif isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)):
        raise TypeError("N must be None or a Python/NumPy integer, excluding booleans.")
    else:
        n = int(n)

    if n <= 0:
        raise ValueError("N must be a positive integer.")
    if n < axis_size:
        raise ValueError(
            "N must be greater than or equal to the selected dimension length "
            f"({axis_size})."
        )

    parameters["N"] = n
    return parameters


# Modulus Calculation
@_units_agnostic_method(
    structured_history=True,
    history_description="modulus calculated",
)
def mc(dataset):
    """
    Modulus calculation.

    Calculates sqrt(real^2 + imag^2)
    """
    return np.sqrt(dataset.real**2 + dataset.imag**2)


@_units_agnostic_method(
    structured_history=True,
    history_description="power spectrum calculated",
)
def ps(dataset):
    """
    Power spectrum. Squared version.

    Calculated real^2+imag^2
    """
    return dataset.real**2 + dataset.imag**2


@_units_agnostic_method(
    structured_history=True,
    history_description="Hilbert transform performed",
    parameter_preprocessor=_normalize_ht_parameters,
)
def ht(dataset, N=None):
    """
    Hilbert transform.

    Reconstruct imaginary data via hilbert transform.
    Copied from NMRGlue (BSD3 licence).

    Parameters
    ----------
    dataset : array-like
        Real or complex NMR data. The Hilbert transform is applied along the
        selected dimension, which is the last dimension by default.
    N : int or None, optional
        Number of Fourier components passed to the Hilbert transform. ``None``
        uses the selected dimension length. Values smaller than that length are
        not supported.

    Returns
    -------
    ndarray
        Complex NMR data reconstructed by the Hilbert transform.

    """
    source_size = dataset.shape[-1]
    fac = N / source_size
    transformed = hilbert(dataset.real, N, axis=-1)[..., :source_size] * fac
    sample = dataset.flat[0]
    output_dtype = (sample + sample * 1.0j).dtype
    z = np.asarray(transformed, dtype=output_dtype)

    # correct the real data as sometimes it changes
    z.real = dataset.real
    return z
