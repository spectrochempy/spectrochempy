# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Integration methods."""

__all__ = ["simpson", "trapezoid"]

__dataset_methods__ = ["simpson", "trapezoid"]

import functools

import numpy as np
import scipy.integrate

from spectrochempy.utils.constants import NOMASK


def _reduced_slice_mask(mask, axis):
    """
    Reduce a source mask along the integrated axis.

    A masked point is a scientific exclusion, so an output slice is considered
    valid only if it was built without any masked point. The canonical unmasked
    sentinel is returned unchanged and whenever no slice is affected, so an
    unmasked input keeps the canonical unmasked representation.
    """
    if not isinstance(mask, np.ndarray):
        # NOMASK, and any other scalar sentinel, means nothing is excluded.
        return NOMASK
    reduced = np.any(mask, axis=axis)
    return reduced if np.any(reduced) else NOMASK


def _exclude_incomplete_slices(data, slice_mask, axis):
    """
    Blank the incomplete slices before the numerical integration.

    The values hidden under a mask are excluded from the calculation, so they
    must neither reach the quadrature nor overflow it. They are replaced by
    zeros, which only affects the incomplete slices whose result is discarded
    afterwards. Complete slices keep their exact values.
    """
    if not np.any(slice_mask):
        return data
    if np.ndim(slice_mask) > 0:
        slice_mask = np.expand_dims(slice_mask, axis)
    return np.where(slice_mask, 0.0, data)


def _integrate_method(method):
    @functools.wraps(method)
    def wrapper(dataset, *args, **kwargs):
        # handle the various syntax to pass the axis
        if args:
            kwargs["dim"] = args[0]
            args = []

        axis, dim = dataset.get_axis(**kwargs)

        if kwargs.get("dim"):
            kwargs.pop("dim")

        # SciPy integration routines expect a plain ndarray-like coordinate.
        # Some NumPy/SciPy combinations are stricter with ndarray subclasses or
        # view semantics, so normalize the integration axis coordinate here.
        x = np.asarray(dataset.coord(dim).data)
        y = np.asarray(dataset.data)

        # A masked point is a scientific exclusion. Reduce the source mask along
        # the integrated axis first, so that the validity of each output slice
        # is known before anything is computed.
        slice_mask = _reduced_slice_mask(dataset.mask, axis)
        y = _exclude_incomplete_slices(y, slice_mask, axis)

        try:
            data = method(y, x=x, axis=axis, **kwargs)
        except NotImplementedError as exc:
            if "multi-dimensional sub-views are not implemented" not in str(exc):
                raise
            data = method(
                np.ascontiguousarray(y),
                x=np.ascontiguousarray(x),
                axis=axis,
                **kwargs,
            )

        if dataset.coord(dim).reversed:
            data *= -1

        if np.any(slice_mask):
            # A slice whose contribution is incomplete has no published area.
            # Publish it as unavailable instead of a number derived from
            # excluded points.
            data = np.where(slice_mask, np.nan, data)

        new = dataset.copy()
        new._data = data
        new._mask = slice_mask

        del new._dims[axis]
        if (
            new._implements("NDDataset")
            and new._coordset
            and (dim in new._coordset.names)
        ):
            idx = new._coordset.names.index(dim)
            del new._coordset.coords[idx]

        new.title = "area"
        new.description = (
            f"Integration of NDDataset '{dataset.name}' along dim: '{dim}'."
        )

        if dataset.units is not None and dataset.coord(dim).units is not None:
            new._units = dataset.units * dataset.coord(dim).units
        elif dataset.units is not None:
            new._units = dataset.units
        elif dataset.coord(dim).units is not None:
            new._units = dataset.coord(dim).units
        new.history = [
            f"Dataset resulting from application of `{method.__name__}` method",
        ]

        return new

    return wrapper


@_integrate_method
def trapezoid(dataset, **kwargs):
    r"""
    Integrate using the composite trapezoidal rule.

    Wrapper of ``scipy.integrate.trapezoid``.

    Performs the integration along the last or given dimension.

    Parameters
    ----------
    dataset : `NDDataset`
        Dataset to be integrated.
    **kwargs
        Additional keywords parameters. See Other Parameters.

    Returns
    -------
    `~spectrochempy.core.dataset.ndataset.NDDataset`
        Definite integral as approximated by trapezoidal rule.

    Notes
    -----
    SpectroChemPy does not publish an integral for a slice whose contribution is
    incomplete. A masked point is a scientific exclusion: an output slice is
    integrated normally, and is not masked, when it was built without any masked
    point. A slice that used at least one masked point, including a fully masked
    slice, is instead published as a masked value with a raw `numpy.nan`.

    No estimate of the missing area is made, so masked points are not replaced,
    removed or interpolated. The values hidden under the mask never reach the
    quadrature, so they cannot influence a result nor overflow it.

    The result mask is always compatible with the result shape: a 1D input
    yields a zero-dimensional result with a scalar mask, and an unmasked input
    yields the canonical unmasked `numpy.False_` mask.

    Other Parameters
    ----------------
    dim : `int` or `str`, optional, default: ``"x"``
        Dimension along which to integrate.
        If an integer is provided, it is equivalent to the numpy axis
        parameter for ``numpy.ndarray``.

    See Also
    --------
    simpson : Integrate using the composite simpson rule.

    Example
    -------
    >>> dataset = scp.read('irdata/nh4y-activation.spg')
    >>> dataset[:,1250.:1800.].trapezoid()
    NDDataset: [float64] a.u..cm^-1 (size: 55)

    """
    return scipy.integrate.trapezoid(np.asarray(dataset), **kwargs)


@_integrate_method
def simpson(dataset, *args, **kwargs):
    r"""
    Integrate using the composite Simpson's rule.

    Wrapper of ``scipy.integrate.simpson``.

    Performs the integration along the last or given dimension.

    If there are an even number of samples, ``N``, then there are an odd
    number of intervals (``N-1``), but Simpson's rule requires an even number
    of intervals. The parameter 'even' controls how this is handled.

    Parameters
    ----------
    dataset : `NDDataset`
        Dataset to be integrated.
    **kwargs
        Additional keywords parameters. See Other Parameters.

    Returns
    -------
    `~spectrochempy.core.dataset.ndataset.NDDataset`
        Definite integral as approximated using the composite Simpson's rule.

    Notes
    -----
    SpectroChemPy does not publish an integral for a slice whose contribution is
    incomplete. A masked point is a scientific exclusion: an output slice is
    integrated normally, and is not masked, when it was built without any masked
    point. A slice that used at least one masked point, including a fully masked
    slice, is instead published as a masked value with a raw `numpy.nan`.

    No estimate of the missing area is made, so masked points are not replaced,
    removed or interpolated. The values hidden under the mask never reach the
    quadrature, so they cannot influence a result nor overflow it.

    The result mask is always compatible with the result shape: a 1D input
    yields a zero-dimensional result with a scalar mask, and an unmasked input
    yields the canonical unmasked `numpy.False_` mask.

    Other Parameters
    ----------------
    dim : `int` or `str`, optional, default: ``"x"``
        Dimension along which to integrate.
        If an integer is provided, it is equivalent to the ``numpy.axis`` parameter
        for ``numpy.ndarray``.
    even : {``'avg'``, ``'first'``, ``'last'``}, optional, default: ``'avg'``

        * ``'avg'`` : Average two results: 1) use the first N-2 intervals with
          a trapezoidal rule on the last interval and 2) use the last
          ``N-2`` intervals with a trapezoidal rule on the first interval.
        * ``'first'`` : Use Simpson's rule for the first ``N-2`` intervals with
          a trapezoidal rule on the last interval.
        * ``'last'`` : Use Simpson's rule for the last ``N-2`` intervals with a
          trapezoidal rule on the first interval.

    See Also
    --------
    trapezoid : Integrate using the composite simpson rule.

    Example
    --------

    >>> dataset = scp.read('irdata/nh4y-activation.spg')
    >>> dataset[:,1250.:1800.].simpson()
    NDDataset: [float64] a.u..cm^-1 (size: 55)

    """
    return scipy.integrate.simpson(np.asarray(dataset), **kwargs)
