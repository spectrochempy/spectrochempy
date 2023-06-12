# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Integration methods.
"""

__all__ = ["simps", "trapz", "simpson", "trapezoid"]

__dataset_methods__ = ["simps", "trapz", "simpson", "trapezoid"]

import functools

import scipy.integrate

from spectrochempy.utils.decorators import deprecated


def _integrate_method(method):
    @functools.wraps(method)
    def wrapper(dataset, *args, **kwargs):

        # handle the various syntax to pass the axis
        if args:
            kwargs["dim"] = args[0]
            args = []

        dim = dataset._get_dims_from_args(*args, **kwargs)
        if dim is None:
            dim = -1
        axis = dataset._get_dims_index(dim)
        axis = axis[0] if axis and not dataset.is_1d else None

        if kwargs.get("dim"):
            kwargs.pop("dim")

        data = method(dataset.data, x=dataset.coord(dim).data, axis=axis, **kwargs)

        if dataset.coord(dim).reversed:
            data *= -1

        new = dataset.copy()
        new._data = data

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
            f"Dataset resulting from application of `{method.__name__}` method"
        ]

        return new

    return wrapper


@_integrate_method
def trapezoid(dataset, **kwargs):
    """
    Integrate using the composite trapezoidal rule.

    Wrapper of `scpy.integrate.trapezoid`\ .

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

    Other Parameters
    ----------------
    dim : `int` or `str`, optional, default: ``"x"``
        Dimension along which to integrate.
        If an integer is provided, it is equivalent to the numpy axis
        parameter for `~numpy.ndarray`\ s.

    See Also
    --------
    trapz : An alias of `trapezoid`.
    simpson : Integrate using the composite simpson rule.

    Example
    -------
    >>> dataset = scp.read('irdata/nh4y-activation.spg')
    >>> dataset[:,1250.:1800.].trapz()
    NDDataset: [float64] a.u..cm^-1 (size: 55)
    """

    return scipy.integrate.trapz(dataset, **kwargs)


@deprecated(replace="Trapezoid")
def trapz(dataset, **kwargs):
    return trapezoid(dataset, **kwargs)


trapz.__doc__ = f"""
    An alias of `trapezoid` kept for backwards compatibility.
{trapezoid.__doc__}"""


@_integrate_method
def simpson(dataset, *args, **kwargs):
    """
    Integrate using the composite Simpson's rule.

    Wrapper of `scpy.integrate.simpson`.

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

    Other Parameters
    ----------------
    dim : `int` or `str`, optional, default: ``"x"``
        Dimension along which to integrate.
        If an integer is provided, it is equivalent to the `numpy.axis` parameter
        for `~numpy.ndarray`\ s.
    even : any of [``'avg'``\ , ``'first'``\ , ``'last'``\ }, optional, default: ``'avg'``

        * ``'avg'`` : Average two results: 1) use the first N-2 intervals with
          a trapezoidal rule on the last interval and 2) use the last
          ``N-2`` intervals with a trapezoidal rule on the first interval.
        * ``'first'`` : Use Simpson's rule for the first ``N-2`` intervals with
          a trapezoidal rule on the last interval.
        * ``'last'`` : Use Simpson's rule for the last ``N-2`` intervals with a
          trapezoidal rule on the first interval.

    See Also
    --------
    simps : An alias of simpson (Deprecated).
    trapezoid : Integrate using the composite simpson rule.

    Example
    --------

    >>> dataset = scp.read('irdata/nh4y-activation.spg')
    >>> dataset[:,1250.:1800.].simps()
    NDDataset: [float64] a.u..cm^-1 (size: 55)
    """
    return scipy.integrate.simps(dataset.data, **kwargs)


@deprecated(replace="simpson")
def simps(dataset, **kwargs):
    return simpson(dataset, **kwargs)


simps.__doc__ = f"""
    An alias of `simpson` kept for backwards compatibility.
{trapezoid.__doc__}"""
