# -*- coding: utf-8 -*-

#  =====================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
#  See full LICENSE agreement in the root directory.
#  =====================================================================================
"""
Integration methods.
"""

__all__ = ["simps", "trapz", "simpson", "trapezoid"]

__dataset_methods__ = ["simps", "trapz", "simpson", "trapezoid"]

import functools
import scipy.integrate
from spectrochempy.utils import deprecated


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
        axis, dim = dataset.get_axis(dim)

        if kwargs.get("dim"):
            kwargs.pop("dim")

        # particular case of datetime64 coord. To simplify the following
        # it is interesting to transform first into a float coordinate with the same units
        # Because only intervals here matters, we can just suppress the first datetiems, and the
        # conversion to na array with the same units will be performed automatically (NDMath module)
        coord = dataset.coord(dim)
        coord = (
            coord - coord[0]
        )  # NOTE that we can't use the inplace assignement in this case

        data = method(dataset.data, x=coord.data, axis=axis, **kwargs)

        if dataset.coord(dim).reversed:
            data *= -1

        new = dataset.copy()
        new.data = data

        # del new._coordset[new._dims[axis]]
        del new._dims[axis]
        if (
            new.implements("NDDataset")
            and new._coordset
            and (dim in new._coordset.names)
        ):
            idx = new._coordset.names.index(dim)
            del new._coordset.coords[idx]

        new.long_name = "area"
        new._units = new._units * coord.units
        new.history = (
            f"Dataset resulting from application of `{method.__name__}` method"
        )

        return new

    return wrapper


@_integrate_method
def trapezoid(dataset, **kwargs):
    """
    Integrate using the composite trapezoidal rule.

    Wrapper of scpy.integrate.trapezoid.

    Performs the integration along the last or given dimension.

    Parameters
    ----------
    dataset : |NDDataset|
        Dataset to be integrated.
    **kwargs
        Additional keywords parameters.
        See Other Parameters.

    Returns
    -------
    integral
        Definite integral as approximated by trapezoidal rule.

    Other Parameters
    ----------------
    dim : int or str, optional, default: "x"
        Dimension along which to integrate.       If an integer is provided, it is equivalent to the `axis` parameter for numpy arrays.

    See Also
    --------
    trapz : An alias of trapezoid.

    simps : Integrate using the composite simpson rule.

    Example
    --------
    >>> dataset = scp.read('irdata/nh4y-activation.spg')
    >>> dataset[:,1250.:1800.].trapz()
    NDDataset: [float64] a.u..cm^-1 (size: 55)
    """

    return scipy.integrate.trapz(dataset, **kwargs)


@deprecated(replace="trapezoid")
def trapz(dataset, **kwargs):
    return trapezoid(dataset, **kwargs)


trapz.__doc__ = f"""
An alias of `trapezoid` kept for backwards compatibility.
{trapezoid.__doc__}"""


@_integrate_method
def simpson(dataset, *args, **kwargs):
    """
    Integrate using the composite Simpson's rule.

    Wrapper of scpy.integrate.trapezoid.

    Performs the integration along the last or given dimension.

    If there are an even number of samples, N, then there are an odd
    number of intervals (N-1), but Simpson's rule requires an even number
    of intervals. The parameter 'even' controls how this is handled.

    Parameters
    ----------
    dataset : |NDDataset|
        Dataset to be integrated.
    **kwargs
        Additional keywords parameters.
        See Other Parameters.

    Returns
    -------
    integral
        Definite integral as approximated using the composite Simpson's rule.

    Other Parameters
    ----------------
    dim : int or str, optional, default: "x"
        Dimension along which to integrate.
        If an integer is provided, it is equivalent to the `axis` parameter for numpy arrays.
    even : str {'avg', 'first', 'last'}, optional, default is 'avg'
        'avg' : Average two results: 1) use the first N-2 intervals with
                  a trapezoidal rule on the last interval and 2) use the last
                  N-2 intervals with a trapezoidal rule on the first interval.
        'first' : Use Simpson's rule for the first N-2 intervals with
                a trapezoidal rule on the last interval.
        'last' : Use Simpson's rule for the last N-2 intervals with a
               trapezoidal rule on the first interval.

    See Also
    --------
    simps : An alias of simpson.
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


simps__doc__ = f"""
An alias of `Simpson` kept for backwards compatibility.
{simpson.__doc__}"""