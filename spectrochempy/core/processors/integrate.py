# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================
"""
Integration methods
"""

__all__ = ["simps", "trapz"]

__dataset_methods__ = ["simps", "trapz"]

import scipy.integrate


def trapz(dataset, *args, **kwargs):
    """
    Wrapper of scpy.integrate.trapz() : Integrate along the given dimension using the composite trapezoidal rule.

    Integrate NDDataset along given dimension.

    Parameters
    ----------
    dataset : |NDDataset|
        Dataset to be integrated.
    dim : str, optional
        Dimension along which to integrate. Default is the dimension corresponding to the last axis, generally 'x'.
    axis : int, optional
        When dim is not used, this is the axis along which to integrate. Default is the last axis.

    Returns
    -------
    trapz
        Definite integral as approximated by trapezoidal rule.

    Example
    --------
    >>> dataset = scp.read('irdata/nh4y-activation.spg')
    >>> dataset[:,1250.:1800.].trapz()
    NDDataset: [float64] a.u..cm^-1 (size: 55)
    """

    # handle the various syntax to pass the axis
    if args:
        kwargs["dim"] = args[0]
        args = []

    dim = dataset._get_dims_from_args(*args, **kwargs)
    if dim is None:
        dim = -1
    axis = dataset._get_dims_index(dim)
    axis = axis[0] if axis and not dataset.is_1d else None

    data = scipy.integrate.trapz(dataset.data, x=dataset.coord(dim).data, axis=axis)
    if dataset.coord(dim).reversed:
        data *= -1

    new = dataset.copy()
    new._data = data

    del new._dims[axis]
    if new.implements("NDDataset") and new._coordset and (dim in new._coordset.names):
        idx = new._coordset.names.index(dim)
        del new._coordset.coords[idx]

    new.title = "area"
    new._units = dataset.units * dataset.coord(dim).units
    new._history = ["Dataset resulting from application of `trapz` method"]

    return new


def simps(dataset, *args, **kwargs):
    """
    Wrapper of scpy.integrate.simps().

    Integrate y(x) using samples along the given axis and the composite
    Simpson's rule. If x is None, spacing of dx is assumed.

    If there are an even number of samples, N, then there are an odd
    number of intervals (N-1), but Simpson's rule requires an even number
    of intervals. The parameter 'even' controls how this is handled.

    Parameters
    ----------
    dataset : |NDDataset|
        dataset to be integrated.
    dim : str, optional
        Dimension along which to integrate. Default is the dimension corresponding to the last axis, generally 'x'.
    axis : int, optional
        When dim is not used, this is the axis along which to integrate. Default is the last axis.
    even : str {'avg', 'first', 'last'}, optional, default is 'avg'
        'avg' : Average two results: 1) use the first N-2 intervals with
                  a trapezoidal rule on the last interval and 2) use the last
                  N-2 intervals with a trapezoidal rule on the first interval.
        'first' : Use Simpson's rule for the first N-2 intervals with
                a trapezoidal rule on the last interval.
        'last' : Use Simpson's rule for the last N-2 intervals with a
               trapezoidal rule on the first interval.

    Returns
    -------
    simps
        Definite integral as approximated using the composite Simpson's rule.

    Example
    --------

    >>> dataset = scp.read('irdata/nh4y-activation.spg')
    >>> dataset[:,1250.:1800.].simps()
    NDDataset: [float64] a.u..cm^-1 (size: 55)
    """

    # handle the various syntax to pass the axis
    if args:
        kwargs["dim"] = args[0]
        args = []

    dim = dataset._get_dims_from_args(*args, **kwargs)
    if dim is None:
        dim = -1
    axis = dataset._get_dims_index(dim)
    axis = axis[0] if axis and not dataset.is_1d else None

    data = scipy.integrate.simps(
        dataset.data,
        x=dataset.coord(dim).data,
        axis=axis,
        even=kwargs.get("even", "avg"),
    )
    if dataset.coord(dim).reversed:
        data *= -1

    new = dataset.copy()
    new._data = data

    del new._dims[axis]
    if new.implements("NDDataset") and new._coordset and (dim in new._coordset.names):
        idx = new._coordset.names.index(dim)
        del new._coordset.coords[idx]

    new.title = "area"
    new._units = dataset.units * dataset.coord(dim).units
    new._history = ["Dataset resulting from application of `simps` method"]

    return new
