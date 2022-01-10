# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory.
# ======================================================================================================================
"""
Plugin module to perform automatic subtraction of ref on a dataset.
"""
__all__ = ["autosub"]

__dataset_methods__ = __all__

import numpy as np
from scipy.optimize import minimize_scalar


from spectrochempy.core.dataset.coordrange import trim_ranges


def autosub(
    dataset, ref, *ranges, dim="x", method="vardiff", return_coefs=False, inplace=False
):
    """
    Automatic subtraction of a reference to the dataset.

    The subtraction coefficient are adjusted to either
    minimise the variance of the subtraction (method = 'vardiff') which will
    minimize peaks due to ref or minimize the sum of squares of the subtraction
    (method = 'ssdiff').

    Parameters
    ----------
    dataset : |NDDataset|
        Dataset to which we want to subtract the reference data.
    ref : |NDDataset|
         1D reference data, with a size maching the axis to subtract.
         (axis parameter).  # TODO : optionally use title of axis.
    *ranges : pair(s) of values
        Any number of pairs is allowed.
        Coord range(s) in which the variance is minimized.
    dim : `str` or `int`, optional, default='x'
        Tells on which dimension to perform the subtraction.
        If dim is an integer it refers to the axis index.
    method : str, optional, default='vardiff'
        'vardiff': minimize the difference of the variance.
        'ssdiff': minimize the sum of sqares difference of sum of squares.
    return_coefs : `bool`, optional, default=`False`
         Returns the table of coefficients.
    inplace : `bool`, optional, default=`False`
        True if the subtraction is done in place.
        In this case we do not need to catch the function output.

    Returns
    --------
    out : |NDDataset|
        The subtracted dataset.
    coefs : `ndarray`.
        The table of subtraction coeffcients
        (only if `return_coefs` is set to `True`).

    See Also
    --------
    BaselineCorrection : Manual baseline corrections.
    abc : Automatic baseline corrections.

    Examples
    ---------

    >>> path_A = 'irdata/nh4y-activation.spg'
    >>> A = scp.read(path_A, protocol='omnic')
    >>> ref = A[0, :]  # let's subtrack the first row
    >>> B = A.autosub(ref, [3900., 3700.], [1600., 1500.], inplace=False)
    >>> B
    NDDataset: [float64]  a.u. (shape: (y:55, x:5549))
    """

    # output dataset

    if not inplace:
        new = dataset.copy()
    else:
        new = dataset

    # we assume that the last dimension ('x' for transposed array) is always the dimension to which we want
    # to subtract.

    # Swap the axes to be sure to be in this situation
    axis, dim = new.get_axis(dim)

    if axis == new.ndim - 1:
        axis = -1

    try:
        ref.to(dataset.units)
    except Exception:
        raise ValueError("Units of the dataset and reference are not compatible")

    swaped = False
    if axis != -1:
        new = new.swapdims(axis, -1)
        swaped = True

    # TODO: detect the case where the ref is not exactly with same coords: interpolate?

    # selection of the multiple ranges

    # shape = list(new.shape)
    ranges = tuple(np.array(ranges, dtype=float))
    # must be float to be considered as frequency for instance

    coords = new.coordset[dim]
    xrange = trim_ranges(*ranges, reversed=coords.reversed)

    s = []
    r = []

    # TODO: this do not work obviously for axis != -1 - correct this
    for xpair in xrange:
        # determine the slices

        sl = slice(*xpair)
        s.append(dataset[..., sl].data)
        r.append(ref[..., sl].data)

    X_r = np.concatenate((*s,), axis=-1)
    ref_r = np.concatenate((*r,), axis=-1).squeeze()

    indices, _ = list(zip(*np.ndenumerate(X_r[..., 0])))  # .squeeze())))

    # two methods
    # @jit
    def _f(alpha, p):
        if method == "ssdiff":
            return np.sum((p - alpha * ref_r) ** 2)
        elif method == "vardiff":
            return np.var(np.diff(p - alpha * ref_r))
        else:
            raise ValueError("Not implemented for method={}".format(method))

    # @jit(cache=True)
    def _minim():
        # table of subtraction coefficients
        x = []

        for tup in indices:
            # slices = [i for i in tup]
            # slices.append(slice(None))
            # args = (X_r[slices],)
            args = X_r[tup]
            res = minimize_scalar(_f, args=(args,), method="brent")
            x.append(res.x)

        x = np.asarray(x)
        return x

    x = _minim()

    new._data -= np.dot(x.reshape(-1, 1), ref.data.reshape(1, -1))

    if swaped:
        new = new.swapdims(axis, -1)

    new.history = (
        str(new.modified) + ": " + "Automatic subtraction of:" + ref.name + "\n"
    )

    if return_coefs:
        return new, x
    else:
        return new
