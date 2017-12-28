# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================

# TODO: create tests

__all__ = ['Lstsq']


import numpy as np

from spectrochempy.dataset.nddataset import NDDataset


def Lstsq(A, B, rcond=-1):
    """
    Return the least-squares solution to a linear matrix equation.

    This is an extension of :meth:`numpy.linalg.lstsq` to |NDDatasets|.

    Solves the equation `A X = B` by computing a vector `X` that
    minimizes the Euclidean 2-norm `|| B - A X ||^2`.  The equation may
    be under-, well-, or over- determined (*i.e.*, the number of
    linearly independent rows of `A` can be less than, equal to, or
    greater than its number of linearly independent columns).  If `A`
    is square and of full rank, then `X` (but for round-off error) is
    the "exact" solution of the equation.

    Parameters
    ----------
    A : (M, N) |NDDataset| or array-like
        "Coefficient" matrix.
    B : {(M,), (M, K)} |NDDataset| or array_like
        Ordinate or "dependent variable" values. If `B` is two-dimensional,
        the least-squares solution is calculated for each of the `K` columns
        of `B`.
    rcond : float, optional
        Cut-off ratio for small singular values of `A`.
        For the purposes of rank determination, singular values are treated
        as zero if they are smaller than `rcond` times the largest singular
        value of `a`.

    Returns
    -------
    X : {(N,), (N, K)} |NDDataset|
        Least-squares solution. If `B` is two-dimensional,
        the solutions are in the `K` columns of `X`.
    residuals : {(), (1,), (K,)} ndarray
        Sums of residuals; squared Euclidean 2-norm for each column in
        ``B - A*X``.
    rank : int
        Rank of matrix `A`.
    s : (min(M, N),) ndarray
        Singular values of `A`.

    Raises
    ------
    LinAlgError
        If computation does not converge.

    Examples
    --------
    Let's take a similar example to the one given in the `numpy.linalg`
    documentation

    >>> from spectrochempy.scp import * # doctest: +ELLIPSIS
    ...

    Fit a line, :math:`d = v.t  + d_0`, through some noisy data-points:

    >>> t = NDDataset([0, 1, 2, 3], units='hour')
    >>> v = NDDataset([-1, 0.2, 0.9, 2.1], units='kilometer')

    By examining the coefficients, we see that the line should have a
    gradient of roughly 1 km/h and cut the y-axis at, more or less, -1 km.
    We can rewrite the line equation as :math:`d = A P`, where
    :math:`A = [[t 1]]` and :math:`P = [[v], [d_0]]`.  Now use `lstsq` to
    solve for `t`:

    >>> A = stack([t, np.ones(len(t))]).T
    >>> A
    array([[ 0.,  1.],
           [ 1.,  1.],
           [ 2.,  1.],
           [ 3.,  1.]])

    >>> v, d0 = np.linalg.lstsq(A, d)[0]
    >>> print(v, d0)
    1.0 -0.95

    Plot the data along with the fitted line:
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(x, y, 'o', label='Original data', markersize=10)
    >>> plt.plot(x, m*x + c, 'r', label='Fitted line')
    >>> plt.legend()
    >>> plt.show()

    """

    X, res, rank, s = np.linalg.lstsq(A.data, B.data, rcond)

    X = NDDataset(X)
    X.name = A.name + ' \ ' + B.name
    X.axes[0] = A.axes[1]
    X.axes[1] = B.axes[1]
    X.history = 'computed by spectrochempy.lstsq \n'
    return X, res, rank, s

if __name__ == '__main__':

    import spectrochempy.scp as sc

    t = sc.NDDataset([0, 1, 2, 3], units='hour')
    d = sc.NDDataset([-1, 0.2, 0.9, 2.1], units='kilometer')
    A = sc.stack([t, np.ones(len(t))]).T

    v, d0 = np.linalg.lstsq(A, d)[0]
    print(v, d0)

    import matplotlib.pyplot as plt
    plt.plot(t, d, 'o',label='Original data',markersize=10)
    plt.plot(t, v * t + d0, 'r',label='Fitted line')
    plt.legend()
    plt.show()