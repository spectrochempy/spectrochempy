# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================

# TODO: create tests

__all__ = ['Lstsq', 'Lsqnonneg']

import numpy as np

from spectrochempy.dataset.nddataset import NDDataset


def Lstsq(A, B, rcond=-1):
    """
    Return the least-squares solution to a linear matrix equation.

    This is an extension of :meth:`numpy.linalg.lstsq` to |NDDataset|.

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

    >>> from spectrochempy import * # doctest: +ELLIPSIS
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


def Lsqnonneg(C, d, x0=None, tol=None, itmax_factor=3):
    """Linear least squares with nonnegativity constraints
    (x, resnorm, residual) = lsqnonneg(C,d)
    returns the vector x that minimizes norm(d-C*x)
    subject to x >= 0, C and d must be real

    # python implementation of NNLS algorithm
    # References: Lawson, C.L. and R.J. Hanson, Solving Least-Squares Problems,
    #             Prentice-Hall, Chapter 23, p. 161, 1974.
    # Contributed by Klaus Schuch (schuch@igi.tugraz.at)
    # based on MATLAB's lsqnonneg function
    #
    # AT: if C and d are datasets, x will be a dataset with relevant dims
    #     else return np array

    """

    returnDataset = False

    if isinstance(C, NDDataset) and isinstance(d, NDDataset):

        returnDataset = True

        X = NDDataset(np.zeros((C.shape[1], d.shape[1])))
        X.axes[0] = C.axes[1].copy()
        X.axes[1] = d.axes[1].copy()

        C = C.data
        d = d.data
        if isinstance(x0, NDDataset):
            x0 = x0.data

    eps = 2.22e-16  # from matlab

    def norm1(x):
        return abs(x).sum().max()

    def msize(x, dim):
        s = x.shape
        if dim >= len(s):
            return 1
        else:
            return s[dim]

    if tol is None:
        tol = 10 * eps * norm1(C) * (max(C.shape) + 1)

    C = np.asarray(C)
    (m, n) = C.shape
    P = np.zeros(n)
    Z = np.arange(1, n + 1)

    if x0 is None:
        x = P
    else:
        if any(x0 < 0):
            x = P
        else:
            x = x0

    ZZ = Z

    resid = d - np.dot(C, x)

    w = np.dot(C.T, resid)
    outeriter = 0
    it = 0
    itmax = itmax_factor * n
    exitflag = 1

    # outer loop to put variables into set to hold positive coefficients
    while np.any(Z) and np.any(w[ZZ - 1] > tol):

        outeriter += 1

        t = w[ZZ - 1].argmax()
        t = ZZ[t]

        P[t - 1] = t
        Z[t - 1] = 0

        PP = np.where(P != 0)[0] + 1
        ZZ = np.where(Z != 0)[0] + 1

        CP = np.zeros(C.shape)
        CP[:, PP - 1] = C[:, PP - 1]
        CP[:, ZZ - 1] = np.zeros((m, msize(ZZ, 1)))

        z = np.dot(np.linalg.pinv(CP), d)
        z[ZZ - 1] = np.zeros((msize(ZZ, 1), msize(ZZ, 0)))

        # inner loop to remove elements from the positve set which no longer belong

        while np.any(z[PP - 1] <= tol):

            it += 1

            if it > itmax:
                max_error = z[PP - 1].max()
                raise Exception(
                    'Exiting: Iteration count (=%d) exceeded\n Try raising the tolerance tol. (max_error=%d)' % (
                    it, max_error))

            QQ = np.where((z <= tol) & (P != 0))[0]
            alpha = min(x[QQ] / (x[QQ] - z[QQ]))
            x = x + alpha * (z - x)

            ij = np.where((abs(x) < tol) & (P != 0))[0] + 1
            Z[ij - 1] = ij
            P[ij - 1] = np.zeros(max(ij.shape))
            PP = np.where(P != 0)[0] + 1
            ZZ = np.where(Z != 0)[0] + 1

            CP[:, PP - 1] = C[:, PP - 1]
            CP[:, ZZ - 1] = np.zeros((m, msize(ZZ, 1)))

            z = np.dot(np.linalg.pinv(CP), d)
            z[ZZ - 1] = np.zeros((msize(ZZ, 1), msize(ZZ, 0)))
        x = z
        resid = d - np.dot(C, x)
        w = np.dot(C.T, resid)

        if returnDataset:
            X.data = x
            x = X.copy()

    return x, sum(resid * resid), resid

if __name__ == '__main__':

    pass