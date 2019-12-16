# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

# TODO: create tests

__all__ = ['LSTSQ']  # , 'NNLS']

import numpy as np

from traitlets import HasTraits, Instance

from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.dataset.ndarray import NDArray

from spectrochempy.units import Quantity


class LSTSQ(HasTraits):
    """
    Least-squares solution to a linear matrix equation.

    Given a vector `X` and a vector Y, the equation `A.X + B = Y` is solved by
    computing  ``A``, and ``B`` that minimizes the Euclidean 2-norm
    `|| Y - (A.X + B) ||^2`.

    """

    X = Instance(NDArray)
    Y = Instance(NDDataset)

    def __init__(self, *datasets):
        """
        Parameters
        ----------
        *datasets : one or two |NDDataset|'s or array-like objects
            If a single dataset `Y` is provided, the `X` data will be the `x`
            coordinates of the `Y` dataset, or the index of the data if not
            coordinates exists.
            If two datasets `X`, and `Y` are given, the `x` coordinates of `Y`
            are ignored and replaced by the `X` data.

        """

        if len(datasets) > 2 or len(datasets) < 1:
            raise ValueError('one or two dataset at max are expected')

        if len(datasets) == 2:
            X, Y = datasets
            if Y.coords is not None:
                if np.any(X.data != Y.x.data) or X.units != Y.x.units:
                    raise ValueError('X and Y dataset are not compatible')

        else:  # nb dataset ==1
            # abscissa coordinates are the X
            X = datasets[0].x

            Y = datasets[0]

        self.X = X
        self.Y = Y

        Xdata = np.vstack([X.data, np.ones(len(X.data))]).T
        Ydata = Y.data

        P, res, rank, s = np.linalg.lstsq(Xdata, Ydata, rcond=-1)

        self._P = P
        self._res = res
        self._rank = rank
        self._s = s

    def transform(self):
        """
        Return the least square coefficients A and B

        Returns
        -------
        Quantity or NDDataset, depending on the dimension of the linear system.

        """
        P = self._P
        X = self.X
        Y = self.Y

        if P.shape == (2,):
            # this is the result of the single equation, so only one value
            # should be returned
            A = P[0] * Y.units / X.units
            B = P[1] * Y.units

        else:
            A = NDDataset(data=P[0],
                          units=Y.units / X.units,
                          title="%s/%s" % (Y.title, X.title), )
            B = NDDataset(data=P[1] * np.ones(X.size),
                          units=Y.units,
                          title="%s at origin" % Y.title)

            A.history = 'Computed by spectrochempy.lstsq \n'
            B.history = 'Computed by spectrochempy.lstsq \n'

        return A, B

    trans = transform  # short-cut

    def inverse_transform(self):
        """
        Return the reconstructed data from the A and B least-square
        coefficients

        Returns
        -------
        |NDDataset|

        """
        A, B = self.transform()

        if isinstance(A, Quantity):
            Yp = self.Y.copy()
            Yp.data = (self.X * A + B).data
        else:
            Yp = (A * self.X + B)
        return Yp

    itrans = inverse_transform


def NNLS(C, d, x0=None, tol=None, itmax_factor=3):
    """Linear least squares with nonnegativity constraints
    (x, resnorm, residual) = lsqnonneg(C,d)
    returns the vector x that minimizes norm(d-C*x)
    subject to x >= 0, C and d must be real

    # python implementation of NNLS algorithm
    # References : Lawson, C.L. and R.J. Hanson, Solving Least-Squares Problems,
    #             Prentice-Hall, Chapter 23, p. 161, 1974.
    # Contributed by Klaus Schuch (schuch@igi.tugraz.at)
    # based on MATLAB's lsqnonneg function
    #
    # AT : if C and d are datasets, x will be a dataset with relevant dims
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
