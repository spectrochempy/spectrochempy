# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory.
# ======================================================================================================================

# TODO: create tests

__all__ = ["CurveFit", "LSTSQ", "NNLS"]

import numpy as np
import scipy.optimize as sopt
import scipy.linalg as lng

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
        super().__init__()

        if len(datasets) > 2 or len(datasets) < 1:
            raise ValueError("one or two dataset at max are expected")

        if len(datasets) == 2:
            X, Y = datasets

        else:  # nb dataset ==1
            # abscissa coordinates are the X
            dim = datasets[0].dims[-1]
            X = getattr(datasets[0], dim)

            Y = datasets[0]

        self.X = X
        self.Y = Y

        Xdata = np.vstack([X.data, np.ones(len(X.data))]).T
        Ydata = Y.data

        P, res, rank, s = lng.lstsq(Xdata, Ydata)

        self._P = P
        self._res = res
        self._rank = rank
        self._s = s

    def transform(self):
        """
        Return the least square coefficients A and B.

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
            A = NDDataset(
                data=P[0],
                units=Y.units / X.units,
                title="%s/%s" % (Y.title, X.title),
            )
            B = NDDataset(
                data=P[1] * np.ones(X.size),
                units=Y.units,
                title="%s at origin" % Y.title,
            )

            A.history = "Computed by spectrochempy.lstsq \n"
            B.history = "Computed by spectrochempy.lstsq \n"

        return A, B

    trans = transform  # short-cut

    def inverse_transform(self):
        """
        Return the reconstructed data from the A and B least-square
        coefficients.

        Returns
        -------
        |NDDataset|
        """
        A, B = self.transform()

        if isinstance(A, Quantity):
            Yp = self.Y.copy()
            Yp.data = (self.X * A + B).data
        else:
            Yp = A * self.X + B
        return Yp

    itrans = inverse_transform


class NNLS(HasTraits):
    """
    Least-squares solution to a linear matrix equation with non-negativity constraints.

    This is a wrapper to the `scipy.optimize.nnls`` function.
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

        maxiter: int, optional
            Maximum number of iterations, optional.
            Default is ``3 * X.shape``.
        """
        super().__init__()

        if len(datasets) > 2 or len(datasets) < 1:
            raise ValueError("one or two dataset at max are expected")

        if len(datasets) == 2:
            X, Y = datasets

        else:  # nb dataset ==1
            # abscissa coordinates are the X
            dim = datasets[0].dims[-1]
            X = getattr(datasets[0], dim)

            Y = datasets[0]

        self.X = X
        self.Y = Y

        Xdata = np.vstack([X.data, np.ones(len(X.data))]).T
        Ydata = Y.data

        P, res = sopt.nnls(Xdata, Ydata, maxiter=None)

        self._P = P
        self._res = res

    def transform(self):
        """
        Return the least square coefficients A and B.

        Returns
        -------
        Coefficient
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
            A = NDDataset(
                data=P[0],
                units=Y.units / X.units,
                title="%s/%s" % (Y.title, X.title),
            )
            B = NDDataset(
                data=P[1] * np.ones(X.size),
                units=Y.units,
                title="%s at origin" % Y.title,
            )

            A.history = "Computed by spectrochempy.lstsq \n"
            B.history = "Computed by spectrochempy.lstsq \n"

        return A, B

    trans = transform  # short-cut

    def inverse_transform(self):
        """
        Return the reconstructed data from the A and B least-square
        coefficients.

        Returns
        -------
        dataset
            |NDDataset|.
        """
        A, B = self.transform()

        if isinstance(A, Quantity):
            Yp = self.Y.copy()
            Yp.data = (self.X * A + B).data
        else:
            Yp = A * self.X + B
        return Yp

    itrans = inverse_transform


class CurveFit(HasTraits):
    """
    Use non-linear least squares to fit a function, ``f``, to data.

    It assumes Y = f(X, *params) + eps.

    This is a wrapper to the `scipy.optimize.curve_fit`` function
    """

    # TODO: Something wrong here! This is exactly the same code as NNLS.
    # Probably a mistake to correct...

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

        maxiter: int, optional
            Maximum number of iterations, optional.
            Default is ``3 * X.shape``.
        """
        super().__init__()

        if len(datasets) > 2 or len(datasets) < 1:
            raise ValueError("one or two dataset at max are expected")

        if len(datasets) == 2:
            X, Y = datasets

        else:  # nb dataset ==1
            # abscissa coordinates are the X
            dim = datasets[0].dims[-1]
            X = getattr(datasets[0], dim)

            Y = datasets[0]

        self.X = X
        self.Y = Y

        Xdata = np.vstack([X.data, np.ones(len(X.data))]).T
        Ydata = Y.data

        P, res = sopt.nnls(Xdata, Ydata, maxiter=None)

        self._P = P
        self._res = res

    def transform(self):
        """
        Return the least square coefficients A and B.

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
            A = NDDataset(
                data=P[0],
                units=Y.units / X.units,
                title="%s/%s" % (Y.title, X.title),
            )
            B = NDDataset(
                data=P[1] * np.ones(X.size),
                units=Y.units,
                title="%s at origin" % Y.title,
            )

            A.history = "Computed by spectrochempy.lstsq \n"
            B.history = "Computed by spectrochempy.lstsq \n"

        return A, B

    trans = transform  # short-cut

    def inverse_transform(self):
        """
        Return the reconstructed data from the A and B least-square
        coefficients.

        Returns
        -------
        |NDDataset|
        """
        A, B = self.transform()

        if isinstance(A, Quantity):
            Yp = self.Y.copy()
            Yp.data = (self.X * A + B).data
        else:
            Yp = A * self.X + B
        return Yp

    itrans = inverse_transform
