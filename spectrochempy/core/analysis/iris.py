# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory.
# ======================================================================================================================
"""
This module implements the IRIS class.
"""
__all__ = ["IRIS"]
__dataset_methods__ = []

import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize

from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core import info_, warning_

from quadprog import solve_qp


class IRIS:
    """
    Integral inversion solver for spectroscopic data.
    """

    def __init__(self, X, param, **kwargs):
        """

        Parameters
        -----------
        X : |NDDataset|
            The 1D or 2D dataset on which to perform the IRIS analysis.
        param : dict
            Dictionary of parameters with the following keys :

            *   'kernel': str or callable
                Kernel function of the integral equation. Pre-defined functions can be chosen among
                {'langmuir', 'ca', 'reactant-first-order', 'product-first-order', diffusion} (see Notes below).
                A custom kernel consisting of a 2-variable lambda function `ker(p, eps)`
                can be passed, where `p` and `eps` are an external experimental variable and an internal
                physico-chemical parameter, respectively.
            *   'epsRange': array-like of three values [start, stop, num]
                Defines the interval of eps values.
                start, stop: the starting and end values of eps, num: number of values.
            *   'lambdaRange': None or array_like of two values [min, max] or three values [start, stop, num]
                (see Notes below).
            *   'p': array or coordinate of the external variable. If none is given, p = X.y.values.
        verbose : bool
            if true, print running information.

        Attributes
        ----------
        f : |NDDataset|
            A 3D/2D dataset containing the solutions (one per regularization parameter).
        RSS: array of float
            Residual sums of squares (one per regularization parameter).
        SM : array of float
            Values of the penalty function (one per regularization parameter).
        lamda : array of float
            Values of the regularization parameters.
        log : str
            Log of the optimization.
        K : |NDDataset|
            Kernel matrix.
        X : |NDDataset|
            Copy of the original dataset.

        Notes
        -----
        IRIS solves integral equation of the first kind of 1 or 2 dimensions,
        i.e. finds a distribution function :math:`f` of contributions to spectra
        :math:`a(\nu,p)` or univariate measurement :math:`a(p)` evolving with an
        external experimental variable :math:`p` (time, pressure, temperature,
        concentration, ...) according to the integral transform:

        .. math:: a(\nu, p) = \\int_{min}^{max} k(\\epsilon, \\p) f(\\nu, \\epsilon) dp

        .. math:: a(p) = \\int_{min}^{max} k(\\epsilon, p) f(\\epsilon) dp

        where the kernel :math:`k(\\epsilon, p)` expresses the functional
        dependence of a single contribution with respect to the experimental
        variable math:`p` and and 'internal' physico-chemical variable
        :math:`\\epsilon`

        Regularization is triggered when 'lambdaRange' is set to an array of two
        or three values.

        - If 'lambdaRange' has two values [min, max], the optimum regularization
        parameter is searched between :math:`10^{min}` and :math:`10^{max}`.
        Automatic search of the regularization is made using the
        Cultrera_Callegaro algorithm (arXiv:1608.04571v2)
        which involves the Menger curvature of a circumcircle and the golden
        section search method.

        - If three values are given ([min, max, num]), then the inversion will
        be made for num values evenly spaced on a log scale between
        :math:`10^{min}` and :math:`10^{max}`
        """

        global _log
        _log = ""

        # If multiple coords for a given dimension, take the default ones:
        coord_x = X.x.default
        coord_y = X.y.default

        # Defines the kernel
        kernel = param.get("kernel", None)
        if kernel is not None:
            if isinstance(kernel, str):
                if kernel.lower() == "langmuir":
                    ker = lambda p_, eps_: np.exp(-eps_) * p_ / (1 + np.exp(-eps_) * p_)
                elif kernel.lower() == "ca":
                    ker = lambda p_, eps_: 0 if p_ < np.exp(eps_) else 1
                elif kernel.lower() == "reactant-first-order":
                    ker = lambda t_, lnk_: np.exp(-1 * np.exp(lnk_) * t_)
                elif kernel.lower() == "product-first-order":
                    ker = lambda t_, lnk_: 1 - np.exp(-1 * np.exp(lnk_) * t_)
                elif kernel.lower() == "diffusion":

                    def ker(t_, tau_inv_):
                        ker_ = np.zeros_like(t_)
                        for n in np.arange(1, 100):
                            ker_ += (1 / n ** 2) * np.exp(
                                -(1 / 9) * n ** 2 * np.pi ** 2 * t_ * tau_inv_
                            )
                            return 1 - (6 / np.pi ** 2) * ker_

                else:
                    raise NameError(f"This kernel: <{kernel}> is not implemented")
            elif callable(kernel):
                ker = kernel
            else:
                raise ValueError("The kernel must be a str or a callable !")
        else:
            raise NameError("A kernel must be given !")

        # Define eps values
        epsRange = param.get("epsRange", None)
        try:
            eps = np.linspace(epsRange[0], epsRange[1], epsRange[2])
        except Exception as exc:
            raise Exception(
                "Parameter epsRange in param must be a list of 3 values: [start, stop, num]"
            ) from exc

        # Defines regularization parameter values
        lamb = [0]
        lambdaRange = param.get("lambdaRange", None)

        if lambdaRange is None:
            regularization = False
            searchLambda = False
        elif len(lambdaRange) == 2:
            regularization = True
            searchLambda = True
        elif len(lambdaRange) == 3:
            regularization = True
            searchLambda = False
            lamb = np.logspace(lambdaRange[0], lambdaRange[1], lambdaRange[2])
        else:
            raise ValueError(
                "lambdaRange should either None, or a set of 2 or 3 integers"
            )

        p = param.get("p", None)
        if p is not None:
            # Check p
            if isinstance(p, Coord):
                if p.shape[1] != X.shape[0]:
                    raise ValueError(
                        "`p` should be consistent with the y coordinate of the dataset"
                    )
                pval = p.data  # values
                # (values contains unit! to use it we must either have eps with units or normalise p
            else:
                if len(p) != X.shape[0]:
                    raise ValueError(
                        "`p` should be consistent with the y coordinate of the dataset"
                    )
                p = Coord(p, title="External variable")
                pval = p.data  # values
        else:
            p = coord_y
            pval = p.data  # values

        # if 'guess' in param:
        #     guess = param['guess']       <-- # TODO: never used.
        # else:
        #     guess = 'previous'

        # define containers for outputs
        if not regularization:
            f = np.zeros((1, len(eps), len(coord_x.data)))
            RSS = np.zeros(1)
            SM = np.zeros(1)
        else:
            if not searchLambda:
                f = np.zeros((len(lamb), len(eps), len(coord_x.data)))
                RSS = np.zeros((len(lamb)))
                SM = np.zeros((len(lamb)))
            else:
                f = np.zeros((4, len(eps), len(coord_x.data)))
                RSS = np.zeros(4)
                SM = np.zeros(4)

        # Define K matrix (kernel)
        msg = "Build kernel matrix...\n"
        info_(msg)
        _log += msg
        # first some weighting coefficients for the numerical quadrature of the Fredholm integral
        w = np.zeros((len(eps), 1))
        w[0] = 0.5 * (eps[-1] - eps[0]) / (len(eps) - 1)  #
        for j in range(1, len(eps) - 1):
            w[j] = 2 * w[0]
        w[-1] = w[0]
        # then compute K  (TODO: allow using weighting matrix W)
        K = NDDataset(np.zeros((p.size, len(eps))))
        K.set_coordset(y=p, x=Coord(eps, title="epsilon"))
        for i, p_i in enumerate(pval):
            for j, eps_j in enumerate(eps):
                K.data[i, j] = w[j] * ker(p_i, eps_j)

        # Define S matrix (sharpness), see function Smat() below
        S = Smat(eps)

        msg = "... done\n"
        info_(msg)
        _log += msg

        # Solve unregularized problem
        if not regularization:
            msg = (
                "Solving for {} wavenumbers and {} spectra, no regularization\n".format(
                    X.shape[1], X.shape[0]
                )
            )
            _log += msg
            info_(msg)

            # use scipy.nnls() to solve the linear problem: X = K f
            for j, freq in enumerate(coord_x.data):
                f[0, :, j] = optimize.nnls(K.data, X[:, j].data.squeeze())[0]
            res = X.data - np.dot(K.data, f[0].data)
            RSS[0] = np.sum(res ** 2)
            SM[0] = np.linalg.norm(np.dot(np.dot(np.transpose(f[0]), S), f[0]))

            msg = "-->  residuals = {:.2e}    curvature = {:.2e}".format(RSS[0], SM[0])
            _log += msg
            info_(msg)

        else:  # regularization
            # some matrices used for QP optimization do not depend on lambdaR
            # and are computed here. The standard form used by quadprog() is
            # minimize (1/2) xT G x - aT x ; subject to: C.T x >= b

            # The first part of the G matrix is independent of lambda:  G = G0 + 2 * lambdaR S
            G0 = 2 * np.dot(K.data.T, K.data)
            a = 2 * np.dot(X.data.T, K.data)
            C = np.eye(len(eps))
            b = np.zeros(len(eps))

            def solve_lambda(X, K, G0, lamda, S):
                """
                QP optimization

                parameters:
                -----------
                X: NDDataset of experimental spectra
                K: NDDataset, kernel datase
                G0: the lambda independent part of G
                lamda: regularization parameter
                S: penalty function (shaprness)
                verbose: print info

                returns:
                --------
                f, RSS and SM for a given regularization parameter
                """
                global _log

                fi = np.zeros((len(eps), len(coord_x.data)))

                for j, freq in enumerate(coord_x.data):
                    try:
                        G = G0 + 2 * lamda * S
                        fi[:, j] = solve_qp(G, a[j].squeeze(), C, b)[0]
                    except ValueError:
                        msg = (
                            f"Warning:G is not positive definite for log10(lambda)={np.log10(lamda):.2f} "
                            f"at {freq:.2f} {coord_x.units}, find nearest PD matrix"
                        )
                        warning_(msg)
                        _log += msg
                        try:
                            G = nearestPD(G0 + 2 * lamda * S, 0)
                            fi[:, j] = solve_qp(G, a[j].squeeze(), C, b)[0]
                        except ValueError:
                            msg = (
                                "... G matrix is still ill-conditioned, "
                                "try with a small shift of diagonal elements..."
                            )
                            warning_(msg)
                            _log += msg
                            G = nearestPD(G0 + 2 * lamda * S, 1e-3)
                            fi[:, j] = solve_qp(G, a[j].squeeze(), C, b)[0]

                resi = X.data - np.dot(K.data, fi)
                RSSi = np.sum(resi ** 2)
                SMi = np.linalg.norm(np.dot(np.dot(np.transpose(fi), S), fi))

                msg = (
                    f"log10(lambda)={np.log10(lamda):.3f} -->  residuals = {RSSi:.3e}    "
                    f"regularization constraint  = {SMi:.3e}\n"
                )
                info_(msg)
                _log += msg

                return fi, RSSi, SMi

            if not searchLambda:
                msg = (
                    f"Solving for {X.shape[1]} wavenumbers, {X.shape[0]} spectra and "
                    f"{len(lamb)} regularization parameters \n"
                )
                info_(msg)
                _log += msg

                for i, lamda in enumerate(lamb):
                    f[i], RSS[i], SM[i] = solve_lambda(X, K, G0, lamda, S)

            else:
                msg = (
                    f"Solving for {X.shape[1]} wavenumbers and {X.shape[0]} spectra, search "
                    f"regularization parameter in the range: [10**{min(lambdaRange)}, 10**{max(lambdaRange)}]\n"
                )
                info_(msg)
                _log += msg

                x = np.zeros(4)
                epsilon = 0.1
                phi = (1 + np.sqrt(5)) / 2

                x[0] = min(lambdaRange)
                x[3] = max(lambdaRange)
                x[1] = (x[3] + phi * x[0]) / (1 + phi)
                x[2] = x[0] + x[3] - x[1]
                lamb = 10 ** x
                msg = "Initial Log(lambda) values = " + str(x)
                info_(msg)
                _log += msg

                for i, xi in enumerate(x):
                    f[i], RSS[i], SM[i] = solve_lambda(X, K, G0, 10 ** xi, S)

                Rx = np.copy(RSS)
                Sy = np.copy(SM)
                while "convergence not reached":
                    C1 = menger(np.log10(Rx[0:3]), np.log10(Sy[0:3]))
                    C2 = menger(np.log10(Rx[1:4]), np.log10(Sy[1:4]))
                    msg = f"Curvatures of the inner points: C1 = {C1:.3f} ; C2 = {C2:.3f} \n"
                    info_(msg)
                    _log += msg

                    while "convergence not reached":
                        x[3] = x[2]
                        Rx[3] = Rx[2]
                        Sy[3] = Sy[2]
                        x[2] = x[1]
                        Rx[2] = Rx[1]
                        Sy[2] = Sy[1]
                        x[1] = (x[3] + phi * x[0]) / (1 + phi)
                        msg = "New range of Log(lambda) values: " + str(x)
                        info_(msg)
                        _log += msg

                        f_, Rx[1], Sy[1] = solve_lambda(X, K, G0, 10 ** x[1], S)
                        lamb = np.append(lamb, np.array(10 ** x[1]))
                        f = np.concatenate((f, np.atleast_3d(f_.T).T))
                        RSS = np.concatenate((RSS, np.array(Rx[1:2])))
                        SM = np.concatenate((SM, np.array(Sy[1:2])))
                        C2 = menger(np.log10(Rx[1:4]), np.log10(Sy[1:4]))
                        msg = f"new curvature: C2 = {C2:.3f}"
                        info_(msg)
                        _log += msg

                        if C2 > 0:
                            break

                    if C1 > C2:
                        x_ = x[1]
                        C_ = C1
                        x[3] = x[2]
                        Rx[3] = Rx[2]
                        Sy[3] = Sy[2]
                        x[2] = x[1]
                        Rx[2] = Rx[1]
                        Sy[2] = Sy[1]
                        x[1] = (x[3] + phi * x[0]) / (1 + phi)
                        msg = "New range (Log lambda): " + str(x)
                        info_(msg)
                        _log += msg
                        f_, Rx[1], Sy[1] = solve_lambda(X, K, G0, 10 ** x[1], S)
                        f = np.concatenate((f, np.atleast_3d(f_.T).T))
                        lamb = np.append(lamb, np.array(10 ** x[1]))
                        RSS = np.concatenate((RSS, np.array(Rx[1:2])))
                        SM = np.concatenate((SM, np.array(Sy[1:2])))
                    else:
                        x_ = x[2]
                        C_ = C2
                        x[0] = x[1]
                        Rx[0] = Rx[1]
                        Sy[0] = Sy[1]
                        x[1] = x[2]
                        Rx[1] = Rx[2]
                        Sy[1] = Sy[2]
                        x[2] = x[0] - (x[1] - x[3])
                        msg = "Log lambda= " + str(x)
                        info_(msg)
                        _log += msg
                        f_, Rx[2], Sy[2] = solve_lambda(X, K, G0, 10 ** x[2], S)
                        f = np.concatenate((f, np.atleast_3d(f_.T).T))
                        lamb = np.append(lamb, np.array(10 ** x[2]))
                        RSS = np.concatenate((RSS, np.array(Rx[1:2])))
                        SM = np.concatenate((SM, np.array(Sy[1:2])))
                    if (10 ** x[3] - 10 ** x[0]) / 10 ** x[3] < epsilon:
                        break
                msg = (
                    f"\n optimum found: log10(lambda) = {x_:.3f} ; curvature = {C_:.3f}"
                )
                info_(msg)
                _log += msg

        msg = "\n Done."
        info_(msg)
        _log += msg

        f = NDDataset(f)
        f.name = "2D distribution functions"
        f.title = "pseudo-concentration"
        f.history = "2D IRIS analysis of {} dataset".format(X.name)
        xcoord = X.coordset["x"]
        ycoord = Coord(data=eps, title="epsilon")
        zcoord = Coord(data=lamb, title="lambda")
        f.set_coordset(z=zcoord, y=ycoord, x=xcoord)
        self.f = f
        self.K = K
        self.X = X
        self.lamda = lamb
        self.RSS = RSS
        self.SM = SM
        self.log = _log

    def reconstruct(self):
        """
        Transform data back to the original space.

        The following matrix operation is performed : :math:`\\hat{X} = K.f[i]` for each value of the regularization
        parameter.

        Returns
        -------
        X_hat : |NDDataset|
            The reconstructed dataset.
        """

        if len(self.lamda) == 1:  # no regularization or single lambda
            X_hat = NDDataset(
                np.zeros((self.f.z.size, *self.X.shape)).squeeze(axis=0),
                title=self.X.title,
                units=self.X.units,
            )
            X_hat.set_coordset(y=self.X.y, x=self.X.x)
            X_hat.data = np.dot(self.K.data, self.f.data.squeeze())
        else:
            X_hat = NDDataset(
                np.zeros((self.f.z.size, *self.X.shape)),
                title=self.X.title,
                units=self.X.units,
            )
            X_hat.set_coordset(z=self.f.z, y=self.X.y, x=self.X.x)
            for i in range(X_hat.z.size):
                X_hat.data[i] = np.expand_dims(
                    np.dot(self.K.data, self.f[i].data.squeeze()), 0
                )

        X_hat.name = "2D-IRIS Reconstructed datasets"
        return X_hat

    def plotlcurve(self, scale="ll", title="L curve"):  # , **kwargs):
        """
        Plots the L Curve.

        Parameters
        ----------
        scale : str, optional, default='ll'
            2 letters among 'l' (log) or 'n' (non-log) indicating whether the y and x axes should be log scales.
        title : str, optional, default='L curve'
            Plot title.

        Returns
        -------
        ax : subplot axis
        """

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(title)
        scale = scale.lower()
        plt.plot(self.RSS, self.SM, "o")
        ax.set_xlabel("Residuals")
        ax.set_ylabel("Curvature")
        if scale[1] == "l":
            ax.set_xscale("log")
        if scale[0] == "l":
            ax.set_yscale("log")
        return ax

    def plotmerit(self, index=None, **kwargs):
        """
        Plots the input dataset, reconstructed dataset and residuals.

        Parameters
        ----------
        index : int, list or tuple of int, optional, default: None
            Index(es) of the inversions (i.e. of the lambda values) to consider.
            If 'None': plots for all indices.

        Returns
        -------
        list of axes
        """

        colX, colXhat, colRes = kwargs.get("colors", ["blue", "green", "red"])

        X_hat = self.reconstruct()
        axeslist = []
        if index is None:
            index = range(len(self.lamda))
        if type(index) is int:
            index = [index]

        for i in index:
            if X_hat.ndim == 3:  # if several lambda
                X_hat_ = X_hat[i].squeeze()
            else:
                X_hat_ = X_hat  # if single lambda or no regularization
            res = self.X - X_hat_
            ax = self.X.plot()
            ax.plot(self.X.x.data, X_hat_.squeeze().T.data, color=colXhat)
            ax.plot(self.X.x.data, res.T.data, color=colRes)
            ax.set_title(f"2D IRIS merit plot, $\\lambda$ = {self.lamda[i]:.2e}")
            axeslist.append(ax)
        return axeslist

    def plotdistribution(self, index=None, **kwargs):
        """
        Plots the input dataset, reconstructed dataset and residuals.

        Parameters
        ----------
        index : optional, int, list or tuple of int. default: None.
            Index(es) of the inversions (i.e. of the lambda values) to consider.
            If 'None': plots for all indices.
        kwargs:
            Other optional arguments are passed in the plots.

        Returns
        -------
        List of axes
        """

        axeslist = []
        if index is None:
            index = range(len(self.lamda))
        if type(index) is int:
            index = [index]
        for i in index:
            self.f[i].plot(method="map", **kwargs)
        return axeslist


# --------------------------------------------
# Utility functions


def Smat(eps):
    """
    Return the matrix used to compute the norm of f second derivative.
    """
    m = len(eps)
    S = np.zeros((m, m))
    S[0, 0] = 6
    S[0, 1] = -4
    S[0, 2] = 1
    S[1, 0] = -4
    S[1, 1] = 6
    S[1, 2] = -4
    S[1, 3] = 1

    for i in range(2, m - 2):
        S[i, i - 2] = 1
        S[i, i - 1] = -4
        S[i, i] = 6
        S[i, i + 1] = -4
        S[i, i + 2] = 1

    S[m - 2, m - 4] = 1
    S[m - 2, m - 3] = -4
    S[m - 2, m - 2] = 6
    S[m - 2, m - 1] = -4
    S[m - 1, m - 3] = 1
    S[m - 1, m - 2] = -4
    S[m - 1, m - 1] = 6

    S = ((eps[m - 1] - eps[0]) / (m - 1)) ** (-3) * S
    return S


def nearestPD(A, shift):
    """
    Find the nearest positive-definite matrix to input.

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].
    With addition of a small increment in the diagonal as in:
    https://github.com/stephane-caron/qpsolvers/pull/12/commits/945554d857e0c1e4623ddda8d8f801cb6f61d6af.

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd.

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6.

    copyright: see https://gist.github.com/fasiha/fdb5cec2054e6f1c6ae35476045a0bbd.
    """

    B = 0.5 * (A + A.T)
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = 0.5 * (B + H)

    A3 = 0.5 * (A2 + A2.T) + np.eye(A2.shape[0]).__mul__(shift)
    if isPD(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrices with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrices of small dimension, be on
    # the order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    Ie = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += Ie * (-mineig * k ** 2 + spacing)
        k += 1
        print("makes PD matrix")
    return A3


def isPD(B):
    """
    Return True when input is positive-definite.

    copyright: see https://gist.github.com/fasiha/fdb5cec2054e6f1c6ae35476045a0bbd.
    """

    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False


def menger(x, y):
    """
    Return the Menger curvature of a triplet of points.

    Parameters
    ==========
    x, y
        Sets of 3 cartesian coordinates.
    """

    numerator = 2 * (
        x[0] * y[1]
        + x[1] * y[2]
        + x[2] * y[0]
        - x[0] * y[2]
        - x[1] * y[0]
        - x[2] * y[1]
    )
    # Euclidian distances
    r01 = (x[1] - x[0]) ** 2 + (y[1] - y[0]) ** 2
    r12 = (x[2] - x[1]) ** 2 + (y[2] - y[1]) ** 2
    r02 = (x[2] - x[0]) ** 2 + (y[2] - y[0]) ** 2

    denominator = np.sqrt(r01 * r12 * r02)
    return numerator / denominator
