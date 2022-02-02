# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory.
# ======================================================================================================================
"""
This module implements the IRIS class.
"""
__all__ = ["IRIS", "kern"]
__dataset_methods__ = []

import numpy as np
import quadprog
from matplotlib import pyplot as plt
from scipy import optimize
from collections.abc import Iterable


from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core import info_, warning_


def kern(K, p, q):
    """
    Compute kernel of Fredholm equation of the 1st kind.

    This function computes a kernel matrix and returns it as NDDataset. Pre-defined kernels can be chosen among:
    {'langmuir', 'ca', 'reactant-first-order', 'product-first-order', diffusion} A custom kernel fucntion - a
    2-variable lambda function `ker(p, q)` or a function returning a ndarray can be passed. `p` and `q` contain
    the values of an external experimental variable and an internal physico-chemical parameter, respectively.

    Parameters
    ----------
    K : str or callable
        Kernel type.
    p : Coord or ndadarray
        External variable.
    q : Coord or ndadarray
        Internal variable.

    Returns
    -------
    NDDataset
        The kernel.

    See Also
    --------
    IRIS : Integral inversion solver for spectroscopic data.

    Examples
    --------
    # the three examples below are equivalents:
    >>> scp.kern('langmuir', np.linspace(0, 1, 100), np.logspace(-10, 1, 10))
    NDDataset: [float64] unitless (shape: (y:100, x:10))

    >>> F = lambda p, q : np.exp(-q) * p[:, None] / (1 + np.exp(-q) * p[:, None])
    >>> scp.kern(F, np.linspace(0, 1, 100), np.logspace(-10, 1, 10))
    NDDataset: [float64] unitless (shape: (y:100, x:10))

    >>> def F(p,q):
    ...    return np.exp(-q) * p[:, None] / (1 + np.exp(-q) * p[:, None])
    >>>
    >>> scp.kern(F, np.linspace(0, 1, 100), np.logspace(-10, 1, 10))
    NDDataset: [float64] unitless (shape: (y:100, x:10))

    # p and q can also be passed as coordinates:
    >>> p = scp.Coord(np.linspace(0, 1, 100), name="pressure", title="p", units="torr")
    >>> q = scp.Coord(np.logspace(-10, 1, 10), name="reduced adsorption energy",
    ...              title="$\Delta_{ads}G^{0}/RT$", units="")
    >>> scp.kern('langmuir', p, q)
    NDDataset: [float64] unitless (shape: (y:100, x:10))
    """

    if not isinstance(q, Coord):  # q was passed as a ndarray
        q = Coord(data=q, name="internal variable", title="$q$", units="")
        q_was_array = True
    else:
        q_was_array = False
    if not isinstance(p, Coord):  # p was passed as a ndarray
        p = Coord(data=p, name="external variable", title="$p$", units="")
        p_was_array = True
    else:
        p_was_array = False

    if isinstance(K, str):
        _adsorption = {"langmuir", "ca"}
        _kinetics = {"reactant-first-order", "product-first-order"}
        _diffusion = {"diffusion"}

        if K.lower() in _adsorption:
            if q_was_array:  # change default values and units
                q.name = "reduced adsorption energy"
                q.title = "$\Delta_{ads}G^{0}/RT$"
            if p_was_array:  # change default values and units
                p.name = "relative pressure"
                p.title = "$p_/p_^{0}$"

            if K.lower() == "langmuir":
                K_ = (
                    np.exp(-q.data)
                    * p.data[:, None]
                    / (1 + np.exp(-q.data) * p.data[:, None])
                )

            else:  # this is 'ca'
                K_ = np.ones((len(p.data), len(q.data)))
                K_[p.data[:, None] < q.data] = 0

            title = "coverage"

        elif K.lower() in _kinetics:
            if q_was_array:  # change default values and units
                q.name = "Ln of rate constant"
                q.title = "$\Ln k$"
            if p_was_array:  # change default values and units
                p.name = "time"
                p.title = "$t$"
                p.units = "s"

            if K.lower() == "reactant-first-order":
                K_ = np.exp(-1 * np.exp(q.data) * p.data[:, None])

            else:  # 'product-first-order'
                K_ = 1 - np.exp(-1 * np.exp(q.data) * p.data[:, None])

            title = "coverage"

        elif K.lower() in _diffusion:
            if q_was_array:  # change default values and units
                q.name = "Diffusion rate constant"
                q.title = "$\\tau^{-1}$"
                # q.to('1/s', force=True)
            if p_was_array:  # change default values and units
                p.name = "time"
                p.title = "$t$"
                # p.to('s', force=True)

            title = "fractional uptake"

            K_ = np.zeros((p.size, q.size))
            for n in np.arange(1, 100):
                K_ += (1 / n ** 2) * np.exp(
                    -(1 / 9) * n ** 2 * np.pi ** 2 * q.data * p.data[:, None]
                )
            K_ = 1 - (6 / np.pi ** 2) * K_

        else:
            raise NameError(f"This kernel: <{K}> is not implemented")

    elif callable(K):
        K_ = K(p.data, q.data)
        title = ""

    else:
        raise ValueError("K must be a str or a callable")

    # weighting coefficients for the numerical quadrature of the Fredholm integral
    w = np.zeros((q.size))
    w[0] = 0.5 * (q.data[-1] - q.data[0]) / (q.size - 1)
    w[1:-1] = 2 * w[0]
    w[-1] = w[0]

    out = NDDataset(K_ * w)
    if isinstance(K, str):
        out.name = K + " kernel matrix"
    else:
        out.name = "kernel matrix"
    out.dims = ["y", "x"]
    out.y = p
    out.x = q
    out.title = title

    return out


class IRIS:
    """
    Integral inversion solver for spectroscopic data.

    Solves integral equations of the first kind of 1 or 2 dimensions, i.e. returns a
    distribution f of contributions to 1D ou 2D datasets.

    Parameters
    -----------
    X : NDDataset
        The 1D or 2D dataset on which to perform the IRIS analysis.
    K : str or callable or NDDataset
        Kernel of the integral equation. Pre-defined kernels can be chosen among
        `["langmuir", "ca", "reactant-first-order", "product-first-order", "diffusion"]`.
    p : Coord or Iterable
        External variable. Must be provided if the kernel is passed as a str or callable.
    q : Coord or Iterable of 3 values
        Internal variable. Must be provided if the kernel is passed as a str or callable.
    reg_par : None or array_like of two values `[min, max]` or three values `[start, stop, num]`
        Regularization parameter.

    Attributes
    ----------
    f : NDDataset
        A 3D/2D dataset containing the solutions (one per regularization parameter).
    RSS: array of float
        Residual sums of squares (one per regularization parameter).
    SM : array of float
        Values of the regularization constraint (one per regularization parameter).
    reg_par : None or array of float
        Values of the regularization parameters.
    log : str
        Log of the optimization.
    K : NDDataset
        Kernel matrix.
    X : NDDataset
        The original dataset.

    See Also
    --------
        ker : Compute kernel of Fredholm equation of the 1st kind.

    Notes
    -----
    IRIS solves integral equation of the first kind of 1 or 2 dimensions, i.e. finds a distribution
    function :math:`f(p)` or :math:`f(c,p)` of contributions to univariate data :math:`a(p)` or multivariate
    :math:`a(c, p)` data evolving with an external experimental variable :math:`p` (time, pressure,
    temperature, concentration, ...) according to the integral transform:

    .. math:: a(c, p) = \int_{min}^{max} k(q, p) f(c, q) dq

    .. math:: a(p) = \int_{min}^{max} k(q, p) f(q) dq

    where the kernel :math:`k(q, p)` expresses the functional dependence of a single contribution
    with respect to the experimental variable :math:`p` and and 'internal' physico-chemical variable :math:`q`
    Regularization is triggered when 'reg_param' is set to an array of two or three values.
    If 'reg_param' has two values [min, max], the optimum regularization parameter is searched between
    :math:`10^{min}` and :math:`10^{max}`. Automatic search of the regularization is made using the
    Cultrera_Callegaro algorithm (arXiv:1608.04571v2) which involves the Menger curvature of a circumcircle
    and the golden section search method.
    If three values are given (`[min, max, num]`), then the inversion will be made for num values
    evenly spaced on a log scale between :math:`10^{min}` and :math:`10^{max}`

    Examples
    --------
    >>> X = scp.read("irdata/CO@Mo_Al2O3.SPG")
    >>> p = [0.003, 0.004, 0.009, 0.014, 0.021, 0.026, 0.036, 0.051, 0.093, 0.150,
    ...      0.203, 0.300, 0.404, 0.503, 0.602, 0.702, 0.801, 0.905, 1.004]
    >>> iris = scp.IRIS(X[:,2250.0:1960.0], "langmuir", q = [-8, -1, 10])
    >>> iris.f
    NDDataset: [float64] unitless (shape: (z:1, y:10, x:301))
    """

    def __init__(self, X, K, p=None, q=None, reg_par=None):
        global _log
        _log = ""

        # check if x dimension exists
        if "x" in X.dims:
            # if multiple coords for a given dimension, take the default ones:
            channels = X.x.default
        else:
            # else, set a single channel:
            channels = Coord([0])

        if p is not None:  # supersedes the default
            if isinstance(p, Coord):
                if p.shape[1] != X.shape[0]:
                    raise ValueError(
                        "'p' should be consistent with the y coordinate of the dataset"
                    )
            else:
                if len(p) != X.shape[0]:
                    raise ValueError(
                        "'p' should be consistent with the y coordinate of the dataset"
                    )
                p = Coord(p, title="External variable")
        else:
            p = X.y.default

        # check options
        # defines the kernel

        if isinstance(K, NDDataset) and q is None:
            q = K.x
        elif isinstance(K, str) or callable(K):
            if isinstance(q, Coord):
                pass
            elif isinstance(q, Iterable):
                if len(q) == 3:
                    q = np.linspace(q[0], q[1], q[2])
            else:
                raise ValueError(
                    "q must be provided as a Coord, a NDarray or an iterable of 3 items"
                )

            msg = f"Build kernel matrix with: {K}\n"
            info_(msg)
            _log += msg
            K = kern(K, p, q)
            q = K.x  # q is now a Coord

        # defines regularization parameter values

        if reg_par is None:
            regularization = False
            search_reg = False
            reg_par = [0]
        elif len(reg_par) == 2:
            regularization = True
            search_reg = True
        elif len(reg_par) == 3:
            regularization = True
            search_reg = False
            reg_par = np.logspace(reg_par[0], reg_par[1], reg_par[2])
        else:
            raise ValueError(
                "reg_par should be either None or a set of 2 or 3 integers"
            )

        # define containers for outputs
        if not regularization:
            f = np.zeros((1, len(q), len(channels.data)))
            RSS = np.zeros((1))
            SM = np.zeros((1))

        if regularization and not search_reg:
            f = np.zeros((len(reg_par), len(q), len(channels.data)))
            RSS = np.zeros((len(reg_par)))
            SM = np.zeros((len(reg_par)))

        if regularization and search_reg:
            f = np.zeros((4, len(q), len(channels.data)))
            RSS = np.zeros((4))
            SM = np.zeros((4))

        # Define S matrix (sharpness), see function _Smat() below
        msg = "Build S matrix (sharpness)\n"
        info_(msg)
        _log += msg
        S = _Smat(q)
        msg = "... done\n"
        info_(msg)
        _log += msg

        # Solve unregularized problem
        if not regularization:
            msg = "Solving for {} channels and {} observations, no regularization\n".format(
                X.shape[1], X.shape[0]
            )
            _log += msg
            info_(msg)

            # use scipy.nnls() to solve the linear problem: X = K f
            for j, _ in enumerate(channels.data):
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
            C = np.eye(len(q))
            b = np.zeros(len(q))

            def solve_for_reg_par(X, K, G0, reg_par, S):
                """
                QP optimization

                parameters:
                -----------
                X: NDDataset of experimental spectra
                K: NDDataset, kernel datase
                G0: the lambda independent part of G
                reg_par: regularization parameter
                S: penalty function (shaprness)
                verbose: print info

                returns:
                --------
                f, RSS and SM for a given regularization parameter
                """
                global _log

                fi = np.zeros((len(q), len(channels.data)))

                for j, channel in enumerate(channels.data):
                    try:
                        G = G0 + 2 * reg_par * S
                        fi[:, j] = quadprog.solve_qp(G, a[j].squeeze(), C, b)[0]
                    except ValueError:  # pragma: no cover
                        msg = f"Warning:G is not positive definite for log10(lambda)={np.log10(reg_par):.2f} at {channel:.2f} {channels.units}, find nearest PD matrix"
                        warning_(msg)
                        _log += msg
                        try:
                            G = _nearestPD(G0 + 2 * reg_par * S, 0)
                            fi[:, j] = quadprog.solve_qp(G, a[j].squeeze(), C, b)[0]
                        except ValueError:
                            msg = (
                                "... G matrix is still ill-conditioned, "
                                "try with a small shift of diagonal elements..."
                            )
                            warning_(msg)
                            _log += msg
                            G = _nearestPD(G0 + 2 * reg_par * S, 1e-3)
                            fi[:, j] = quadprog.solve_qp(G, a[j].squeeze(), C, b)[0]

                resi = X.data - np.dot(K.data, fi)
                RSSi = np.sum(resi ** 2)
                SMi = np.linalg.norm(np.dot(np.dot(np.transpose(fi), S), fi))

                msg = (
                    f"log10(lambda)={np.log10(reg_par):.3f} -->  residuals = {RSSi:.3e}    "
                    f"regularization constraint  = {SMi:.3e}\n"
                )
                info_(msg)
                _log += msg

                return fi, RSSi, SMi

            if not search_reg:
                msg = (
                    f"Solving for {X.shape[1]} channels, {X.shape[0]} observations and "
                    f"{len(reg_par)} regularization parameters \n"
                )
                info_(msg)
                _log += msg

                for i, lamda_ in enumerate(reg_par):
                    f[i], RSS[i], SM[i] = solve_for_reg_par(X, K, G0, lamda_, S)

            else:
                msg = (
                    f"Solving for {X.shape[1]} channel(s) and {X.shape[0]} observations, search "
                    f"optimum regularization parameter in the range: [10**{min(reg_par)}, 10**{max(reg_par)}]\n"
                )
                info_(msg)
                _log += msg

                x = np.zeros(4)
                epsilon = 0.1
                phi = (1 + np.sqrt(5)) / 2

                x[0] = min(reg_par)
                x[3] = max(reg_par)
                x[1] = (x[3] + phi * x[0]) / (1 + phi)
                x[2] = x[0] + x[3] - x[1]
                reg_par = 10 ** x
                msg = "Initial Log(lambda) values = " + str(x)
                info_(msg)
                _log += msg

                for i, xi in enumerate(x):
                    f[i], RSS[i], SM[i] = solve_for_reg_par(X, K, G0, 10 ** xi, S)

                Rx = np.copy(RSS)
                Sy = np.copy(SM)
                while "convergence not reached":
                    C1 = _menger(np.log10(Rx[0:3]), np.log10(Sy[0:3]))
                    C2 = _menger(np.log10(Rx[1:4]), np.log10(Sy[1:4]))
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

                        f_, Rx[1], Sy[1] = solve_for_reg_par(X, K, G0, 10 ** x[1], S)
                        reg_par = np.append(reg_par, np.array(10 ** x[1]))
                        f = np.concatenate((f, np.atleast_3d(f_.T).T))
                        RSS = np.concatenate((RSS, np.array(Rx[1:2])))
                        SM = np.concatenate((SM, np.array(Sy[1:2])))
                        C2 = _menger(np.log10(Rx[1:4]), np.log10(Sy[1:4]))
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
                        f_, Rx[1], Sy[1] = solve_for_reg_par(X, K, G0, 10 ** x[1], S)
                        f = np.concatenate((f, np.atleast_3d(f_.T).T))
                        reg_par = np.append(reg_par, np.array(10 ** x[1]))
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
                        msg = "New range (Log lambda):" + str(x)
                        info_(msg)
                        _log += msg
                        f_, Rx[2], Sy[2] = solve_for_reg_par(X, K, G0, 10 ** x[2], S)
                        f = np.concatenate((f, np.atleast_3d(f_.T).T))
                        reg_par = np.append(reg_par, np.array(10 ** x[2]))
                        RSS = np.concatenate((RSS, np.array(Rx[1:2])))
                        SM = np.concatenate((SM, np.array(Sy[1:2])))
                    if (10 ** x[3] - 10 ** x[0]) / 10 ** x[3] < epsilon:
                        break
                id_opt = np.argmin(np.abs(reg_par - np.power(10, x_)))
                id_opt_ranked = np.argmin(np.abs(np.argsort(reg_par) - id_opt))
                msg = f"\n optimum found: index = {id_opt_ranked} ; Log(lambda) = {x_:.3f} ; lambda = {np.power(10, x_):.5e} ; curvature = {C_:.3f}"
                info_(msg)
                _log += msg

            # sort by lamba values
            argsort = np.argsort(reg_par)
            reg_par = reg_par[argsort]
            RSS = RSS[argsort]
            SM = SM[argsort]
            f = f[argsort]

        msg = "\n Done."
        info_(msg)
        _log += msg

        f = NDDataset(f)
        f.name = "2D distribution functions"
        f.title = "density"
        f.history = "2D IRIS analysis of {} dataset".format(X.name)
        f.set_coordset(z=Coord(data=reg_par, title="lambda"), y=q.copy(), x=channels)
        self.f = f
        self.K = K
        self.X = X
        self.reg_par = reg_par
        self.RSS = RSS
        self.SM = SM
        self.log = _log

    def reconstruct(self):
        """
        Transform data back to the original space.

        The following matrix operation is performed : :math:`\\hat{X} = K.f[i]`
        for each value of the regularization parameter.

        Returns
        -------
            NDDataset
                The reconstructed dataset.
        """

        if len(self.reg_par) == 1:  # no regularization or single lambda
            X_hat = NDDataset(
                np.zeros((self.X.shape)), title=self.X.title, units=self.X.units
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

    def plotlcurve(self, scale="ll", title="L curve"):
        """
        Plot the L Curve.

        Parameters
        ----------
        scale : str, optional, default='ll'
            String of 2 letters among 'l' (log) or 'n' (non-log) indicating whether the y and x
            axes should be log scales.
        title : str, optional, default='L curve'
            Plot title.

        Returns
        -------
            matplotlib.pyplot.axes
                The axes.
        """

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title("L curve")
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
        Plot the input dataset, reconstructed dataset and residuals.

        Parameters
        ----------
        index : int, list or tuple of int, optional, default: None
            Index(es) of the inversions (i.e. of the lambda values) to consider.
            If 'None': plots for all indices.

        **kwargs
            Keywords arguments passed to the plot() function.

        Returns
        -------
            list of axes
                The axes.
        """

        colX, colXhat, colRes = kwargs.get("colors", ["blue", "green", "red"])

        X_hat = self.reconstruct()
        axeslist = []
        if index is None:
            index = range(len(self.reg_par))
        if type(index) is int:
            index = [index]

        for i in index:
            if X_hat.ndim == 3:  # if several lambda
                X_hat_ = X_hat[i].squeeze()
            else:
                X_hat_ = X_hat  # if single lambda or no regularization
            res = self.X - X_hat_
            ax = self.X.plot()
            ax.plot(self.X.x.data, X_hat_.squeeze().T.data, "-", color=colXhat)
            ax.plot(self.X.x.data, res.T.data, "-", color=colRes)
            ax.set_title(f"2D IRIS merit plot, $\lambda$ = {self.reg_par[i]:.2e}")
            axeslist.append(ax)
        return axeslist

    def plotdistribution(self, index=None, **kwargs):
        """
        Plot the distribution function.

        This fucntion plots the distribution function f of the IRIS object.

        Parameters
        ----------
        index : optional, int, list or tuple of int. default: None
            Index(es) of the inversions (i.e. of the regularization parameter) to consider.
            If 'None': plots for all indices.
        **kwargs
            Other optional arguments are passed in the plots.

        Returns
        -------
            List of axes
                The axes.
        """

        axeslist = []
        if index is None:
            index = range(len(self.reg_par))
        if type(index) is int:
            index = [index]
        for i in index:
            axeslist.append(self.f[i].plot(method="map", **kwargs))
        return axeslist


# --------------------------------------------
# Utility private functions


def _menger(x, y):
    """
    returns the Menger curvature of a triplet of
    points. x, y = sets of 3 cartesian coordinates
    """
    numerator = 2 * (((x[1] - x[0]) * (y[2] - y[1])) - ((y[1] - y[0]) * (x[2] - x[1])))
    # euclidian distances
    r01 = (x[1] - x[0]) ** 2 + (y[1] - y[0]) ** 2
    r12 = (x[2] - x[1]) ** 2 + (y[2] - y[1]) ** 2
    r02 = (x[2] - x[0]) ** 2 + (y[2] - y[0]) ** 2

    denominator = np.sqrt(r01 * r12 * r02)
    return numerator / denominator


def _Smat(q):
    """returns the matrix used to compute the norm of f second derivative"""
    m = len(q)
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

    S = ((q.data[m - 1] - q.data[0]) / (m - 1)) ** (-3) * S
    return S


def _nearestPD(A, shift):  # pragma: no cover
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
    if _isPD(A3):
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
    while not _isPD(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += Ie * (-mineig * k ** 2 + spacing)
        k += 1
        print("makes PD matrix")
    return A3


def _isPD(B):  # pragma: no cover
    """
    Return True when input is positive-definite.

    copyright: see https://gist.github.com/fasiha/fdb5cec2054e6f1c6ae35476045a0bbd.
    """

    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False
