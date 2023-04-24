# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
This module implements the IRIS class.
"""
__all__ = ["IrisKernel", "IRIS"]
__configurables__ = ["IRIS"]

# from collections.abc import Iterable

import numpy as np
import quadprog
import traitlets as tr
from matplotlib import pyplot as plt
from scipy import optimize

from spectrochempy.analysis._base import DecompositionAnalysis, NotFittedError
from spectrochempy.core import info_, warning_
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.coordset import CoordSet
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.extern.traittypes import Array
from spectrochempy.utils.constants import EPSILON
from spectrochempy.utils.decorators import signature_has_configurable_traits
from spectrochempy.utils.docstrings import _docstring
from spectrochempy.utils.traits import CoordType, NDDatasetType


@tr.signature_has_traits
class IrisKernel(tr.HasTraits):
    """
    Define a kernel matrix of Fredholm equation of the 1st kind.

    This class define a kernel matrix as a `NDDataset` compatible
    with the `X` input `NDDataset`\ .

    Pre-defined kernels can be chosen among: {``'langmuir'``\ , ``'ca'``\ ,
    ``'reactant-first-order'``\ , ``'product-first-order'``\ , ``'diffusion'``\ },
    a custom kernel function - a 2-variable lambda
    function `K`\ ``(p, q)`` or a function returning a `~numpy.ndarray` can be passed.
    `p` and `q` contain the values of an external experimental variable and an internal
    physico-chemical parameter, respectively.

    Parameters
    -----------
    X : `NDDataset`
        The 1D or 2D dataset for the kernel is defined.
    K : any of [ ``'langmuir'`` , ``'ca'`` , ``'reactant-first-order'`` , ``'product-first-order'`` , ``'diffusion'`` ] or `callable` or `NDDataset`
        Predefined or user-defined Kernel for the integral equation.
    p :  `Coord` or ``iterable``
        External variable. Must be provided if the kernel `K` is passed as a `str` or
        `callable` .
    q :  `Coord` or ``iterable`` of 3 values
        Internal variable. Must be provided if the kernel `K` is passed as a `str` or
        `callable`.
    """

    _X = NDDatasetType(allow_none=True)
    _K = (
        tr.Union(
            (
                tr.Enum(
                    [
                        "langmuir",
                        "ca",
                        "reactant-first-order",
                        "product-first-order",
                        "diffusion",
                    ]
                ),
                tr.Callable(),
                NDDatasetType(),
            ),
            default_value=None,
            allow_none=True,
        ),
    )

    _p = CoordType(
        # CoordType include array-like iterables
    )
    _q = tr.Union(
        (
            tr.List(),
            CoordType(),
            # CoordType include array-like iterables
        ),
    )

    # ----------------------------------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------------------------------
    def __init__(self, X, K, p=None, q=None, **kwargs):

        info_("Creating Kernel...")

        self._X = X
        self._K = K
        if p is not None:
            self._p = p
        if q is not None:
            self._q = q

        info_("Kernel now ready as IrisKernel().kernel!")

    # ----------------------------------------------------------------------------------
    # default values
    # ----------------------------------------------------------------------------------
    @tr.default("_p")
    def _p_default(self):
        return self._X.coordset[self._X.dims[0]].default

    @tr.default("_q")
    def _q_default(self):
        q = None
        if isinstance(self._K, NDDataset):
            q = self._K.coordset[self._K.dims[-1]]
        return q

    # ----------------------------------------------------------------------------------
    # Validation
    # ----------------------------------------------------------------------------------
    @tr.validate("_p")
    def _p_validate(self, proposal):
        p = proposal.value
        # as p belong CoordType, it is necessarily a Coord at this point.
        if len(p) != self._X.shape[0]:
            raise ValueError(
                "'p' size should be consistent with the y coordinate of the dataset"
            )
        # change its default metadata if necessary
        # (i.e. if p was not provided as already a Coord).
        if p.name == p.id:
            p.name = "external variable"
        if p.title == "<untitled>":
            p.title = "$p$"
        if p.units is None:
            p.units = ""
        return p

    @tr.validate("_q")
    def _q_validate(self, proposal):
        q = proposal.value
        if isinstance(self._K, str) or callable(self._K):
            # case of a list
            if isinstance(q, list):
                if len(q) == 3:
                    q = np.linspace(q[0], q[1], q[2])
                else:
                    warning_(
                        "Provided q is a list a {len(q)} items. "
                        "It will be converted to a Coord object. "
                        "If this is not what you wanted, remember that only list "
                        "of strictly 3 items are treated differently."
                    )
                # Transform the list or array-like to Coord
                q = Coord(q)

            # q should now be a Coord in all cases
            if not isinstance(q, Coord):
                raise ValueError(
                    "q must be provided as a list of 3 items or a array-like object "
                    "that can be casted to a Coord object"
                )

            # At this point, q is surely a Coord.
            # change its default metadata if necessary.
            # (i.e. if q was not provided as already a Coord).
            if q.name == q.id:
                q.name = "internal variable"
            if q.title == "<untitled>":
                q.title = "$q$"
            if q.units is None:
                q.units = ""
        else:
            pass
            # K was probably defined as a NDDataset, so q will be provided
            # as the default
        return q

    # ----------------------------------------------------------------------------------
    # Kernel property
    # ----------------------------------------------------------------------------------
    @property
    def kernel(self):

        _adsorption = ["langmuir", "ca"]
        _kinetics = ["reactant-first-order", "product-first-order"]
        _diffusion = ["diffusion"]

        K = self._K
        p = self._p.copy()
        q = self._q.copy()

        qdefault = q.name == "internal variable" and q.title == "$q$"
        pdefault = p.name == "external variable" and p.title == "$p$"

        if isinstance(K, str):

            if K.lower() not in _adsorption + _kinetics + _diffusion:
                raise NotImplementedError(
                    f"Kernel type `{K.lower()}` is not implemented"
                )

            elif K.lower() in _adsorption:
                title = "coverage"

                # change default metadata
                if qdefault:
                    q.name = "reduced adsorption energy"
                    q.title = "$\Delta_{ads}G^{0}/RT$"

                if pdefault:
                    p.name = "relative pressure"
                    p.title = "$p_/p_^{0}$"

                if K.lower() == "langmuir":
                    kernel = (
                        np.exp(-q.data)
                        * p.data[:, None]
                        / (1 + np.exp(-q.data) * p.data[:, None])
                    )

                else:  # 'ca'
                    kernel = np.ones((len(p.data), len(q.data)))
                    kernel[p.data[:, None] < q.data] = 0

            elif K.lower() in _kinetics:
                title = "coverage"

                # change default metadata
                if qdefault:
                    q.name = "Ln of rate constant"
                    q.title = "$\Ln k$"
                if pdefault:
                    # change default values and units
                    p.name = "time"
                    p.title = "$t$"
                    p.units = "s"

                if K.lower() == "reactant-first-order":
                    kernel = np.exp(-1 * np.exp(q.data) * p.data[:, None])

                else:  # 'product-first-order'
                    kernel = 1 - np.exp(-1 * np.exp(q.data) * p.data[:, None])

            elif K.lower() in _diffusion:
                title = "fractional uptake"

                # change default metadata
                if qdefault:
                    q.name = "Diffusion rate constant"
                    q.title = "$\\tau^{-1}$"
                    # q.to('1/s', force=True)
                if pdefault:
                    p.name = "time"
                    p.title = "$t$"
                    # p.to('s', force=True)

                kernel = np.zeros((p.size, q.size))
                for n in np.arange(1, 100):
                    kernel += (1 / n**2) * np.exp(
                        -(1 / 9) * n**2 * np.pi**2 * q.data * p.data[:, None]
                    )
                kernel = 1 - (6 / np.pi**2) * kernel

        elif callable(K):
            kernel = K(p.data, q.data)
            title = ""

        else:
            raise ValueError("K must be a str or a callable")

        # weighting coefficients for the numerical quadrature of the Fredholm integral
        w = np.zeros((q.size))
        w[0] = 0.5 * (q.data[-1] - q.data[0]) / (q.size - 1)
        w[1:-1] = 2 * w[0]
        w[-1] = w[0]

        kernel = kernel * w
        if isinstance(K, str):
            name = K + " kernel matrix"
        else:
            name = "kernel matrix"
        dims = ["y", "x"]
        out = NDDataset(
            kernel, dims=dims, coordset=CoordSet(y=p, x=q), title=title, name=name
        )

        return out


@signature_has_configurable_traits
class IRIS(DecompositionAnalysis):

    _docstring.delete_params("DecompositionAnalysis.see_also", "IRIS")

    __doc__ = _docstring.dedent(
        """
    Integral inversion solver for spectroscopic data (IRIS).

    `IRIS`, a model developed by :cite:t:`stelmachowski:2013`\ , solves integral
    equation of the first kind of 1 or 2 dimensions, *i.e.,*
    finds a distribution function :math:`f(p)` or :math:`f(c,p)` of contributions to
    univariate data :math:`a(p)` or multivariate :math:`a(c, p)` data evolving with an
    external experimental variable :math:`p` (time, pressure, temperature,
    concentration, ...) according to the integral transform:

    .. math:: a(c, p) = \int_{min}^{max} k(q, p) f(c, q) dq

    .. math:: a(p) = \int_{min}^{max} k(q, p) f(q) dq

    where the kernel :math:`k(q, p)` expresses the functional dependence of a single
    contribution with respect to the experimental variable :math:`p` and 'internal'
    physico-chemical variable :math:`q` .

    Regularization is triggered when `reg_par` is set to an array of two or three
    values.

    If `reg_par` has two values [``min``\ , ``max``\ ], the optimum regularization
    parameter is searched between :math:`10^{min}` and :math:`10^{max}`\ .
    Automatic search of the regularization is made using the Cultrera_Callegaro
    algorithm (:cite:p:cultrera:2020) which involves the Menger curvature of a
    circumcircle and the golden section search method.

    If three values are given ([``min``\ , ``max``\ , ``num``\ ]), then the inversion
    will be made for ``num`` values evenly spaced on a log scale between
    :math:`10^{min}` and :math:`10^{max}`\ .

    Parameters
    ----------
    %(AnalysisConfigurable.parameters)s

    See Also
    --------
    %(DecompositionAnalysis.see_also.no_IRIS)s
    """
    )

    # ----------------------------------------------------------------------------------
    # Runtime Parameters (in addition to those of AnalysisConfigurable)
    # ----------------------------------------------------------------------------------
    _Y = tr.Union(
        (
            tr.Instance(IrisKernel),
            NDDatasetType(),
        ),
        default_value=None,
        allow_none=True,
        help="Target/profiles taken into account to fit a model",
    )
    _Y_preprocessed = Array(help="preprocessed Y")
    _q = CoordType()
    _channels = CoordType()
    _lambdas = CoordType()
    _regularization = tr.Bool(False)
    _search_reg = tr.Bool(False)

    # ----------------------------------------------------------------------------------
    # Configuration parameters
    # ----------------------------------------------------------------------------------
    reg_par = tr.List(
        minlen=2,
        maxlen=3,
        default_value=None,
        allow_none=True,
        help="Regularization parameter (two values [ ``min`` , ``max`` ] "
        "or three values [ ``start`` , ``stop`` , ``num`` ]. "
        "If `reg_par` is None, no :term:`regularization` is applied.",
    ).tag(config=True)

    # ----------------------------------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------------------------------
    def __init__(
        self,
        log_level="WARNING",
        warm_start=False,
        **kwargs,
    ):
        # call the super class for initialisation of the configuration parameters
        # to do before anything else!
        super().__init__(
            log_level=log_level,
            warm_start=warm_start,
            **kwargs,
        )
        # no validation of reg_par triggred when it is None
        # so we need to init self._lambdas manually else itt will not be inited
        if self.reg_par is None:
            self._lambdas = Coord([0], title="lambda")

    # ----------------------------------------------------------------------------------
    # Private validation and default getter methods
    # ----------------------------------------------------------------------------------
    @tr.validate("reg_par")
    def _reg_par_validate(self, proposal):
        reg_par = proposal.value
        if reg_par is None or len(reg_par) == 1:
            self._regularization = False
            self._search_reg = False
            _lambdas = [0]
        elif len(reg_par) == 2:
            self._regularization = True
            self._search_reg = True
            _lambdas = reg_par
        elif len(reg_par) == 3:
            self._regularization = True
            self._search_reg = False
            _lambdas = np.logspace(reg_par[0], reg_par[1], reg_par[2])
        else:
            raise ValueError(
                "reg_par should be either None or a set of 2 or 3 integers"
            )

        # create the lambdas coordinate
        self._lambdas = Coord(_lambdas, title="lambda")

        # return the validated reg_par with no transformation
        return reg_par

    @tr.validate("_Y")
    def _Y_validate(self, proposal):
        # validation of the _Y attribute: fired when self._Y is assigned
        # In this IRIS model, we can have either a IrisKernel object or a NDDataset
        Y = proposal.value
        # we need a dataset
        if isinstance(Y, IrisKernel):
            Y = Y.kernel
        return Y

    @tr.observe("_Y")
    def _preprocess_as_Y_changed(self, change):
        Y = change.new
        # store the coordinate q
        self._q = Y.coordset[Y.dims[-1]]
        # use data only
        self._Y_preprocessed = Y.data

    @tr.observe("_X")
    def _preprocess_as_X_changed(self, change):
        # we need the X.x axis (called channels) later in the IRIS calculation
        # get it from the self._X nddataset (where masked data have been removed)
        X = change.new
        self._channels = X.coordset[X.dims[-1]]
        # use data only
        self._X_preprocessed = X.data

    # ----------------------------------------------------------------------------------
    # Private methods (overloading abstract classes)
    # ----------------------------------------------------------------------------------
    def _fit(self, X, K):
        # X is the data array to fit
        # K is the kernel data array

        q = self._q.data
        lambdas = self._lambdas.data

        # define containers for outputs
        M, N, W = K.shape[-1], X.shape[-1], X.shape[0]  # noqa: F475
        L = len(lambdas) if not self._search_reg else 4
        f = np.zeros((L, M, N))
        RSS = np.zeros((L))
        SM = np.zeros((L))

        # Define S matrix (sharpness), see function _Smat() below
        info_("Build S matrix (sharpness)")
        S = _Smat(q)
        info_("... done")

        # Solve non-regularized problem
        if not self._regularization:
            msg = f"Solving for {N} channels and {W} observations, no regularization"
            info_(msg)

            # use scipy.nnls() to solve the linear problem: X = K f
            for j in range(N):
                f[0, :, j] = optimize.nnls(K, X[:, j].squeeze())[0]
            res = X - np.dot(K, f[0])
            RSS[0] = np.sum(res**2)
            SM[0] = np.linalg.norm(np.dot(np.dot(np.transpose(f[0]), S), f[0]))

            msg = f"-->  residuals = {RSS[0]:.2e}    curvature = {SM[0]:.2e}"
            info_(msg)

        else:  # regularization
            # some matrices used for QP optimization do not depend on lambdaR
            # and are computed here. The standard form used by quadprog() is
            # minimize (1/2) xT G x - aT x ; subject to: C.T x >= b

            # The first part of the G matrix is independent of lambda:
            #     G = G0 + 2 * lambdaR S
            G0 = 2 * np.dot(K.T, K)
            a = 2 * np.dot(X.T, K)
            C = np.eye(M)
            b = np.zeros(M)

            # --------------------------------------------------------------------------
            def solve_for_lambda(X, K, G0, lamda, S):
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
                M, N, _ = K.shape[-1], X.shape[-1], X.shape[0]
                fi = np.zeros((M, N))
                channels = self._channels
                for j, channel in enumerate(channels.data):
                    try:
                        G = G0 + 2 * lamda * S
                        fi[:, j] = quadprog.solve_qp(G, a[j].squeeze(), C, b)[0]
                    except ValueError:  # pragma: no cover
                        msg = (
                            f"Warning:G is not positive definite for log10(lambda)="
                            f"{np.log10(lamda):.2f} at {channel:.2f} "
                            f"{channels.units}, find nearest PD matrix"
                        )
                        warning_(msg)
                        try:
                            G = _nearestPD(G0 + 2 * lamda * S, 0)
                            fi[:, j] = quadprog.solve_qp(G, a[j].squeeze(), C, b)[0]
                        except ValueError:
                            msg = (
                                "... G matrix is still ill-conditioned, "
                                "try with a small shift of diagonal elements..."
                            )
                            warning_(msg)
                            G = _nearestPD(G0 + 2 * lamda * S, 1e-3)
                            fi[:, j] = quadprog.solve_qp(G, a[j].squeeze(), C, b)[0]

                resi = X.data - np.dot(K.data, fi)
                RSSi = np.sum(resi**2)
                SMi = np.linalg.norm(np.dot(np.dot(np.transpose(fi), S), fi))

                msg = (
                    f"log10(lambda)={np.log10(lamda):.3f} -->  "
                    f"residuals = {RSSi:.3e}    "
                    f"regularization constraint  = {SMi:.3e}"
                )
                info_(msg)

                return fi, RSSi, SMi

            # --------------------------------------------------------------------------

            if not self._search_reg:
                msg = (
                    f"Solving for {X.shape[1]} channels, {X.shape[0]} observations and "
                    f"{len(lambdas)} regularization parameters"
                )
                info_(msg)

                for i, lamda_ in enumerate(lambdas):
                    f[i], RSS[i], SM[i] = solve_for_lambda(X, K, G0, lamda_, S)

            else:
                msg = (
                    f"Solving for {X.shape[1]} channel(s) and {X.shape[0]} "
                    f"observations, search optimum regularization parameter "
                    f"in the range: [10**{min(lambdas)}, 10**{max(lambdas)}]"
                )
                info_(msg)

                x = np.zeros(4)
                epsilon = 0.1
                phi = (1 + np.sqrt(5)) / 2

                x[0] = min(lambdas)
                x[3] = max(lambdas)
                x[1] = (x[3] + phi * x[0]) / (1 + phi)
                x[2] = x[0] + x[3] - x[1]
                lambdas = 10**x
                msg = "Initial Log(lambda) values = " + str(x)
                info_(msg)

                for i, xi in enumerate(x):
                    f[i], RSS[i], SM[i] = solve_for_lambda(X, K, G0, 10**xi, S)

                Rx = np.copy(RSS)
                Sy = np.copy(SM)
                while "convergence not reached":
                    C1 = _menger(np.log10(Rx[0:3]), np.log10(Sy[0:3]))
                    C2 = _menger(np.log10(Rx[1:4]), np.log10(Sy[1:4]))
                    msg = (
                        f"Curvatures of the inner points: C1 = {C1:.3f} ;"
                        f" C2 = {C2:.3f}"
                    )
                    info_(msg)

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

                        f_, Rx[1], Sy[1] = solve_for_lambda(X, K, G0, 10 ** x[1], S)
                        lambdas = np.append(lambdas, np.array(10 ** x[1]))
                        f = np.concatenate((f, np.atleast_3d(f_.T).T))
                        RSS = np.concatenate((RSS, np.array(Rx[1:2])))
                        SM = np.concatenate((SM, np.array(Sy[1:2])))
                        C2 = _menger(np.log10(Rx[1:4]), np.log10(Sy[1:4]))
                        msg = f"new curvature: C2 = {C2:.3f}"
                        info_(msg)

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
                        f_, Rx[1], Sy[1] = solve_for_lambda(X, K, G0, 10 ** x[1], S)
                        f = np.concatenate((f, np.atleast_3d(f_.T).T))
                        lambdas = np.append(lambdas, np.array(10 ** x[1]))
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
                        f_, Rx[2], Sy[2] = solve_for_lambda(X, K, G0, 10 ** x[2], S)
                        f = np.concatenate((f, np.atleast_3d(f_.T).T))
                        lambdas = np.append(lambdas, np.array(10 ** x[2]))
                        RSS = np.concatenate((RSS, np.array(Rx[1:2])))
                        SM = np.concatenate((SM, np.array(Sy[1:2])))
                    if (10 ** x[3] - 10 ** x[0]) / 10 ** x[3] < epsilon:
                        break
                id_opt = np.argmin(np.abs(lambdas - np.power(10, x_)))
                id_opt_ranked = np.argmin(np.abs(np.argsort(lambdas) - id_opt))
                msg = (
                    f" optimum found: index = {id_opt_ranked} ; "
                    f"Log(lambda) = {x_:.3f} ; "
                    f"lambda = {np.power(10, x_):.5e} ; curvature = {C_:.3f}"
                )
                info_(msg)

            # sort by lamba values
            argsort = np.argsort(lambdas)
            lambdas = lambdas[argsort]
            RSS = RSS[argsort]
            SM = SM[argsort]
            f = f[argsort]

        info_("Done.")

        self._lambdas.data = lambdas
        _outfit = f, RSS, SM
        return _outfit

    # ----------------------------------------------------------------------------------
    # Public methods and property
    # ----------------------------------------------------------------------------------
    @property
    def f(self):
        if self._X_is_missing:
            raise NotFittedError

        f = self._outfit[0]
        f = NDDataset(f, name="2D distribution functions", title="density")
        f.history = "2D IRIS analysis of {X.name} dataset"
        f.set_coordset(z=self._lambdas, y=self._q, x=self._channels)
        if np.any(self._X_mask):
            # restore masked column if necessary
            f = self._restore_masked_data(f, axis=-1)
        return f

    @property
    def K(self):
        return self._Y  # Todo: eventuallly restore row mask

    @property
    def q(self):
        return self._q

    @property
    def lambdas(self):
        return self._lambdas

    @property
    def RSS(self):
        return self._outfit[1]

    @property
    def SM(self):
        return self._outfit[2]

    def inverse_transform(self):  # override the decomposition method
        """
        Transform data back to the original space.

        The following matrix operation is performed : :math:`\hat{X} = K.f[i]`
        for each value of the regularization parameter.

        Returns
        -------
        `~spectrochempy.core.dataset.nddataset.NDDataset`
            The reconstructed dataset.
        """
        if not self._fitted:
            raise NotFittedError("The fit method must be used before using this method")

        lambdas = self._lambdas.data
        X = self.X
        K = self._Y
        f = self.f

        if len(lambdas) == 1:  # no regularization or single lambda
            X_hat = NDDataset(np.zeros((X.shape)), title=X.title, units=X.units)
            X_hat.set_coordset(y=X.y, x=X.x)
            X_hat.data = np.dot(K.data, f.data.squeeze(axis=0))
        else:
            X_hat = NDDataset(
                np.zeros((f.z.size, *X.shape)),
                title=X.title,
                units=X.units,
            )
            X_hat.set_coordset(z=f.z, y=X.y, x=X.x)
            for i in range(X_hat.z.size):
                X_hat.data[i] = np.dot(K.data, f[i].data.squeeze(axis=0))

        X_hat.name = "2D-IRIS Reconstructed datasets"
        return X_hat

    def plotlcurve(self, scale="ll", title="L curve"):
        """
        Plot the ``L-Curve``\ .

        Parameters
        ----------
        scale : `str`\ , optional, default: ``'ll'``
            String of 2 letters among ``'l'``\ (log) or ``'n'``\ (non-log) indicating
            whether the ``y`` and ``x`` axes should be log scales.
        title : `str`, optional, default: ``'L-curve'``
            Plot title.

        Returns
        -------
        `~matplotlib.axes.Axes`
                The matplotlib axe.
        """
        if not self._fitted:
            raise NotFittedError("The fit method must be used before using this method")

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(title)
        plt.plot(self.RSS, self.SM, "o")
        ax.set_xlabel("Residuals")
        ax.set_ylabel("Curvature")
        if scale[1] == "l":
            ax.set_xscale("log")
        if scale[0] == "l":
            ax.set_yscale("log")
        return ax

    @_docstring.dedent
    def plotmerit(self, index=None, **kwargs):
        """
        Plot the input dataset, reconstructed dataset and residuals.

        Parameters
        ----------
        index : `int`\ , `list` or `tuple` of `int`\ , optional, default: `None`
            Index(es) of the inversions (*i.e.,* of the lambda values) to consider.
            If `None` plots for all indices.
        %(kwargs)s

        Returns
        -------
        `list` of `~matplotlib.axes.Axes`
            Subplots.

        Other Parameters
        ----------------
        %(plotmerit.other_parameters)s
        """
        X = self.X
        X_hat = self.inverse_transform()
        axeslist = []
        if index is None:
            index = range(len(self._lambdas))
        if type(index) is int:
            index = [index]

        for i in index:
            if X_hat.ndim == 3:  # if several lambda
                X_hat_ = X_hat[i].squeeze()
            else:
                X_hat_ = X_hat  # if single lambda or no regularization

            ax = super().plotmerit(X, X_hat_, **kwargs)

            ax.set_title(
                f"2D IRIS merit plot, $\lambda$ = {self._lambdas[i].value:.2e}"
            )
            axeslist.append(ax)

        return axeslist

    def plotdistribution(self, index=None, **kwargs):
        """
        Plot the distribution function.

        This function plots the distribution function f of the `IRIS` object.

        Parameters
        ----------
        index : `int` , `list` or `tuple` of `int`, optional, default: `None`
            Index(es) of the inversions (i.e. of the :term:`regularization` parameter)
            to consider.
            If `None`, plots for all indices.
        **kwargs
            Other optional arguments are passed in the plots.

        Returns
        -------
        `list` of `~matplotlib.axes.Axes`
            Subplots.
        """

        axeslist = []
        if index is None:
            index = range(len(self._lambdas))
        if type(index) is int:
            index = [index]
        for i in index:
            axeslist.append(self.f[i].plot(method="map", **kwargs))
        return axeslist


# --------------------------------------------------------------------------------------
# Utility private functions


def _menger(x, y):
    """
    returns the Menger curvature of a triplet of
    points. x, y = sets of 3 cartesian coordinates
    """
    numerator = 2 * (((x[1] - x[0]) * (y[2] - y[1])) - ((y[1] - y[0]) * (x[2] - x[1])))
    if abs(numerator) <= EPSILON:
        return 0.0

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

    S = ((q[m - 1] - q[0]) / (m - 1)) ** (-3) * S
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
    # for `np.spacing` ), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)` , since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrices of small dimension, be on
    # the order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    Ie = np.eye(A.shape[0])
    k = 1
    while not _isPD(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += Ie * (-mineig * k**2 + spacing)
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
