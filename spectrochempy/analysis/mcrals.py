# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory.
# ======================================================================================================================
"""
This module implements the MCRALS class.
"""

__all__ = ["MCRALS"]

__dataset_methods__ = []

import numpy as np
from traitlets import HasTraits, Instance, Dict, Unicode

from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.analysis.pca import PCA
from spectrochempy.core.dataset.npy import dot
from spectrochempy.core import info_, set_loglevel, INFO


class MCRALS(HasTraits):
    """
    Performs MCR-ALS of a dataset knowing the initial C or St matrix.

    MCR-ALS (Multivariate Curve Resolution - Alternating Least Squares) resolve"s a set (or several sets) of
    spectra X of an evolving mixture (or a set of mixtures) into the spectra St of ‘pure’ species and their
    concentration profiles C. In terms of matrix equation:
    .. math::`X = CS^T + E`
    where :math:`E` is the lmatrix of residuals.

    Parameters
    ----------
    dataset : NDDataset
        The dataset on which to perform the MCR-ALS analysis.

    guess : NDDataset
        Initial concentration or spectra.

    **kwargs
        Optional parameters, see Other parameters below.

    Other Parameters
    ----------------

        tol : float, optional, default 0.1
        Convergence criterion on the change of residuals (percent change of standard deviation of residuals).

    maxit : int, optional, default=50
        Maximum number of ALS minimizations.

    maxdiv : int, optional, default=5
        Maximum number of successive non-converging iterations.

    nonnegConc : str or array of indexes, default `"all"`
        Non negativity constraint on concentrations. If set to `'all'` (default) all concentrations profiles are
        considered non-negative. If an array of indexes is passed, the corresponding profiles are considered
        non-negative, not the others. For instance `[0, 2]` indicates that profile #0 and #2 are non-negative while
         profile #1 *can* be negative. If set to `"None"` or `[]`, all profiles can be negative.

    unimodConc : str or array of indexes, default `"all"`
        Unimodality constraint on concentrations. If set to `'all'` (default) all concentrations profiles are
        considered unimodal. If an array of indexes is passed, the corresponding profiles are considered
        unimodal, not the others. For instance `[0, 2]` indicates that profile #0 and #2 are unimodal while
         profile #1 *can* be multimodal. If set to `"None"` or `[]`, all profiles can be multimodal.

    unimodMod : str, default "strict"
        When set to `"strict"`, values deviating from unimodality are reset to the value of the previous point.
        When set to `"smooth"`, both values (deviating point and previous point) are modified to avoid ="steps"
        in the concentration profile.

    unimodTol : float, default 1.1
        Tolerance parameter for unimodality. Correction is applied only if:
        ```C[i,j] > C[i-1,j] * unimodTol```  on the decreasing branch of profile #j,
        ```C[i,j] < C[i-1,j] * unimodTol```  on the increasing branch of profile  #j.

    monoDecConc : `None` or array of indexes
        Monotonic decrease constraint on concentrations.  If set to `None` (default) or `[]` no constraint is applied.
        If an array of indexes is passed, the corresponding profiles are considered do decrease monotonically, not
         the others. For instance `[0, 2]` indicates that profile #0 and #2 are decreasing while profile #1 *can*
         increase.

    monoDecTol : float, default 1.1
         Tolerance parameter for monotonic decrease. Correction is applied only if:
        ```C[i,j] > C[i-1,j] * unimodTol```  along profile #j.

    monoIncConc : `None` or array of indexes
        Monotonic increase constraint on concentrations.  If set to `None` (default) or `[]` no constraint is applied.
        If an array of indexes is passed, the corresponding profiles are considered to increase monotonically, not
         the others. For instance `[0, 2]` indicates that profile #0 and #2 are increasing while profile #1 *can*
         decrease.

    monoIncTol : float, default 1.1
            Tolerance parameter for monotonic decrease. Correction is applied only if:
        ```C[i,j] < C[i-1,j] * unimodTol``` along profile  #j.

    closureConc : None or array of indexes, default `None`
        Defines the concentration profiles subjected to closure constraint. If set to `None` or `[]`, no constraint is
        applied. If an array of indexes is passed, the corresponding profile will be constrained so that their weighted
        sum equals the `closureTarget`.

    closureTarget : str or array of float, default `"default"`
        The value of the sum of concentrations profiles subjected to closure.  If set to `default`, the total
        concentration is set to 1.0 for all observations. If an array is passed: the values of concentration for each
        observation. Hence,`np.ones[X.shape[0]` would be equivalent to "default".

    closureMethod : str, `"scaling"` or `"constantSum"`, default `"scaling"`
        The method used to enforce Closure. "scaling" recompute the concentration profiles using linear algebra:
        ```
        C.data[:, closureConc] = np.dot(C.data[:, closureConc],
                                        np.diag(np.linalg.lstsq(C.data[:, closureConc], closureTarget.T, rcond=None)[0]))
        ```
        `"constantSum"` normalize the sum of concentration profiles to `closureTarget`.

    hardConc : None or or array of indexes, default `None`
        Defines hard constraints obn the concentration profiles. If set to `None` or `[]`, no constraint is
        applied. If an array of indexes is passed, the corresponding profiles will set by `getC` (see below).

    getC : Callable
        An external function that will provide `len(hardConc)` concentration profiles:
        ```
        getC(*args) -> hardC
        ```
        or:
        ```
        getC(*args) -> (hardC, out2, out3, ...)
        ```
        where *args are the parameters needed to completely specify the function (see , `harC` is a nadarray or NDDataset
        of shape `(C.y, len(hardConc)`, and `out1`, `out2`, ... are supplementary outputs returned by the function.

    argsGetC : tuple, optional
        Extra arguments passed to the external function.

    hardC_to_C_idx : None or list or tuple, default None
        Indicates the correspondence between the indexes of the columns of hardC and of the C matrix. [1, None, 0]
        indicates that the first profile in hardC (index O) corrsponds to the second profile of C (index 1).

    nonnegSpec : str, list or tuple, default `"all"`
        Indicates non-negative spectral profile. If set to `'all'` (default) all spectral profiles are
        considered non-negative. If an array of indexes is passed, the corresponding profiles are considered
        non-negative, not the others. For instance `[0, 2]` indicates that profile #0 and #2 are non-negative while
         profile #1 *can* be negative. If set to `"None"` or `[]`, all profiles can be negative.

    normSpec : None or str, default None
        Defines whether the spectral profiles should be normalized. If set to `None` no normalization is applied.
        when set to "euclid", spectra are normalized with respect to their total area, when set to "max", spectra are
        normalized with respect to the maximum af their value.

    See Also
    --------
    PCA
       Performs MCR-ALS of a |NDDataset| knowing the initial C or St matrix.
    NNMF
       Performs a Non Negative Matrix Factorization of a |NDDataset|.
    EFA
       Perform an Evolving Factor Analysis (forward and reverse) of the input |NDDataset|.

    Examples
    --------
    >>> data = scp.read("matlabdata/als2004dataset.MAT")
    >>> X, guess = data[-1], data[3]
    >>> mcr = scp.MCRALS(X, guess)
    >>> mcr.St, mcr.St
    (NDDataset: [float64] unitless (shape: (y:4, x:96)), NDDataset: [float64] unitless (shape: (y:4, x:96)))
    """

    _X = Instance(NDDataset)
    _C = Instance(NDDataset, allow_none=True)
    _fixedC = Instance(NDDataset, allow_none=True)
    _extOutput = Instance(NDDataset, allow_none=True)
    _St = Instance(NDDataset, allow_none=True)
    _logs = Unicode
    _params = Dict()

    def __init__(self, dataset, guess, **kwargs):
        # list all default arguments:
        # Todo: add unimodSpec (and unimodSpecMod, ...), default `None`
        tol = kwargs.get("tol", 0.1)
        maxit = kwargs.get("maxit", 50)
        maxdiv = kwargs.get("maxdiv", 5)
        nonnegConc = kwargs.get("nonnegConc", "all")
        unimodConc = kwargs.get("unimodConc", "all")
        unimodTol = kwargs.get("unimodTol", 1.1)
        unimodMod = kwargs.get("unimodMod", "strict")
        monoDecConc = kwargs.get("monoDecConc", None)
        monoIncTol = kwargs.get("monoIncTol", 1.1)
        monoIncConc = kwargs.get("monoIncConc", None)
        monoDecTol = kwargs.get("monoDecTol", 1.1)
        closureConc = kwargs.get("closureConc", None)
        closureTarget = kwargs.get("closureTarget", "default")
        closureMethod = kwargs.get("closureMethod", "scaling")
        hardConc = kwargs.get("hardConc", None)
        getConc = kwargs.get("getConc", None)
        argsGetConc = kwargs.get("argsGetConc", None)
        hardC_to_C_idx = kwargs.get("hardC_to_C_idx", "default")
        nonnegSpec = kwargs.get("nonnegSpec", "all")
        normSpec = kwargs.get("normSpec", None)
        verbose = kwargs.get("verbose", False)

        # now check input
        if verbose:
            set_loglevel(INFO)

        # Check initial data
        # ------------------------------------------------------------------------

        initConc, initSpec = False, False

        if type(guess) is np.ndarray:
            guess = NDDataset(guess)

        X = dataset

        if X.shape[0] == guess.shape[0]:
            initConc = True
            C = guess.copy()
            C.name = "Pure conc. profile, mcs-als of " + X.name
            nspecies = C.shape[1]

        elif X.shape[1] == guess.shape[1]:
            initSpec = True
            St = guess.copy()
            St.name = "Pure spectra profile, mcs-als of " + X.name
            nspecies = St.shape[0]

        else:  # pragma: no cover
            raise ValueError(
                "the dimensions of initial concentration "
                "or spectra dataset do not match the data"
            )

        ny, _ = X.shape

        # makes a PCA with same number of species for further comparison
        Xpca = PCA(X).reconstruct(n_pc=nspecies)

        # reset default text to indexes
        # ------------------------------

        if nonnegConc == "all":
            nonnegConc = np.arange(nspecies)
        elif nonnegConc is None:  # pragma: no cover
            nonnegConc = []
        elif (
            len(nonnegConc) > nspecies or max(nonnegConc + 1) > nspecies
        ):  # pragma: no cover
            raise ValueError(
                f"The guess has only {nspecies} species, please check nonnegConc"
            )

        if unimodConc == "all":
            unimodConc = np.arange(nspecies)
        elif (
            len(unimodConc) > nspecies or max(unimodConc + 1) > nspecies
        ):  # pragma: no cover
            raise ValueError(
                f"The guess has only {nspecies} species, please check unimodConc"
            )

        if closureTarget == "default":
            closureTarget = np.ones(ny)
        elif len(closureTarget) > ny:  # pragma: no cover
            raise ValueError(
                f"The data contain only {ny} observations, please check closureTarget"
            )

        if hardC_to_C_idx == "default":
            hardC_to_C_idx = np.arange(nspecies)
        elif (
            len(hardC_to_C_idx) > nspecies or max(hardC_to_C_idx + 1) > nspecies
        ):  # pragma: no cover
            raise ValueError(
                f"The guess has only {nspecies} species, please check fixedConc_to_C_idx"
            )

        # constraints on spectra
        if nonnegSpec == "all":
            nonnegSpec = np.arange(nspecies)
        elif (
            len(nonnegSpec) > nspecies or max(nonnegSpec + 1) > nspecies
        ):  # pragma: no cover
            raise ValueError(
                f"The guess has only {nspecies} species, please check nonnegSpec"
            )

        # Compute initial spectra or concentrations   (first iteration...)
        # ------------------------------------------------------------------------

        if initConc:
            if C.coordset is None:
                C.set_coordset(y=X.y, x=C.x)
            St = NDDataset(np.linalg.lstsq(C.data, X.data, rcond=None)[0])
            St.name = "Pure spectra profile, mcs-als of " + X.name
            St.title = X.title
            cy = C.x.copy() if C.x else None
            cx = X.x.copy() if X.x else None
            St.set_coordset(y=cy, x=cx)

        if initSpec:
            if St.coordset is None:
                St.set_coordset(y=St.y, x=X.x)
            Ct = np.linalg.lstsq(St.data.T, X.data.T, rcond=None)[0]
            C = NDDataset(Ct.T)
            C.name = "Pure conc. profile, mcs-als of " + X.name
            C.title = "concentration"
            cx = St.y.copy() if St.y else None
            cy = X.y.copy() if X.y else None
            C.set_coordset(y=cy, x=cx)

        change = tol + 1
        stdev = X.std()
        niter = 0
        ndiv = 0

        logs = "*** ALS optimisation log***\n"
        logs += "#iter     Error/PCA        Error/Exp      %change\n"
        logs += "---------------------------------------------------"
        info_(logs)

        while change >= tol and niter < maxit and ndiv < maxdiv:

            C.data = np.linalg.lstsq(St.data.T, X.data.T, rcond=None)[0].T
            niter += 1

            # Force non-negative concentration
            # --------------------------------
            if nonnegConc is not None:
                for s in nonnegConc:
                    C.data[:, s] = C.data[:, s].clip(min=0)

            # Force unimodal concentration
            # ----------------------------
            if unimodConc is not None:
                for s in unimodConc:
                    maxid = np.argmax(C.data[:, s])
                    curmax = C.data[maxid, s]
                    curid = maxid

                    while curid > 0:
                        curid -= 1
                        if C.data[curid, s] > curmax * unimodTol:
                            if unimodMod == "strict":
                                C.data[curid, s] = C.data[curid + 1, s]
                            if unimodMod == "smooth":
                                C.data[curid, s] = (
                                    C.data[curid, s] + C.data[curid + 1, s]
                                ) / 2
                                C.data[curid + 1, s] = C.data[curid, s]
                                curid = curid + 2
                        curmax = C.data[curid, s]

                    curid = maxid
                    while curid < ny - 1:
                        curid += 1
                        if C.data[curid, s] > curmax * unimodTol:
                            if unimodMod == "strict":
                                C.data[curid, s] = C.data[curid - 1, s]
                            if unimodMod == "smooth":
                                C.data[curid, s] = (
                                    C.data[curid, s] + C.data[curid - 1, s]
                                ) / 2
                                C.data[curid - 1, s] = C.data[curid, s]
                                curid = curid - 2
                        curmax = C.data[curid, s]

            # Force monotonic increase
            # ------------------------
            if monoIncConc is not None:
                for s in monoIncConc:
                    for curid in np.arange(ny - 1):
                        if C.data[curid + 1, s] < C.data[curid, s] / monoIncTol:
                            C.data[curid + 1, s] = C.data[curid, s]

            # Force monotonic decrease
            # ----------------------------------------------
            if monoDecConc is not None:
                for s in monoDecConc:
                    for curid in np.arange(ny - 1):
                        if C.data[curid + 1, s] > C.data[curid, s] * monoDecTol:
                            C.data[curid + 1, s] = C.data[curid, s]

            # Closure
            # ------------------------------------------
            if closureConc is not None:
                if closureMethod == "scaling":
                    Q = np.linalg.lstsq(
                        C.data[:, closureConc], closureTarget.T, rcond=None
                    )[0]
                    C.data[:, closureConc] = np.dot(C.data[:, closureConc], np.diag(Q))
                elif closureMethod == "constantSum":
                    totalConc = np.sum(C.data[:, closureConc], axis=1)
                    C.data[:, closureConc] = (
                        C.data[:, closureConc]
                        * closureTarget[:, None]
                        / totalConc[:, None]
                    )

            # external concentration profiles
            # ------------------------------------------
            if hardConc is not None:
                extOutput = getConc(
                    *(
                        (
                            C,
                            hardConc,
                            hardC_to_C_idx,
                        )
                        + argsGetConc
                    )
                )
                if isinstance(extOutput, dict):
                    fixedC = extOutput["concentrations"]
                    argsGetConc = extOutput["new_args"]
                else:
                    fixedC = extOutput
                if type(fixedC) is NDDataset:
                    extC = fixedC.data
                C.data[:, hardConc] = fixedC[:, hardC_to_C_idx]

            # stores C in C_hard
            Chard = C.copy()

            # compute St
            St.data = np.linalg.lstsq(C.data, X.data, rcond=None)[0]

            # stores St in Stsoft
            Stsoft = St.copy()

            # Force non-negative spectra
            # --------------------------
            if nonnegSpec is not None:
                St.data[nonnegSpec, :] = St.data[nonnegSpec, :].clip(min=0)

            # recompute C for consistency(soft modeling)
            C.data = np.linalg.lstsq(St.data.T, X.data.T)[0].T

            # rescale spectra & concentrations
            if normSpec == "max":
                alpha = np.max(St.data, axis=1).reshape(nspecies, 1)
                St.data = St.data / alpha
                C.data = C.data * alpha.T
            elif normSpec == "euclid":
                alpha = np.linalg.norm(St.data, axis=1).reshape(nspecies, 1)
                print(alpha)
                St.data = St.data / alpha
                C.data = C.data * alpha.T

            # compute residuals
            # -----------------
            X_hat = dot(C, St)
            stdev2 = (X_hat - X.data).std()
            change = 100 * (stdev2 - stdev) / stdev
            stdev = stdev2

            stdev_PCA = (X_hat - Xpca.data).std()  #

            logentry = "{:3d}      {:10f}      {:10f}      {:10f}".format(
                niter, stdev_PCA, stdev2, change
            )
            logs += logentry + "\n"
            info_(logentry)

            if change > 0:
                ndiv += 1
            else:
                ndiv = 0
                change = -change

            if change < tol:
                logentry = "converged !"
                logs += logentry + "\n"
                info_(logentry)

            if ndiv == maxdiv:
                logline = (
                    f"Optimization not improved since {maxdiv} iterations... unconverged "
                    f"or 'tol' set too small ?\n"
                )
                logline += "Stop ALS optimization"
                logs += logline + "\n"
                info_(logline)

            if niter == maxit:
                logline = "Convergence criterion ('tol') not reached after {:d} iterations.".format(
                    maxit
                )
                logline += "Stop ALS optimization"
                logs += logline + "\n"
                info_(logline)

        self._X = X
        self._params = {
            "tol": tol,
            "maxit": maxit,
            "maxdiv": maxdiv,
            "nonnegConc": nonnegConc,
            "unimodConc": unimodConc,
            "unimodTol": unimodTol,
            "unimodMod": unimodMod,
            "closureConc": closureConc,
            "closureTarget ": closureTarget,
            "closureMethod": closureMethod,
            "monoDecConc": monoDecConc,
            "monoDecTol": monoDecTol,
            "monoIncConc": monoIncConc,
            "monoIncTol": monoIncTol,
            "hardConc": hardConc,
            "getConc": getConc,
            "argsGetConc": argsGetConc,
            "hardC_to_C_idx": hardC_to_C_idx,
            "nonnegSpec": nonnegSpec,
            "normSpec": normSpec,
            "verbose": verbose,
        }

        self._C = C
        if hardConc is not None:
            self._fixedC = extC
            self._extOutput = extOutput
        else:
            self._fixedC = None
            self._extOutput = None

        self._St = St
        self._logs = logs

        self._Stsoft = Stsoft
        self._Chard = Chard

    @property
    def X(self):
        """
        The original dataset.
        """
        return self._X

    @property
    def fixedC(self):
        """
        The last concentration profiles including external profiles.
        """
        return self._fixedC

    @property
    def extOutput(self):
        """
        The last output of the external function used to get.
        """
        return self._extOutput

    @property
    def C(self):
        """
        The final concentration profiles.
        """
        return self._C

    @property
    def St(self):
        """
        The final spectra profiles.
        """
        return self._St

    @property
    def Stsoft(self):
        """
        The soft spectra profiles.
        """
        return self._Stsoft

    @property
    def Chard(self):
        """
        The hard concentration profiles.
        """
        return self._Chard

    @property
    def params(self):
        """
        The parameters used to perform the MCR als.
        """
        return self._params

    @property
    def logs(self):
        """
        Logs output.
        """
        return self._logs

    def reconstruct(self):
        """
        Transform data back to the original space.

        The following matrice operation is performed : :math:`X'_{hat} = C'.S'^t`.

        Returns
        -------
        X_hat : |NDDataset|
            The reconstructed dataset based on the MCS-ALS optimization.
        """

        # reconstruct from concentration and spectra profiles
        C = self.C
        St = self.St

        X_hat = dot(C, St)

        X_hat.history = "Dataset reconstructed by MCS ALS optimization"
        X_hat.title = "X_hat: " + self.X.title
        return X_hat

    def plotmerit(self, **kwargs):
        """
        Plots the input dataset, reconstructed dataset and residuals.

        Parameters
        ----------
        **kwargs
            optional "colors" argument: tuple or array of 3 colors for :math:`X`, :math:`\hat X` and :math:`E`.

        Returns
        -------
        ax
            subplot.
        """
        colX, colXhat, colRes = kwargs.get("colors", ["blue", "green", "red"])

        X_hat = self.reconstruct()
        res = self.X - X_hat
        ax = self.X.plot()
        if self.X.x is not None:
            ax.plot(self.X.x.data, X_hat.T.data, color=colXhat)
            ax.plot(self.X.x.data, res.T.data, color=colRes)
        else:
            ax.plot(X_hat.T.data, color=colXhat)
            ax.plot(res.T.data, color=colRes)
        ax.autoscale(enable=True)
        ax.set_title("MCR ALS merit plot")
        return ax
