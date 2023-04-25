# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
This module implement the SIMPLISMA class.
"""

__all__ = ["SIMPLISMA"]
__configurables__ = ["SIMPLISMA"]

from warnings import warn

import numpy as np
import traitlets as tr

from spectrochempy.analysis._base import DecompositionAnalysis
from spectrochempy.core import info_
from spectrochempy.utils import exceptions
from spectrochempy.utils.decorators import deprecated, signature_has_configurable_traits
from spectrochempy.utils.docstrings import _docstring


# ======================================================================================
# class SIMPLISMA
# ======================================================================================
@signature_has_configurable_traits
class SIMPLISMA(DecompositionAnalysis):
    _docstring.delete_params("DecompositionAnalysis.see_also", "SIMPLISMA")

    __doc__ = _docstring.dedent(
        """
    SIMPLe to use Interactive Self-modeling Mixture Analysis (SIMPLISMA).

    This class performs a SIMPLISMA analysis of a 2D `NDDataset` .
    The algorithm is adapted from :cite:t:`windig:1997`\ .

    Parameters
    ----------
    %(AnalysisConfigurable.parameters)s

    See Also
    --------
    %(DecompositionAnalysis.see_also.no_SIMPLISMA)s
    """
    )

    # TODO : adapt to 3DDataset ?

    # ----------------------------------------------------------------------------------
    # Runtime Parameters,
    # only those specific to PCA, the other being defined in AnalysisConfigurable.
    # ----------------------------------------------------------------------------------
    # define here only the variable that you use in fit or transform functions

    # ----------------------------------------------------------------------------------
    # Configuration parameters
    # They will be written in a file from which the default can be modified)
    # Obviously, the parameters can also be modified at runtime as usual by assignment.
    # ----------------------------------------------------------------------------------
    interactive = tr.Bool(
        default_value=False,
        help=(
            "If True, the determination of purest variables is carried out "
            "interactively."
        ),
    ).tag(config=True)
    max_components = tr.Integer(
        default_value=2,
        help=(
            "The maximum number of pure compounds. Used only for non interactive"
            "analysis."
        ),
    ).tag(config=True)
    tol = tr.Float(
        default_value=0.1,
        help="The convergence criterion on the percent of unexplained variance.",
    ).tag(config=True)
    noise = tr.Float(
        default_value=3,
        help=(
            "A correction factor (%) for low intensity variables (0 - no offset, "
            "15 - large offset."
        ),
    ).tag(config=True)

    # ----------------------------------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------------------------------
    def __init__(
        self,
        *args,
        log_level="WARNING",
        warm_start=False,
        **kwargs,
    ):
        if len(args) > 0:
            raise ValueError(
                "Passing arguments such as SIMPLISMA(X) is now deprecated. "
                "Instead, use SIMPLISMA() followed by SIMPLISMA.fit(X). "
                "See the documentation and exemples"
            )

        # warn about deprecations
        # -----------------------
        if "verbose" in kwargs:
            deprecated("verbose", replace="log_level='INFO'", removed="0.6.5")
            verbose = kwargs.pop("verbose")
            if verbose:
                log_level = "INFO"

        # unimodMod deprecation
        if "n_pc" in kwargs:
            deprecated("n_pc", replace="max_components", removed="0.6.5")
            kwargs["max_components"] = kwargs.pop("n_pc")

        # call the super class for initialisation
        super().__init__(
            log_level=log_level,
            warm_start=warm_start,
            **kwargs,
        )

    # ----------------------------------------------------------------------------------
    # Private validation methods and default getter
    # ----------------------------------------------------------------------------------
    @tr.validate("max_components")
    def _max_components_validate(self, proposal):
        n = proposal.value
        if n < 2:
            raise ValueError(
                "Oh you did not just... 'MA' in simplisMA stands for Mixture Analysis. "
                "The number of pure compounds should be an integer larger than 2"
            )
        return n  # <-- do not forget this, or the returned value
        # for max_components is None

    @tr.default("_components")
    def _components_default(self):
        if self._fitted:
            return self._outfit[1]
        else:
            raise exceptions.NotFittedError(
                "The model was not yet fitted. Execute `fit` first!"
            )

    # ------------------------------------------------------------------------------
    # Utility functions
    # ------------------------------------------------------------------------------
    @staticmethod
    def _figures_of_merit(X, maxPIndex, C, St, j):
        # return %explained variance and stdev of residuals when the jth compound
        # is added
        C[:, j] = X[:, maxPIndex[j]]
        St[0 : j + 1, :] = np.linalg.lstsq(C[:, 0 : j + 1], X, rcond=None)[0]
        Xhat = np.dot(C[:, 0 : j + 1], St[0 : j + 1, :])
        res = Xhat - X
        stdev_res = np.std(res)
        rsquare = 1 - np.linalg.norm(res) ** 2 / np.linalg.norm(X) ** 2
        return rsquare, stdev_res

    @staticmethod
    def _str_iter_summary(j, index, coord, rsquare, stdev_res, diff):
        # return formatted list of figure of merits at a given iteration

        string = "{:4}  {:5}  {:8.1f} {:10.4f} {:10.4f} ".format(
            j + 1, index, coord, stdev_res, rsquare
        )
        return string

    # ----------------------------------------------------------------------------------
    # Private methods (overloading abstract classes)
    # ----------------------------------------------------------------------------------
    @tr.observe("_X")
    def _preprocess_as_X_changed(self, change):
        X = change.new

        # add some validation
        if len(X.shape) != 2:
            raise ValueError("For now, SIMPLISMA only handles 2D Datasets")

        if np.min(X) < 0:
            warn("SIMPLISMA does not handle easily negative values.")
            # TODO: check whether negative values should be set to zero or not.

        self._X_preprocessed = X.data
        # also store the name for future display
        self._Xname = X.name

    def _fit(self, X, Y=None):
        # remember most of the treatments is done in the abstract method
        # X is _X_preprocessed, so just a np.ndarray
        # Y is ignored

        interactive = self.interactive
        tol = self.tol
        noise = self.noise
        n_components = self.max_components
        M, N = X.shape
        xdata = np.arange(N)

        if interactive:
            n_components = 100

        # ------------------------------------------------------------------------------
        # Core
        # ------------------------------------------------------------------------------
        if not interactive:
            info_("*** Automatic SIMPL(I)SMA analysis ***")
        else:
            info_("*** Interactive SIMPLISMA analysis ***")

        info_(f"     dataset: {self._Xname}")
        info_(f"       noise: {noise:2} %")
        if not interactive:
            info_(f"         tol: {tol:2} %")
            info_(f"n_components: {n_components:2}")
        info_("\n")
        info_("#iter index_pc  coord_pc   Std(res)   R^2    ")
        info_("---------------------------------------------")

        # Containers for returned objects and intermediate data
        # ---------------------------------------------------
        # purity 'spectra' (generally spectra if X is passed,
        # but could also be concentrations if X.T is passed)
        Pt = np.zeros((n_components, N))
        # Pt.name = "Purity spectra"
        # Pt.set_coordset(y=Pt.y, x=X.x)
        # Pt.y.title = "# pure compound"

        # weight matrix
        w = np.zeros((n_components, N))

        # Stdev spectrum
        s = np.zeros((n_components, N))

        # maximum purity indexes and coordinates
        maxPIndex = [0] * n_components
        maxPCoordinate = [0] * n_components

        # Concentration matrix
        C = np.zeros((M, n_components))

        # Pure component spectral profiles
        St = np.zeros((n_components, N))

        # Compute Statistics
        # ------------------
        sigma = np.std(X, axis=0)
        mu = np.mean(X, axis=0)
        alpha = (noise / 100) * np.max(mu)
        lamda = np.sqrt(mu**2 + sigma**2)
        p = sigma / (mu + alpha)

        # scale dataset
        Xscaled = X / np.sqrt(mu**2 + (sigma + alpha) ** 2)

        # COO dispersion matrix
        COO = (1 / M) * np.dot(Xscaled.T, Xscaled)

        # Determine the purest variables
        j = 0
        finished = False
        while not finished:
            # compute first purest variable and weights
            if j == 0:
                w[j, :] = lamda**2 / (mu**2 + (sigma + alpha) ** 2)
                s[j, :] = sigma * w[j, :]
                Pt[j, :] = p * w[j, :]

                # get index and coordinate of pure variable
                maxPIndex[j] = np.argmax(Pt[j, :])
                maxPCoordinate[j] = xdata[maxPIndex[j]]

                # compute figures of merit
                rsquare0, stdev_res0 = self._figures_of_merit(X, maxPIndex, C, St, j)

                # add summary to log
                llog = self._str_iter_summary(
                    j, maxPIndex[j], maxPCoordinate[j], rsquare0, stdev_res0, ""
                )
                info_(llog)

                if interactive:
                    print(llog)

                    # should plot purity and stdev, does not work for the moment
                    # TODO: fix the code below
                    # fig1, (ax1, ax2) = plt.subplots(2,1)
                    # Pt[j, :].plot(ax=ax1)
                    # ax1.set_title('Purity spectrum #{}'.format(j+1))
                    # ax1.axvline(maxPCoordinate[j], color='r')
                    # s[j, :].plot(ax=ax2)
                    # ax2.set_title('standard deviation spectrum #{}'.format(j+1))
                    # ax2.axvline(maxPCoordinate[j], color='r')
                    # plt.show()

                    ans = ""
                    while ans.lower() not in ["a", "c"]:
                        ans = input("   |--> (a) Accept, (c) Change: ")

                    while ans.lower() != "a":
                        new = input(
                            "   |--> enter the new index (int) or variable value (float): "
                        )
                        try:
                            new = int(new)
                            maxPIndex[j] = new
                            maxPCoordinate[j] = xdata[maxPIndex[j]]
                        except ValueError:
                            try:
                                new = float(new)
                                maxPIndex[j] = np.argmin(abs(xdata - new))
                                maxPCoordinate[j] = xdata[maxPIndex[j]]
                            except ValueError:
                                print(
                                    "Incorrect answer. Please enter a valid index or value"
                                )

                        rsquare0, stdev_res0 = self._figures_of_merit(
                            X, maxPIndex, C, St, j
                        )

                        llog = self._str_iter_summary(
                            j, maxPIndex[j], maxPCoordinate[j], rsquare0, stdev_res0, ""
                        )
                        info_("   |--> changed pure variable #1")
                        info_(llog)

                        ans = input("   |--> (a) Accept, (c) Change: ")
                    # and was [a]ccept
                    j += 1
                if not interactive:
                    j += 1

                prev_stdev_res = stdev_res0

            else:
                # compute jth purest variable
                for i in range(X.shape[-1]):
                    Mji = np.zeros((j + 1, j + 1))
                    idx = [i] + maxPIndex[0:j]
                    for line in range(j + 1):
                        for col in range(j + 1):
                            Mji[line, col] = COO[idx[line], idx[col]]
                    w[j, i] = np.linalg.det(Mji)
                Pt[j:] = p * w[j, :]
                s[j, :] = sigma * w[j, :]

                # get index and coordinate of jth pure variable
                maxPIndex[j] = np.argmax(Pt[j, :])
                maxPCoordinate[j] = xdata[maxPIndex[j]]

                # compute figures of merit
                rsquarej, stdev_resj = self._figures_of_merit(X, maxPIndex, C, St, j)
                diff = 100 * (stdev_resj - prev_stdev_res) / prev_stdev_res
                prev_stdev_res = stdev_resj

                # add summary to log
                llog = self._str_iter_summary(
                    j, maxPIndex[j], maxPCoordinate[j], rsquarej, stdev_resj, diff
                )
                info_(llog)

                if interactive:
                    # TODO: I suggest to use jupyter widgets for the interactivity!
                    # should plot purity and stdev, does not work for the moment
                    # TODO: fix the code below
                    # ax1.clear()
                    # ax1.set_title('Purity spectrum #{}'.format(j+1))
                    # Pt[j, :].plot(ax=ax1)
                    # for coord in maxPCoordinate[:-1]:
                    #     ax1.axvline(coord, color='g')
                    # ax1.axvline(maxPCoordinate[j], color='r')
                    # ax2.clear()
                    # ax2.set_title('standard deviation spectrum #{}'.format(j+1))
                    # s[j, :].plot(ax=ax2)
                    # for coord in maxPCoordinate[:-1]:
                    #     ax2.axvline(coord, color='g')
                    # ax2.axvline(maxPCoordinate[j], color='r')
                    # plt.show()

                    ans = ""
                    while ans.lower() not in ["a", "c", "r", "f"]:
                        ans = input(
                            "   |--> (a) Accept and continue, (c) Change, (r) Reject, "
                            "(f) Accept and finish: "
                        )

                    while ans.lower() == "c":
                        new = input(
                            "   |--> enter the new index (int) or variable value (float): "
                        )
                        try:
                            new = int(new)
                            maxPIndex[j] = new
                            maxPCoordinate[j] = xdata[maxPIndex[j]]
                        except ValueError:
                            try:
                                new = float(new)
                                maxPIndex[j] = np.argmin(abs(xdata - new))
                                maxPCoordinate[j] = xdata[maxPIndex[j]]
                            except ValueError:
                                print(
                                    "   |--> Incorrect answer. Please enter a valid index or value"
                                )

                        rsquarej, stdev_resj = self._figures_of_merit(
                            X, maxPIndex, C, St, j
                        )
                        diff = 100 * (stdev_resj - prev_stdev_res) / prev_stdev_res
                        prev_stdev_res + stdev_resj

                        info_(f"   |--> changed pure variable #{j + 1}")
                        llog = self._str_iter_summary(
                            j,
                            maxPIndex[j],
                            maxPCoordinate[j],
                            rsquarej,
                            stdev_resj,
                            "diff",
                        )
                        info_(llog)

                        info_(
                            f"purest variable #{j + 1} set at index = {maxPIndex[j]} ; x = {maxPCoordinate[j]}"
                        )
                        ans = input(
                            "   |--> (a) Accept and continue, (c) Change, (r) Reject, (f) Accept and stop: "
                        )

                    if ans.lower() == "r":
                        maxPCoordinate[j] = 0
                        maxPIndex[j] = 0
                        info_(f"   |--> rejected pure variable #{j + 1}\n")
                        j = j - 1

                    elif ans.lower() == "a":
                        j = j + 1

                    elif ans.lower() == "f":
                        finished = True
                        j = j + 1
                        info_("**** Interrupted by user at compound # {j}")
                        info_("**** End of SIMPL(I)SMA analysis.")
                        Pt = Pt[0:j, :]
                        St = St[0:j, :]
                        s = s[0:j, :]
                        C = C[:, 0:j]

                # not interactive
                else:
                    j = j + 1
                    if (1 - rsquarej) < tol / 100:
                        info_(f"**** Unexplained variance lower than 'tol' ({tol} %)")
                        info_("**** End of SIMPL(I)SMA analysis.")
                        Pt = Pt[0:j, :]
                        St = St[0:j, :]
                        s = s[0:j, :]
                        C = C[:, 0:j]
                        finished = True

            if j == n_components:
                if not interactive:
                    info_(
                        f"**** Reached maximum number of pure compounds 'n_components' "
                        f"({n_components})"
                    )
                    info_("**** End of SIMPL(I)SMA analysis.")
                    finished = True

        # found components
        self._n_components = Pt.shape[0]

        # results
        _outfit = (C, St, Pt, s)
        return _outfit

    def _transform(self, X=None):
        # X is ignored for SIMPLISMA
        return self._outfit[0]

    def _inverse_transform(self, X_transform=None):
        # X_transform is ignored for MCRALS
        return np.dot(self._transform(), self._components)

    def _get_components(self):
        return self._components

    # ----------------------------------------------------------------------------------
    # Public methods and properties
    # ----------------------------------------------------------------------------------
    _docstring.keep_params("analysis_fit.parameters", "X")

    @_docstring.dedent
    def fit(self, X):
        """
        Fit the SIMPLISMA model on X.

        Parameters
        ----------
        %(analysis_fit.parameters.X)s

        Returns
        -------
        %(analysis_fit.returns)s

        See Also
        --------
        %(analysis_fit.see_also)s
        """
        return super().fit(X, Y=None)

    @property
    def C(self):
        """
        Intensities ('concentrations') of pure compounds in spectra.
        """
        C = self.transform()
        C.name = "Relative Concentrations"
        C.x.title = "# pure compound"
        C.description = "Concentration/contribution matrix from SIMPLISMA:"  # + logs
        return C

    @property
    def St(self):
        """
        Spectra of pure compounds.
        """
        St = self.components
        St.name = "Pure compound spectra"
        St.description = "Pure compound spectra matrix from SIMPLISMA:"  # + logs
        return St

    @property
    def Pt(self):
        """
        Purity spectra.
        """
        Pt = self.St.copy()  # get a container
        Pt.data = self._outfit[2]
        Pt.name = "Purity spectra"
        Pt.y.title = "# pure compound"
        Pt.description = "Purity spectra from SIMPLISMA:"  # + logs
        return Pt

    @property
    def s(self):
        """
        Standard deviation spectra.
        """
        s = self.St.copy()  # get a container
        s.data = self._outfit[3]
        s.name = "Standard deviation spectra"
        s.description = "Standard deviation spectra matrix from SIMPLISMA:"  # + logs
        return s
