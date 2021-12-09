# -*- coding: utf-8 -*-

#
# =============================================================================
# Copyright (©) 2015-2022 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================
"""
This module implement the SIMPLISMA class.
"""

__all__ = ["SIMPLISMA"]

__dataset_methods__ = []

# ----------------------------------------------------------------------------
# imports
# ----------------------------------------------------------------------------
import numpy as np

import warnings
from traitlets import HasTraits, Instance, Unicode

from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.dataset.npy import dot
from spectrochempy.core import info_, set_loglevel, INFO


# ============================================================================
# class SIMPLISMA
# ============================================================================


class SIMPLISMA(HasTraits):
    """
    SIMPLe to use Interactive Self-modeling Mixture Analysis.

    This class performs a SIMPLISMA analysis of a 2D |NDDataset|. The algorithm is adapted from Windig's paper,
    Chemometrics and Intelligent Laboratory Systems, 36, 1997, 3-16.

    TODO : adapt to 3DDataset ?
    """

    _St = Instance(NDDataset)
    _C = Instance(NDDataset)
    _X = Instance(NDDataset)
    _Pt = Instance(NDDataset)
    _s = Instance(NDDataset)
    _logs = Unicode

    def __init__(self, dataset, **kwargs):
        """
        Parameters
        ----------
        dataset : |NDDataset|
            A 2D dataset containing the data matrix (spectra in rows).
        interactive : bool, optional, default=False
            If True, the determination of purest variables is carried out interactively
        n_pc : int, optional, default=2 in non-interactive mode; 100 in interactive mode
            The maximum number of pure compounds. Used only for non interactive analysis
            (the default in interative mode (100) will never be reached in practice).
        tol : float, optional, default=0.1
            The convergence criterion on the percent of unexplained variance.
        noise : float or int, optional, default=5
            A correction factor (%) for low intensity variables (0 - no offset, 15 - large offset).
        verbose : bool, optional, default=True
            If True some information is given during the analysis.
        """

        super().__init__()

        # ------------------------------------------------------------------------
        # Utility functions
        # ------------------------------------------------------------------------
        def figures_of_merit(X, maxPIndex, C, St, j):
            # return %explained variance and stdev of residuals when the jth compound is added
            C[:, j] = X[:, maxPIndex[j]]
            St[0 : j + 1, :] = np.linalg.lstsq(
                C.data[:, 0 : j + 1], X.data, rcond=None
            )[0]
            Xhat = dot(C[:, 0 : j + 1], St[0 : j + 1, :])
            res = Xhat - X
            stdev_res = np.std(res)
            rsquare = 1 - np.linalg.norm(res) ** 2 / np.linalg.norm(X) ** 2
            return rsquare, stdev_res

        def str_iter_summary(j, index, coord, rsquare, stdev_res, diff):
            # return formatted list of figure of merits at a given iteration

            string = "{:4}  {:5}  {:8.1f} {:10.4f} {:10.4f} ".format(
                j + 1, index, coord, stdev_res, rsquare
            )
            return string

        def get_x_data(X):
            if X.x is not None and not X.x.is_empty:  # TODO what about labels?
                return X.x.data
            else:
                return np.arange(X.shape[-1])

        # ------------------------------------------------------------------------
        # Check data
        # ------------------------------------------------------------------------

        X = dataset

        if len(X.shape) != 2:
            raise ValueError("For now, SIMPLISMA only handles 2D Datasets")

        if np.min(X.data) < 0:
            warnings.warn("SIMPLISMA does not handle easily negative values.")
            # TODO: check whether negative values should be set to zero or not.

        verbose = kwargs.get("verbose", True)
        if verbose:
            set_loglevel(INFO)

        interactive = kwargs.get("interactive", False)
        tol = kwargs.get("tol", 0.1)
        noise = kwargs.get("noise", 3)
        n_pc = kwargs.get("n_pc", 2)
        if n_pc < 2 or not isinstance(n_pc, int):
            raise ValueError(
                "Oh you did not just... 'MA' in simplisMA stands for Mixture Analysis. "
                "The number of pure compounds should be an integer larger than 2"
            )
        if interactive:
            n_pc = 100

        # ------------------------------------------------------------------------
        # Core
        # ------------------------------------------------------------------------

        if not interactive:
            logs = "*** Automatic SIMPL(I)SMA analysis *** \n"
        else:
            logs = "*** Interative SIMPLISMA analysis *** \n"
        logs += "dataset: {}\n".format(X.name)
        logs += "  noise: {:2} %\n".format(noise)
        if not interactive:
            logs += "    tol: {:2} %\n".format(tol)
            logs += "   n_pc: {:2}\n".format(n_pc)
        logs += "\n"
        logs += "#iter index_pc  coord_pc   Std(res)   R^2   \n"
        logs += "---------------------------------------------"
        info_(logs)
        logs += "\n"

        # Containers for returned objects and intermediate data
        # ---------------------------------------------------
        # purity 'spectra' (generally spectra if X is passed,
        # but could also be concentrations if X.T is passed)
        Pt = NDDataset.zeros((n_pc, X.shape[-1]))
        Pt.name = "Purity spectra"
        Pt.set_coordset(y=Pt.y, x=X.x)
        Pt.y.title = "# pure compound"

        # weight matrix
        w = NDDataset.zeros((n_pc, X.shape[-1]))
        w.set_coordset(y=Pt.y, x=X.x)

        # Stdev spectrum
        s = NDDataset.zeros((n_pc, X.shape[-1]))
        s.name = "Standard deviation spectra"
        s.set_coordset(y=Pt.y, x=X.x)

        # maximum purity indexes and coordinates
        maxPIndex = [0] * n_pc
        maxPCoordinate = [0] * n_pc

        # Concentration matrix
        C = NDDataset.zeros((X.shape[-2], n_pc))
        C.name = "Relative Concentrations"
        C.set_coordset(y=X.y, x=C.x)
        C.x.title = "# pure compound"

        # Pure component spectral profiles
        St = NDDataset.zeros((n_pc, X.shape[-1]))
        St.name = "Pure compound spectra"
        St.set_coordset(y=Pt.y, x=X.x)

        # Compute Statistics
        # ------------------
        sigma = np.std(X.data, axis=0)
        mu = np.mean(X.data, axis=0)
        alpha = (noise / 100) * np.max(mu.data)
        lamda = np.sqrt(mu ** 2 + sigma ** 2)
        p = sigma / (mu + alpha)

        # scale dataset
        Xscaled = X.data / np.sqrt(mu ** 2 + (sigma + alpha) ** 2)

        # COO dispersion matrix
        COO = (1 / X.shape[-2]) * np.dot(Xscaled.T, Xscaled)

        # Determine the purest variables
        j = 0
        finished = False
        while not finished:
            # compute first purest variable and weights
            if j == 0:
                w[j, :] = lamda ** 2 / (mu ** 2 + (sigma + alpha) ** 2)
                s[j, :] = sigma * w[j, :]
                Pt[j, :] = p * w[j, :]

                # get index and coordinate of pure variable
                maxPIndex[j] = np.argmax(Pt[j, :].data)
                maxPCoordinate[j] = get_x_data(X)[maxPIndex[j]]

                # compute figures of merit
                rsquare0, stdev_res0 = figures_of_merit(X, maxPIndex, C, St, j)

                # add summary to log
                llog = str_iter_summary(
                    j, maxPIndex[j], maxPCoordinate[j], rsquare0, stdev_res0, ""
                )
                logs += llog + "\n"

                if verbose or interactive:
                    print(llog)

                if interactive:
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
                            maxPCoordinate[j] = get_x_data(X)[maxPIndex[j]]
                        except ValueError:
                            try:
                                new = float(new)
                                maxPIndex[j] = np.argmin(abs(get_x_data(X) - new))
                                maxPCoordinate[j] = get_x_data(X)[maxPIndex[j]]
                            except ValueError:
                                print(
                                    "Incorrect answer. Please enter a valid index or value"
                                )

                        rsquare0, stdev_res0 = figures_of_merit(X, maxPIndex, C, St, j)

                        llog = str_iter_summary(
                            j, maxPIndex[j], maxPCoordinate[j], rsquare0, stdev_res0, ""
                        )
                        logs += "   |--> changed pure variable #1"
                        logs += llog + "\n"
                        info_(llog)

                        ans = input("   |--> (a) Accept, (c) Change: ")
                    # ans was [a]ccept
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
                maxPIndex[j] = np.argmax(Pt[j, :].data)
                maxPCoordinate[j] = get_x_data(X)[maxPIndex[j]]

                # compute figures of merit
                rsquarej, stdev_resj = figures_of_merit(X, maxPIndex, C, St, j)
                diff = 100 * (stdev_resj - prev_stdev_res) / prev_stdev_res
                prev_stdev_res = stdev_resj

                # add summary to log
                llog = str_iter_summary(
                    j, maxPIndex[j], maxPCoordinate[j], rsquarej, stdev_resj, diff
                )
                logs += llog + "\n"

                if verbose or interactive:
                    info_(llog)

                if (
                    interactive
                ):  # TODO: I suggest to use jupyter widgets for the interactivity!
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
                            "   |--> (a) Accept and continue, (c) Change, (r) Reject, (f) Accept and finish: "
                        )

                    while ans.lower() == "c":
                        new = input(
                            "   |--> enter the new index (int) or variable value (float): "
                        )
                        try:
                            new = int(new)
                            maxPIndex[j] = new
                            maxPCoordinate[j] = get_x_data(X)[maxPIndex[j]]
                        except ValueError:
                            try:
                                new = float(new)
                                maxPIndex[j] = np.argmin(abs(get_x_data(X) - new))
                                maxPCoordinate[j] = get_x_data(X)[maxPIndex[j]]
                            except ValueError:
                                print(
                                    "   |--> Incorrect answer. Please enter a valid index or value"
                                )

                        rsquarej, stdev_resj = figures_of_merit(X, maxPIndex, C, St, j)
                        diff = 100 * (stdev_resj - prev_stdev_res) / prev_stdev_res
                        prev_stdev_res + stdev_resj

                        logs += f"   |--> changed pure variable #{j + 1}\n"
                        llog = str_iter_summary(
                            j,
                            maxPIndex[j],
                            maxPCoordinate[j],
                            rsquarej,
                            stdev_resj,
                            "diff",
                        )
                        logs += llog + "\n"
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
                        logs += f"   |--> rejected pure variable #{j + 1}\n"
                        j = j - 1

                    elif ans.lower() == "a":
                        j = j + 1

                    elif ans.lower() == "f":
                        finished = True
                        j = j + 1
                        llog = f"\n**** Interrupted by user at compound # {j} \n**** End of SIMPL(I)SMA analysis."
                        logs += llog + "\n"
                        Pt = Pt[0:j, :]
                        St = St[0:j, :]
                        s = s[0:j, :]
                        C = C[:, 0:j]
                # not interactive
                else:
                    j = j + 1
                    if (1 - rsquarej) < tol / 100:
                        llog = (
                            f"\n**** Unexplained variance lower than 'tol' ({tol}%) \n"
                            "**** End of SIMPL(I)SMA analysis."
                        )
                        logs += llog + "\n"
                        Pt = Pt[0:j, :]
                        St = St[0:j, :]
                        s = s[0:j, :]
                        C = C[:, 0:j]

                        info_(llog)
                        finished = True
            if j == n_pc:
                if not interactive:
                    llog = (
                        f"\n**** Reached maximum number of pure compounds 'n_pc' ({n_pc}) \n"
                        "**** End of SIMPL(I)SMA analysis."
                    )
                    logs += llog + "\n"
                    info_(llog)
                    finished = True

        Pt.description = "Purity spectra from SIMPLISMA:\n" + logs
        C.description = "Concentration/contribution matrix from SIMPLISMA:\n" + logs
        St.description = "Pure compound spectra matrix from SIMPLISMA:\n" + logs
        s.description = "Standard deviation spectra matrix from SIMPLISMA:\n" + logs

        self._logs = logs
        self._X = X
        self._Pt = Pt
        self._C = C
        self._St = St
        self._s = s

    @property
    def X(self):
        """
        The original dataset.
        """
        return self._X

    @property
    def St(self):
        """
        Spectra of pure compounds.
        """
        return self._St

    @property
    def C(self):
        """
        Intensities ('concentrations') of pure compounds in spectra.
        """
        return self._C

    @property
    def Pt(self):
        """
        Purity spectra.
        """
        return self._Pt

    @property
    def s(self):
        """
        Standard deviation spectra.
        """
        return self._s

    @property
    def logs(self):
        """
        Logs ouptut.
        """
        return self._logs

    def reconstruct(self):
        """
        Transform data back to the original space.

        The following matrix operation is performed: :math:`X'_{hat} = C'.S'^t`

        Returns
        -------
        X_hat
            The reconstructed dataset based on the SIMPLISMA Analysis.
        """

        # reconstruct from concentration and spectra profiles

        X_hat = dot(self.C, self.St)
        X_hat.description = "Dataset reconstructed by SIMPLISMA\n" + self.logs
        X_hat.title = "X_hat: " + self.X.title
        return X_hat

    def plotmerit(self, **kwargs):
        """
        Plots the input dataset, reconstructed dataset and residuals.

        Parameters
        ----------
        **kwargs : dict
            Plotting parameters.

        Returns
        -------
        ax
            subplot.
        """

        colX, colXhat, colRes = kwargs.get("colors", ["blue", "green", "red"])

        X_hat = self.reconstruct()

        res = self.X - X_hat

        ax = self.X.plot(label="$X$")
        ax.plot(X_hat.data.T, color=colXhat, label=r"$\hat{X}")
        ax.plot(res.data.T, color=colRes, label="Residual")
        ax.set_title("SIMPLISMA plot: " + self.X.name)

        return ax


# ============================================================================
if __name__ == "__main__":
    pass
