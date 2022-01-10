# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory.
# ======================================================================================================================
"""
This module implement the PCA (Principal Component Analysis) class.
"""

__all__ = ["PCA"]

__dataset_methods__ = []

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from scipy import special
from traitlets import HasTraits, Instance

from spectrochempy.core import info_
from spectrochempy.core.analysis.svd import SVD
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.dataset.npy import dot
from spectrochempy.utils import NRed, NBlue


# ======================================================================================================================
# class PCA
# ======================================================================================================================


class PCA(HasTraits):
    """
    Principal Component Analysis.

    This class performs a Principal Component Analysis of a |NDDataset|,
    *i.e.*, a linear dimensionality reduction using Singular Value
    Decomposition (`SVD`)
    of the data to perform its projection to a lower dimensional space.

    The reduction of a dataset :math:`X` with shape (`M`,`N`) is achieved
    using the decomposition : :math:`X = S.L^T`, where
    :math:`S` is the score's matrix with shape (`M`, `n_pc`) and :math:`L^T` is
    the transposed loading's matrix with shape (`n_pc`, `N`).

    If the dataset `X` contains masked values, these values are silently
    ignored in the calculation.
    """

    _LT = Instance(NDDataset)
    _S = Instance(NDDataset)
    _X = Instance(NDDataset)

    _ev = Instance(NDDataset)
    """|NDDataset| - Explained variances (The eigenvalues of the covariance matrix)."""

    _ev_ratio = Instance(NDDataset)
    """|NDDataset| - Explained variance per singular values."""

    _ev_cum = Instance(NDDataset)
    """|NDDataset| - Cumulative Explained Variances."""

    # ..........................................................................
    def __init__(self, dataset, centered=True, standardized=False, scaled=False):
        """
        Parameters
        ----------
        dataset : |NDDataset| object
            The input dataset has shape (M, N). M is the number of
            observations (for examples a series of IR spectra) while N
            is the number of features (for example the wavenumbers measured
            in each IR spectrum).
        centered : bool, optional, default:True
            If True the data are centered around the mean values: :math:`X' = X - mean(X)`.
        standardized : bool, optional, default:False
            If True the data are scaled to unit standard deviation: :math:`X' = X / \\sigma`.
        scaled : bool, optional, default:False
            If True the data are scaled in the interval [0-1]: :math:`X' = (X - min(X)) / (max(X)-min(X))`
        """
        super().__init__()

        self.prefs = dataset.preferences

        self._X = X = dataset

        Xsc = X.copy()

        # mean center the dataset
        # -----------------------
        self._centered = centered
        if centered:
            self._center = center = X.mean(dim=0)
            Xsc = X - center
            Xsc.title = f"centered {X.title}"

        # Standardization
        # ---------------
        self._standardized = standardized
        if standardized:
            self._std = Xsc.std(dim=0)
            Xsc /= self._std
            Xsc.title = f"standardized {Xsc.title}"

        # Scaling
        # -------
        self._scaled = scaled
        if scaled:
            self._min = Xsc.min(dim=0)
            self._ampl = Xsc.ptp(dim=0)
            Xsc -= self._min
            Xsc /= self._ampl
            Xsc.title = "scaled %s" % Xsc.title

        self._Xscaled = Xsc

        # perform SVD
        # -----------
        svd = SVD(Xsc)
        sigma = svd.s.diag()
        U = svd.U
        VT = svd.VT

        # select n_pc loadings & compute scores
        # --------------------------------------------------------------------

        # loadings

        LT = VT
        LT.title = "loadings (L^T) of " + X.name
        LT.history = "Created by PCA"

        # scores

        S = dot(U, sigma)
        S.title = "scores (S) of " + X.name
        S.set_coordset(
            y=X.y,
            x=Coord(
                None,
                labels=["#%d" % (i + 1) for i in range(svd.s.size)],
                title="principal component",
            ),
        )

        S.description = "scores (S) of " + X.name
        S.history = "Created by PCA"

        self._LT = LT
        self._S = S

        # other attributes
        # ----------------

        self._sv = svd.sv
        self._sv.x.title = "PC #"

        self._ev = svd.ev
        self._ev.x.title = "PC #"

        self._ev_ratio = svd.ev_ratio
        self._ev_ratio.x.title = "PC #"

        self._ev_cum = svd.ev_cum
        self._ev_cum.x.title = "PC #"

        return

    # ------------------------------------------------------------------------
    # Special methods
    # ------------------------------------------------------------------------

    def __str__(self, n_pc=5):

        s = "\nPC\t\tEigenvalue\t\t%variance\t" "%cumulative\n"
        s += "   \t\tof cov(X)\t\t per PC\t" "     variance\n"

        n_pc = min(n_pc, len(self.ev.data))
        for i in range(n_pc):
            tup = (
                i + 1,
                np.sqrt(self.ev.data[i]),
                self.ev_ratio.data[i],
                self.ev_cum.data[i],
            )
            s += "#{}  \t{:8.3e}\t\t {:6.3f}\t      {:6.3f}\n".format(*tup)

        return s

    # ------------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------------

    def _get_n_pc(self, n_pc=None):

        max_n_pc = self.ev.size
        if n_pc is None:
            n_pc = max_n_pc
            return n_pc
        elif isinstance(n_pc, int):
            n_pc = min(n_pc, max_n_pc)
            return n_pc
        elif n_pc == "auto":
            M, N = self.X.shape
            if M >= N:
                n_pc = self._infer_pc_()
                return n_pc
            else:
                info_(
                    "Cannot use `auto` if n_observations < "
                    "n_features. Try with threshold 0.9999"
                )
                n_pc = 0.9999

        if 0 < n_pc < 1.0:
            # number of PC for which the cumulated explained variance is
            # less than a given ratio
            n_pc = np.searchsorted(self.ev_cum.data / 100.0, n_pc) + 1
            return n_pc
        else:
            raise ValueError("could not get a valid number of components")

    def _assess_dimension_(self, rank):
        """Compute the likelihood of a rank ``rank`` dataset
        The dataset is assumed to be embedded in gaussian noise having
        spectrum ``spectrum`` (here, the explained variances `ev` ).

        Parameters
        ----------
        rank : int
            Tested rank value.

        Returns
        -------
        float
            The log-likelihood.

        Notes
        -----
        This implements the method of Thomas P. Minka :
        Automatic Choice of Dimensionality for PCA. NIPS 2000 : 598-604.
        Copied and modified from scikit-learn.decomposition.pca (BSD-3 license)
        """
        spectrum = self._ev.data
        M, N = self._X.shape

        if rank > len(spectrum):
            raise ValueError("The tested rank cannot exceed the rank of the" " dataset")

        pu = -rank * np.log(2.0)
        for i in range(rank):
            pu += special.gammaln((N - i) / 2.0) - np.log(np.pi) * (N - i) / 2.0

        pl = np.sum(np.log(spectrum[:rank]))
        pl = -pl * M / 2.0

        if rank == N:
            pv = 0
            v = 1
        else:
            v = np.sum(spectrum[rank:]) / (N - rank)
            pv = -np.log(v) * M * (N - rank) / 2.0

        m = N * rank - rank * (rank + 1.0) / 2.0
        pp = np.log(2.0 * np.pi) * (m + rank + 1.0) / 2.0

        pa = 0.0
        spectrum_ = spectrum.copy()
        spectrum_[rank:N] = v
        for i in range(rank):
            for j in range(i + 1, len(spectrum)):
                pa += np.log(
                    (spectrum[i] - spectrum[j])
                    * (1.0 / spectrum_[j] - 1.0 / spectrum_[i])
                ) + np.log(M)

        ll = pu + pl + pv + pp - pa / 2.0 - rank * np.log(M) / 2.0

        return ll

    def _infer_pc_(self):
        """Infers the number of principal components.

        Notes
        -----
        Copied and modified from _infer_dimensions in
        scikit-learn.decomposition.pca (BSD-3 license).
        """
        n_ev = self._ev.size
        ll = np.empty(n_ev)
        for rank in range(n_ev):
            ll[rank] = self._assess_dimension_(rank)
        return ll.argmax()

    # ------------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------------

    def reduce(self, n_pc=None):
        """
        Apply a dimensionality reduction to the X dataset of shape [M, N].

        Loadings `L` with shape [``n_pc``, `N`] and scores `S`
        with shape [`M`, `n_pc`] are obtained using the following
        decomposition : :math:`X = S.L^T`.

        Parameters
        ----------
        n_pc : int, optional
            The number of principal components to compute. If not set all
            components are returned, except if n_pc is set to ``auto`` for
            an automatic determination of the number of components.

        Returns
        -------
        S, LT : |NDDataset| objects.
            n_pc loadings and their corresponding scores for each observations.
        """

        # get n_pc (automatic or determined by the n_pc arguments)
        n_pc = self._get_n_pc(n_pc)

        # scores (S) and loading (L^T) matrices
        # ------------------------------------

        S = self._S[:, :n_pc]
        LT = self._LT[:n_pc]

        return S, LT

    def reconstruct(self, n_pc=None):
        """
        Transform data back to the original space using the given number of
        PC's.

        The following matrice operation is performed : :math:`X' = S'.L'^T`
        where S'=S[:, n_pc] and L=L[:, n_pc].

        Parameters
        ----------
        n_pc : int, optional
            The number of PC to use for the reconstruction.

        Returns
        -------
        X_reconstructed : |NDDataset|
            The reconstructed dataset based on n_pc principal components.
        """

        # get n_pc (automatic or determined by the n_pc arguments)
        n_pc = self._get_n_pc(n_pc)

        # reconstruct from scores and loadings using n_pc components
        S = self._S[:, :n_pc]
        LT = self._LT[:n_pc]

        X = dot(S, LT)

        # try to reconstruct something close to the original scaled, standardized or centered data
        if self._scaled:
            X *= self._ampl
            X += self._min
        if self._standardized:
            X *= self._std
        if self._centered:
            X += self._center

        X.history = f"PCA reconstructed Dataset with {n_pc} principal components"
        X.title = self._X.title
        return X

    def printev(self, n_pc=None):
        """
        Prints figures of merit : eigenvalues and explained variance for the first n_pc PS's.

        Parameters
        ----------
        n_pc : int, optional
            The number of components to print.
        """
        # get n_pc (automatic or determined by the n_pc arguments)
        n_pc = self._get_n_pc(n_pc)

        print((self.__str__(n_pc)))

    def screeplot(self, n_pc=None, **kwargs):
        """
        Scree plot of explained variance + cumulative variance by PCA.

        Parameters
        ----------
        n_pc : int
            Number of components to plot.
        """
        # get n_pc (automatic or determined by the n_pc arguments) - min = 3
        n_pc = max(self._get_n_pc(n_pc), 3)

        color1, color2 = kwargs.get("colors", [NBlue, NRed])
        # pen = kwargs.get('pen', True)
        ylim1, ylim2 = kwargs.get("ylims", [(0, 100), "auto"])

        if ylim2 == "auto":
            y1 = np.around(self._ev_ratio.data[0] * 0.95, -1)
            y2 = 101.0
            ylim2 = (y1, y2)

        ax1 = self._ev_ratio[:n_pc].plot_bar(
            ylim=ylim1, color=color1, title="Scree plot"
        )
        ax2 = self._ev_cum[:n_pc].plot_scatter(
            ylim=ylim2, color=color2, pen=True, markersize=7.0, twinx=ax1
        )
        ax1.set_title("Scree plot")
        return ax1, ax2

    def scoreplot(self, *pcs, colormap="viridis", color_mapping="index", **kwargs):
        """
        2D or 3D scoreplot of samples.

        Parameters
        ----------
        *pcs : a series of int argument or a list/tuple
            Must contain 2 or 3 elements.
        colormap : str
            A matplotlib colormap.
        color_mapping : 'index' or 'labels'
            If 'index', then the colors of each n_scores is mapped sequentially
            on the colormap. If labels, the labels of the n_observation are
            used for color mapping.
        """

        if isinstance(pcs[0], (list, tuple, set)):
            pcs = pcs[0]

        # transform to internal index of component's index (1->0 etc...)
        pcs = np.array(pcs) - 1

        # colors
        if color_mapping == "index":

            if np.any(self._S.y.data):
                colors = self._S.y.data
            else:
                colors = np.array(range(self._S.shape[0]))

        elif color_mapping == "labels":

            labels = list(set(self._S.y.labels))
            colors = [labels.index(lab) for lab in self._S.y.labels]

        if len(pcs) == 2:
            # bidimentional score plot

            fig = plt.figure(**kwargs)
            ax = fig.add_subplot(111)
            ax.set_title("Score plot")

            ax.set_xlabel(
                "PC# {} ({:.3f}%)".format(pcs[0] + 1, self._ev_ratio.data[pcs[0]])
            )
            ax.set_ylabel(
                "PC# {} ({:.3f}%)".format(pcs[1] + 1, self._ev_ratio.data[pcs[1]])
            )
            axsc = ax.scatter(
                self._S.masked_data[:, pcs[0]],
                self._S.masked_data[:, pcs[1]],
                s=30,
                c=colors,
                cmap=colormap,
            )

            number_x_labels = self.prefs.number_of_x_labels  # get from config
            number_y_labels = self.prefs.number_of_y_labels
            # the next two line are to avoid multipliers in axis scale
            y_formatter = ScalarFormatter(useOffset=False)
            ax.yaxis.set_major_formatter(y_formatter)
            ax.xaxis.set_major_locator(MaxNLocator(number_x_labels))
            ax.yaxis.set_major_locator(MaxNLocator(number_y_labels))
            ax.xaxis.set_ticks_position("bottom")
            ax.yaxis.set_ticks_position("left")

        if len(pcs) == 3:
            # tridimensional score plot
            plt.figure(**kwargs)
            ax = plt.axes(projection="3d")
            ax.set_title("Score plot")
            ax.set_xlabel(
                "PC# {} ({:.3f}%)".format(pcs[0] + 1, self._ev_ratio.data[pcs[0]])
            )
            ax.set_ylabel(
                "PC# {} ({:.3f}%)".format(pcs[1] + 1, self._ev_ratio.data[pcs[1]])
            )
            ax.set_zlabel(
                "PC# {} ({:.3f}%)".format(pcs[2] + 1, self._ev_ratio.data[pcs[2]])
            )
            axsc = ax.scatter(
                self._S.masked_data[:, pcs[0]],
                self._S.masked_data[:, pcs[1]],
                self._S.masked_data[:, pcs[2]],
                zdir="z",
                s=30,
                c=colors,
                cmap=colormap,
                depthshade=True,
            )

        if color_mapping == "labels":
            import matplotlib.patches as mpatches

            leg = []
            for lab in labels:
                i = labels.index(lab)
                c = axsc.get_cmap().colors[int(255 / (len(labels) - 1) * i)]
                leg.append(mpatches.Patch(color=c, label=lab))

            ax.legend(handles=leg, loc="best")

        return ax

    @property
    def LT(self):
        """
        LT.
        """
        return self._LT

    @property
    def S(self):
        """
        S.
        """
        return self._S

    @property
    def X(self):
        """
        X.
        """
        return self._X

    @property
    def ev(self):
        """
        Explained variances (|NDDataset|).

        (The eigenvalues of the covariance matrix).
        """
        return self._ev

    @property
    def ev_ratio(self):
        """
        Explained variance per singular values (|NDDataset|).
        """
        return self._ev_ratio

    @property
    def ev_cum(self):
        """
        Cumulative Explained Variances (|NDDataset|).
        """
        return self._ev_cum


# ============================================================================
if __name__ == "__main__":
    pass
