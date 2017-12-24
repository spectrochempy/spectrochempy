# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================

__all__ = ['PCA']

# ----------------------------------------------------------------------------
# third party imports
# ----------------------------------------------------------------------------
import numpy as np
from traitlets import HasTraits, Instance
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator, ScalarFormatter

from spectrochempy.dataset.nddataset import NDDataset, CoordSet
from spectrochempy.dataset.ndcoords import Coord
from spectrochempy.analysis.svd import SVD
from spectrochempy.processors.numpy import diag, dot
from spectrochempy.dataset.ndplot import NRed, NBlue

# ============================================================================
# Global preferences
# ============================================================================
from spectrochempy.application import app

# ----------------------------------------------------------------------------
# localimports
# ----------------------------------------------------------------------------

plotter_preferences = app.plotter_preferences
log = app.log
preferences = app


# ============================================================================
# class PCA
# ============================================================================

class PCA(HasTraits):
    """
    Principal Component Analysis

    This class performs a Principal Component Analysis of a NDDataset, *i.e.*,
    a linear dimensionality reduction using Singular Value Decomposition (SVD)
    of the data to project it to a lower dimensional space.

    If the dataset contains masked values, the corresponding ranges are
    ignored in the calculation.

    """

    T = Instance(NDDataset)
    Pt = Instance(NDDataset)

    ev = Instance(NDDataset)
    ev_ratio = Instance(NDDataset)
    ev_cum = Instance(NDDataset)

    def __init__(self, X,
                 centered=True,
                 standardized=False,
                 scaled = False):
        """
        Parameters
        ----------
        X : :class:`~spectrochempy.dataset.nddataset.NDDataset` object.
            The dataset has shape (``M``, ``N``)
        centered : Bool, optional, default=True
            If True the data are centered around the mean values:
            X' = X - mean(X)
        standardized : Bool, optional, default=False
            If True the data are scaled to unit standard deviation:
            X' = X / sigma
        scaled : Bool, optional, default=False
            If True the data are scaled in the interval [0-1]
            X' = (X - min(X)) / (max(X)-min(X))

        Attributes
        ----------
        ev : :class:`~spectrochempy.dataset.nddataset.NDDataset`.
            Eigenvalues of the covariance matrix
        ev_ratio : :class:`~spectrochempy.dataset.nddataset.NDDataset`.
            Explained Variance per singular values
        ev_cum : :class:`~spectrochempy.dataset.nddataset.NDDataset`.
            Cumulative Explained Variance

        """

        self._X = X

        Xsc = X.copy()

        # mean center the dataset
        # -----------------------
        self._centered = centered
        if centered:
            self._center = center = np.mean(X, axis=0)
            Xsc = X - center
            Xsc.title = "centered %s"% X.title

        # Standardization
        # ---------------
        self._standardized = standardized
        if standardized:
            self._std = np.std(Xsc, axis=0)
            Xsc /= self._std
            Xsc.title = "standardized %s" % Xsc.title

        # Scaling
        # -------
        self._scaled = scaled
        if scaled:
            self._min = np.min(Xsc, axis=0)
            self._ampl = np.ptp(Xsc, axis=0)
            Xsc -= self._min
            Xsc /= self._ampl
            Xsc.title = "scaled %s" % Xsc.title

        self._Xscaled = Xsc

        # perform SVD
        # -----------
        svd = svd = SVD(Xsc)
        sigma = diag(svd.s)
        U = svd.U
        VT = svd.VT

        # select n_pc loadings & compute scores
        # --------------------------------------------------------------------

        # loadings

        LT = VT
        LT.title = 'Loadings (L^T) of ' + X.name
        LT.history = 'created by PCA'

        # scores

        S = dot(U, sigma)
        S.title = 'scores (S) of ' + X.name
        S.coordset = CoordSet(X.y, Coord(None,
                          labels=['#%d' % (i + 1) for i in range(svd.s.size)],
                              title='principal component'))

        S.description = 'scores (S) of ' + X.name
        S.history = 'created by PCA'

        self._LT = LT
        self._S = S

        # other attributes
        # ----------------

        #: Eigenvalues of the covariance matrix
        self.ev = svd.ev
        self.ev.x.title = 'PC #'

        #: Explained Variance per singular values
        self.ev_ratio= svd.ev_ratio
        self.ev_ratio.x.title = 'PC #'

        #: Cumulative Explained Variance
        self.ev_cum = svd.ev_cum
        self.ev_cum.x.title = 'PC #'

        return

    # ------------------------------------------------------------------------
    # Special methods
    # ------------------------------------------------------------------------

    def __str__(self, n_pc=10):

        s = '\nPC\t\tEigenvalue\t\t%variance\t' \
            '%cumulative\n'
        s += '   \t\tof cov(X)\t\t per PC\t' \
             '     variance\n'
        for i in range(n_pc):
            tuple = (
            i, self.ev.data[i], self.ev_ratio.data[i], self.ev_cum.data[i])
            s += '#{}  \t{:8.3e}\t\t {:6.3f}\t      {:6.3f}\n'.format(*tuple)

        return s

    # ------------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------------

    def transform(self, n_pc=None):
        """
        Apply the dimensionality reduction to the X dataset of shape
        [n_observation, n_features].

        Loadings :math:`L` with shape [n_pc, n_features] and scores :math:`S`
        with shape [n_observation, n_pc] are obtained using the following
        decomposition: :math:`X = S.L^T`

        Parameters
        ----------
        n_pc : int, optional, default=10
            The number of principal components to compute. If not set all
            components are returned, except if n_pc is set to ``auto`` for
            an automatic determination of the number of components.

        Returns
        -------
        :math:`L^T`, :math:`S` : :class:`~spectrochempy.dataset.nddataset.NDDataset
        `objects.
            n_pc loadings and their corresponding scores for each observations.


        """

        X = self._X

        if n_pc is None:
            n_pc = self._LT.shape[0]


        # scores (S) and loading (L^T) matrices
        # ------------------------------------

        S = self._S[:, :n_pc]
        LT = self._LT[:n_pc]

        return LT, S

    #_infer_dimension_(explained_variance_, n_samples, n_features)

    def inverse_transform(self, n_pc=None):
        """
        Transform data back to the original space using the given number of
        PC's.

        The following matrice operation is performed: :math:`X' = S'.L'^T`
        where :math:`S'=S[:, n_pc]` and :math:`L'=L[:, n_pc].

        Parameters
        ----------
        n_pc : int, optional, default=10
            The number of PC to use for the reconstruction

        Return
        ------
        X_reconstructed : :class:`~spectrochempy.dataset.nddataset.NDDataset`.
            The reconstructed dataset based on n_pc principal components.

        """

        if n_pc is None:
            n_pc = self._S.shape[-1]
        else:
            n_pc = min(n_pc, self._S.shape[-1])

        S = self._S[:, :n_pc]
        LT = self._LT[:n_pc]

        X = dot(S, LT)

        # try to reconstruct something close to the original scaled,
        # standardized or centered data
        if self._scaled:
            X *= self._ampl
            X += self._min
        if self._standardized:
            X *= self._std
        if self._centered:
            X += self._center

        X.history = 'PCA reconstructed Dataset with {} principal ' \
                    'components'.format(n_pc)
        X.title = self._X.title
        return X

    def printev(self, n_pc=10):
        """prints figures of merit: eigenvalues and explained variance
        for the first n_pc PS's

        Parameters
        ----------
        n_pc : int, optional, default=10

          The number of PC to print

        """
        print((self.__str__(n_pc)))

    def screeplot(self,
                  n_pc=5,
                  **kwargs):
        """
        Scree plot of explained variance + cumulative variance by PCA

        Parameters
        ----------
        n_pc: int
            Number of components to plot

        """
        n_pc = min(n_pc, self.ev.size)
        color1, color2 = kwargs.get('colors', [NBlue, NRed])
        pen = kwargs.get('pen', True)
        ylim1, ylim2 = kwargs.get('ylims', [(0,100), 'auto'])

        if ylim2 == 'auto':
            y1 = np.around(self.ev_ratio.data[0]*.95,-1)
            y2 = 101.
            ylim2 = (y1, y2)

        ax1 = self.ev_ratio[:n_pc].plot_bar(ylim = ylim1,
                                           color = color1,
                                           title='Scree plot')
        ax2 = self.ev_cum[:n_pc].plot_scatter(ylim = ylim2,
                                             color=color2,
                                             pen=True,
                                             markersize = 7.,
                                             twinx = ax1
                                            )
        return ax1, ax2


    def scoreplot(self, *pcs, colormap='viridis', color_mapping='index' ,
                  **kwargs):
        """
        2D or 3D scoreplot of samples

        Parameters
        ----------
        *pcs: a series of int argument or a list/tuple
            Must contain 2 or 3 elements
        colormap : str
            A matplotlib colormap
        color_mapping : 'index' or 'labels'
            If 'index', then the colors of each n_scores is mapped sequentially
            on the colormap. If labels, the labels of the n_observation are
            used for color mapping.

        Examples
        --------
        >>> pca.scoreplot(1,2)

        """

        if isinstance(pcs[0], (list,tuple, set)):
            pcs = pcs[0]

        # transform to internal index of component's index (1->0 etc...)
        pcs = np.array(pcs) - 1

        # colors
        if color_mapping == 'index':

            if np.any(self._S.y.data):
                colors = self._S.y.data
            else:
                colors = np.array(range(self._S.shape[0]))

        elif color_mapping == 'labels':

            labels = list(set(self._S.y.labels))
            colors = [labels.index(l) for l in self._S.y.labels]

        if len(pcs) == 2:
            # bidimentional score plot

            fig = plt.figure(**kwargs)
            ax = fig.add_subplot(111)
            ax.set_title('Score plot')

            ax.set_xlabel('PC# {} ({:.3f}%)'.format(
                                           pcs[0], self.ev_ratio.data[pcs[0]]))
            ax.set_ylabel('PC# {} ({:.3f}%)'.format(
                                           pcs[1], self.ev_ratio.data[pcs[1]]))
            ax.scatter( self._S.masked_data[:, pcs[0]],
                        self._S.masked_data[:, pcs[1]],
                        s=30,
                        c=colors,
                        cmap=colormap)

            number_x_labels = plotter_preferences.number_of_x_labels  # get
            # from config
            number_y_labels = plotter_preferences.number_of_y_labels
            # the next two line are to avoid multipliers in axis scale
            y_formatter = ScalarFormatter(useOffset=False)
            ax.yaxis.set_major_formatter(y_formatter)
            ax.xaxis.set_major_locator(MaxNLocator(number_x_labels))
            ax.yaxis.set_major_locator(MaxNLocator(number_y_labels))
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')

        if len(pcs) == 3:
            # tridimensional score plot
            fig = plt.figure(**kwargs)
            ax = plt.axes(projection='3d')
            ax.set_title('Score plot')
            ax.set_xlabel(
                    'PC# {} ({:.3f}%)'.format(pcs[0], self.ev_ratio.data[pcs[
                        0]]))
            ax.set_ylabel(
                    'PC# {} ({:.3f}%)'.format(pcs[1], self.ev_ratio.data[pcs[
                        1]]))
            ax.set_zlabel(
                    'PC# {} ({:.3f}%)'.format(pcs[2], self.ev_ratio.data[pcs[
                        2]]))
            ax.scatter(self._S.masked_data[:, pcs[0]],
                       self._S.masked_data[:, pcs[1]],
                       self._S.masked_data[:, pcs[2]],
                       zdir='z',
                       s=30,
                       c=colors,
                       cmap=colormap,
                       depthshade=True)

if __name__ == '__main__':

    from tests.conftest import IR_source_2D
    from spectrochempy.api import *

    source = IR_source_2D()

    # columns masking
    source[:, 1320.0:840.0] = masked  # do not forget to use float in slicing

    # row masking
    source[10:12] = masked

    ax = source.plot_stack()

    pca = PCA(source) #, standardized=True, scaled=True)
    Pt, T = pca.transform(n_pc=6)

    pca.printev(n_pc=6)

    pca.Pt.plot_stack()
    pca.screeplot(n_pc=6)
    pca.scoreplot(1,2)
    pca.scoreplot(1,2,3)

    Xp = pca.inverse_transform(n_pc=6)
    Xp.plot_stack()

    show()
