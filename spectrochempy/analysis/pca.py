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

    def __init__(self, X, npc=None,
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
        self._svd = svd = SVD(Xsc)

        self._sigma = diag(svd.s)

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

    def __str__(self, npc=10):

        s = '\nPC\t\tEigenvalue\t\t%variance\t' \
            '%cumulative\n'
        s += '   \t\tof cov(X)\t\t per PC\t' \
             '     variance\n'
        for i in range(npc):
            tuple = (
            i, self.ev.data[i], self.ev_ratio.data[i], self.ev_cum.data[i])
            s += '#{}  \t{:8.3e}\t\t {:6.3f}\t      {:6.3f}\n'.format(*tuple)

        return s

    # ------------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------------

    def transform(self, npc=None):
        """
        Apply the dimensionality reduction

        Parameters
        ----------
        npc : int, optional, default=10
            The number of components to compute. If not set all components
            are computed.

        Returns
        -------
        Pt : :class:`~spectrochempy.dataset.nddataset.NDDataset`.
            Loadings
        T : :class:`~spectrochempy.dataset.nddataset.NDDataset`.
            Scores


        """

        X = self._X

        # select npc loadings & compute scores
        # ------------------------------------

        if npc is None:
            npc = self._svd.s.size

        s = self._sigma[:npc, :npc]
        u = self._svd.U[:, :npc]
        T = dot(u, s)

        T.title = 'scores (T) of ' + X.name
        T.coordset = CoordSet(X.y,
                              Coord(None,
                              labels=['#%d' % (i+1) for i in range(npc)],
                              title='PC')
                              )

        T.description = 'scores (T) of ' + X.name
        T.history = 'created by PCA'

        Pt = self._svd.VT[0:npc]
        Pt.title = 'Loadings (P.t) of ' + X.name
        Pt.history = 'created by PCA'

        # scores (T) and loading (Pt) matrices
        # ------------------------------------

        self.T = T
        self.Pt = Pt

        return Pt, T


    def inverse_transform(self, npc=None):
        """
        Transform data back to the original space using the given number of
        PC's.

        Parameters
        ----------
        npc : int, optional, default=10
            The number of PC to use for the reconstruction

        Return
        ------
        X : :class:`~spectrochempy.dataset.nddataset.NDDataset`.

        """

        if npc is None:
            npc = self.T.shape[-1]
        else:
            npc = min(npc, self.T.shape[-1])

        T = self.T[:, :npc]
        Pt = self.Pt[:npc]

        X = dot(T, Pt)

        # try ti reconstruct something close to the original scaled,
        # standardized or centered data
        if self._scaled:
            X *= self._ampl
            X += self._min
        if self._standardized:
            X *= self._std
        if self._centered:
            X += self._center

        X.history = 'PCA reconstructed Dataset with {} PCs'.format(npc)
        X.title = self._X.title
        return X

    def printev(self, npc=10):
        """prints figures of merit: eigenvalues and explained variance
        for the first npc PS's

        Parameters
        ----------
        npc : int, optional, default=10

          The number of PC to print

        """
        print((self.__str__(npc)))

    def screeplot(self,
                  npc=5,
                  **kwargs):
        """
        Scree plot of explained variance + cumulative variance by PCA

        Parameters
        ----------
        npc: int
            Number of components to plot

        """
        npc = min(npc, self.ev.size)
        color1, color2 = kwargs.get('colors', [NBlue, NRed])
        pen = kwargs.get('pen', True)
        ylim1, ylim2 = kwargs.get('ylims', [(0,100), 'auto'])

        if ylim2 == 'auto':
            y1 = np.around(self.ev_ratio.data[0]*.95,-1)
            y2 = 101.
            ylim2 = (y1, y2)

        ax1 = self.ev_ratio[:npc].plot_bar(ylim = ylim1,
                                           color = color1,
                                           title='Scree plot')
        ax2 = self.ev_cum[:npc].plot_scatter(ylim = ylim2,
                                             color=color2,
                                             pen=True,
                                             markersize = 7.,
                                             twinx = ax1
                                            )
        return ax1, ax2


    def scoreplot(self, *pcs, cmap='jet', **kwargs):
        """
        2D or 3D scoreplot of samples

        Parameters
        ----------
        *pcs: a series of int argument or a list/tuple
            Must contain 2 or 3 elements

        Examples
        --------
        >>> pca.scoreplot(1,2)

        """

        if isinstance(pcs[0], (list,tuple, set)):
            pcs = pcs[0]

        # transform to internal index of unitary vectors (1->0 etc...)
        pcs = np.array(pcs) - 1

        # colors
        if np.any(self.T.y.data):
            colors = self.T.y.data
        else:
            colors = np.array(range(self.T.shape[0]))

        if len(pcs) == 2:

            fig = plt.figure(**kwargs)
            ax = fig.add_subplot(111)
            ax.set_title('Score plot')

            ax.set_xlabel('PC# {} ({:.3f}%)'.format(
                                           pcs[0], self.ev_ratio.data[pcs[0]]))
            ax.set_ylabel('PC# {} ({:.3f}%)'.format(
                                           pcs[1], self.ev_ratio.data[pcs[1]]))
            ax.scatter( self.T.masked_data[:, pcs[0]],
                        self.T.masked_data[:, pcs[1]],
                        s=30,
                        c=colors,
                        cmap=cmap)

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
            ax.scatter(self.T.masked_data[:, pcs[0]],
                       self.T.masked_data[:, pcs[1]],
                       self.T.masked_data[:, pcs[2]],
                       zdir='z',
                       s=30,
                       c=colors,
                       cmap=cmap,
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
    Pt, T = pca.transform(npc=6)

    pca.printev(npc=6)

    pca.Pt.plot_stack()
    pca.screeplot(npc=6)
    pca.scoreplot(1,2)
    pca.scoreplot(1,2,3)

    Xp = pca.inverse_transform(npc=6)
    Xp.plot_stack()

    show()
