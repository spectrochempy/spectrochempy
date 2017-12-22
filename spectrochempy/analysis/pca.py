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

    This class performs a Principal Component Analysis of a NDDataset

    If the dataset contains masked values, the corresponding ranges are
    ignored in the calculation.

    """

    T = Instance(NDDataset)
    Pt = Instance(NDDataset)
    center = Instance(NDDataset)

    ev = Instance(NDDataset)
    ev_ratio = Instance(NDDataset)
    ev_cum = Instance(NDDataset)

    def __init__(self, X, npc=None):
        """
        Parameters
        -----------
        X : :class:`~spectrochempy.dataset.nddataset.NDDataset`object.
            The dataset has shape (``M``, ``N``)
        npc : int, optional
            The number of components to compute. If not set all components
            are computed.

        """

        # mean center the dataset
        # -----------------------

        self.center = center = np.mean(X, axis=0)
        Xc = X - center

        svd = SVD(X)

        # select npc loadings & compute scores
        # ------------------------------------

        if npc is None:
            npc = svd.s.size

        sigma = diag(svd.s)
        s = sigma[:npc, :npc]
        u = svd.U[:, :npc]
        T = dot(u, s)

        T.title = 'scores (T) of ' + X.name
        T.coordset = CoordSet(X.y,
                              Coord(None,
                              labels=['#%d' % (i+1) for i in range(npc)],
                              title='PC')
                              )

        T.description = 'scores (T) of ' + X.name
        T.history = 'created by PCA'

        Pt = svd.VT[0:npc]
        Pt.title = 'Loadings (P.t) of ' + X.name
        Pt.history = 'created by PCA'

        # scores (T) and loading (Pt) matrices
        # ------------------------------------

        self.T = T
        self.Pt = Pt

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

    # special methods
    # -----------------
    def __str__(self, npc=10):

        s = '\nPC\t\tEigenvalue\t\t%variance\t' \
            '%cumulative\n'
        s += '   \t\tof cov(X)\t\t per PC\t' \
             '     variance\n'
        for i in range(npc):
            tuple = (i, self.ev.data[i], self.ev_ratio.data[i],
                     self.ev_cum.data[i])
            s += '#{}  \t{:8.3e}\t\t {:6.3f}\t      {:6.3f}\n'.format(*tuple)

        return s

    # Public methods
    # -----------------

    def construct(self, npc=5):
        """reconstructs a dataset with npc PC's

        Parameters
        ----------
        npc : int, optional, default=10

            The number of PC to use for the reconstruction

        """
        #TODO: make a function to performs dot on two datasets

        X = self.center + np.dot(self.T.data[:, 0:npc], self.Pt.data[0:npc, :])
        X = NDDataset(X)
        X.name = 'PCA constructed Dataset with {} PCs'.format(npc)
        X.coordset = CoordSet(self.T.coords(0).copy(), self.Pt.coords(1).copy())
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
                  npc=5, **kwargs):
        """
        Scree plot of explained variance + cumulative variance by PCA

        Parameters
        ----------
        npc: int
            Number of components to plot

        """
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


    def scoreplot(self, pcs):
        """
        2D or 3D scoreplot of samples

        Parameters
        ----------
        pcs: list or tuple of int
            Must contain 2 or 3 elements

        """

        col = self.T.coordset[0]

        plt.title('Score plot')
        if len(pcs) == 2:
            plt.xlabel('PC# {} ({:.3f}%)'.format(pcs[0], self.ev_ratio[pcs[0]]))
            plt.ylabel('PC# {} ({:.3f}%)'.format(pcs[1], self.ev_ratio[pcs[1]]))
            plt.scatter(self.T.data[:, pcs[0]], self.T.data[:, pcs[1]], s=30,
                        c=col)

        if len(pcs) == 3:
            ax = fig.add_subplot(111,
                                 projection='3d')  # FIXME: projection does not work
            ax.set_title('Score plot')
            ax.set_xlabel(
                    'PC# {} ({:.3f}%)'.format(pcs[0], self.ev_ratio[pcs[0]]))
            ax.set_ylabel(
                    'PC# {} ({:.3f}%)'.format(pcs[1], self.ev_ratio[pcs[1]]))
            ax.set_zlabel(
                    'PC# {} ({:.3f}%)'.format(pcs[2], self.ev_ratio[pcs[2]]))
            ax.scatter(self.T.data[:, pcs[0]], self.T.data[:, pcs[1]],
                       self.T.data[:, pcs[2]], zdir='z', s=30, c=col,
                       depthshade=True)

        #if not _do_not_block:
        plt.show()

        return

if __name__ == '__main__':

    from tests.conftest import IR_source_2D
    from spectrochempy.api import *

    source = IR_source_2D()

    # columns masking
    source[:, 1240.0:920.0] = masked  # do not forget to use float in slicing

    # row masking
    source[10:12] = masked

    ax = source.plot_stack()

    pca = PCA(source)

    pca.printev(npc=6)

    pca.screeplot(npc=6)

    show()
