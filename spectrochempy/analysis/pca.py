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

from spectrochempy.dataset.nddataset import NDDataset, CoordSet
from spectrochempy.dataset.ndcoords import Coord
from spectrochempy.analysis.svd import SVD
from spectrochempy.processors.numpy import diag, dot

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

        if npc is None:
            npc = min(Xc.shape)

        svd = SVD(X)

        # select npc loadings & compute scores
        # ------------------------------------
        sigma = diag(svd.s)
        s = sigma[:npc, :npc]
        u = svd.U[:, :npc]
        T = dot(u, s)

        T.title = 'scores (T) of ' + X.name
        T.coordset = CoordSet(X.y,
                              Coord([ i+1 for i in range(npc)],
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

        #: Explained Variance per singular values
        self.ev_ratio= svd.ev_ratio

        #: Cumulative Explained Variance
        self.ev_cum = svd.ev_cum

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

    def screeplot(self, npc=5, nfig=None):
        """scree plot of explained variance + cumulative variance by pca or svd
        :param npc: number of components to plot
        :type npc: int
        :param nfig: figure number. If None (default), a new figure is made
        :type nfig: int"""

        if nfig is None:
            plt.figure()
        else:
            plt.figure(nfig)

        plt.title('Scree plot')
        plt.xlabel('PC #')
        plt.ylabel('per PC / %')
        plt.bar(np.arange(npc), self.ev_ratio[0:npc], align='center')
        plt.twinx()
        plt.plot(np.arange(npc), self.ev_cum[0:npc], '-ro')
        plt.ylabel('cummulative / %')
        # plt.ylim((0.,100.))

        #if not _do_not_block:
        plt.show()
        #return

    def scoreplot(self, pcs, nfig=None):
        """2D or 3D scoreplot of samples
        :param pcs: set of  or 3 pcs
        :type npc: list or tuplet of int with 2 or 3 elements
        :param nfig: figure number. If None (default), a new figure is made
        :type nfig: int"""

        if nfig is None:
            fig = plt.figure()
        else:
            fig = plt.figure(nfig)

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
    #source[:, 1240.0:920.0] = masked  # do not forget to use float in slicing
    # row masking
    #source[10:12] = masked

    ax = source.plot_stack()


    center = np.mean(source, axis=0)

    center.plot()


    Xc1 = source - center
    Xc1.plot_stack()

    pca = PCA(source)

    pca.printev(npc=5)

    # columns masking
    source[:, 1240.0:920.0] = masked  # do not forget to use float in slicing
    # row masking
    source[10:12] = masked

    ax = source.plot_stack()


    center = np.mean(source, axis=0)

    center.plot()


    Xc2 = source - center
    Xc2.plot_stack()

    pca = PCA(source)

    pca.printev(npc=5)

    assert str(pca)[:3] == '\nPC'

    show()
