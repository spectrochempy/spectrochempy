# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================




__all__ = ['Pca']

import matplotlib.pyplot as plt
from traitlets import HasTraits, Instance
from spectrochempy.dataset.nddataset import NDDataset
import numpy as np

from spectrochempy.dataset.ndcoords import Coord, CoordSet
from .svd import SVD

# ==============================================================================
# Global preferences
# ==============================================================================
from spectrochempy.application import app

plotter_preferences = app.plotter_preferences
log = app.log
preferences = app


# ==============================================================================
# class Pca
# ==============================================================================
class Pca(HasTraits):
    """
    Principal Component Analysis

    This class performs a Principal Component Analysis of a NDDataset

    If the dataset contains masked values, the corresponding ranges are
    ignored in the calculation.

    """

    T = Instance(NDDataset)
    Pt = Instance(NDDataset)
    center = Instance(np.ndarray)

    # private attributes
    _ev = Instance(np.ndarray)

    def __init__(self, X, npc=None):
        """
        Parameters
        -----------
        X : :class:`~spectrochempy.dataset.nddataset.NDDataset`object.
            The dataset has shape (``N``, ``M``)
        npc : int, optional
            The number of components to compute. If not set all components
            are computed.

        """

        # check if we have the correct input
        if isinstance(X, NDDataset):
            data = X.data
        else:
            raise TypeError('A dataset of type NDDataset is  '
                               'expected as a source of data, but an object'
                               ' of type {} has been provided'.format(
                               type(X).__name__))

        # mean center the dataset
        self.center = center = np.mean(X.masked_data, axis=0)
        X = X.copy()
        X.data = X.data - center

        if npc is None:
            npc = min(X.shape)

        Xsvd = Svd(X)

        # select npc loadings & compute scores
        T = np.dot(Xsvd.U.data[:, 0:npc], np.diag(Xsvd.s)[0:npc, 0:npc])
        T = NDDataset(T)
        T.title = 'scores (T) of ' + X.name
        T.coordset = CoordSet(X.coordset[0],
                              Coord([ i+1 for i in range(len(Xsvd.s))],
                                labels=['#%d' % (i+1) for i in range(len(Xsvd.s))],
                                title='PC')
                              )
        T.description = 'scores (T) of ' + X.name
        T.history = 'created by Pca'

        Pt = Xsvd.Vt[0:npc]
        Pt.title = 'Loadings (P.t) of ' + X.name
        Pt.history = 'created by Pca'

        # scores (T) and loading (Pt) matrixes
        self.T = T
        self.Pt = Pt
        # eigenvalues of the covariance matrix
        self._ev = (Xsvd.s * Xsvd.s) / (np.size(Xsvd.s) - 1)

        return

    # Properties
    # ------------
    @property
    def ev_ratio(self):
        """% Explained Variance per PC"""
        return 100 * self._ev / sum(self._ev)

    @property
    def ev_cum(self):
        """% Cummulative Explained Variance"""
        return np.cumsum(self.ev_ratio)

    # special methods
    # -----------------
    def __str__(self, npc=10):

        s = '\nPC\t\tEigenvalue\t\t%variance\t' \
            '%cumulative\n'
        s += '   \t\tof cov(X)\t\t per PC\t' \
             '     variance\n'
        for i in np.arange(npc):
            tuple = (i, self._ev[i], self.ev_ratio[i], self.ev_cum[i])
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
        """scree plot of explained variance + cummulative variance by pca or svd
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
