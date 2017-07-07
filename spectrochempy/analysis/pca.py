# -*- coding: utf-8; tab-width: 4; indent-tabs-mode: t; python-indent: 4 -*-
#
# =============================================================================
# Copyright (©) 2015-2017 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
# This software is a computer program whose purpose is to [describe
# functionalities and technical features of your software].
#
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software. You can use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty and the software's author, the holder of the
# economic rights, and the successive licensors have only limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading, using, modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean that it is complicated to manipulate, and that also
# therefore means that it is reserved for developers and experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and, more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
# =============================================================================


__all__ = ['Pca']

import matplotlib.pyplot as plt

from traits.api import HasTraits, Property, Instance, Array

from ..core import NDDataset, Axes, Axis
from .svd import Svd

import numpy as np

# ==============================================================================
# Global preferences
# ==============================================================================
from spectrochempy.preferences.preference_manager import \
                                                        preference_manager as pm
_DO_NOT_BLOCK = pm.general._DO_NOT_BLOCK


# ==============================================================================
# class Pca
# ==============================================================================
class Pca(HasTraits):
    """
    Principal Component Analysis

    This class performs a Principal Component Analysis of a NDDataset

    Parameters
    -----------
    X : :class:`~spectrochempy.core.dataset.nddataset.NDDataset` object of
    shape (``N``, ``M``)

    npc : `int`, optional

        The number of components to compute. If not set all components
        are computed.

    """

    T = Instance(NDDataset)
    Pt = Instance(NDDataset)
    center = Array

    # private attributes
    _ev = Array

    def __init__(self, X, npc=None):
        '''Constructor'''

        # mean center the dataset
        self.center = Xmean = np.nanmean(X.data, axis=0)
        Xc = X.copy()
        Xc.data = X.data - Xmean

        if npc is None:
            npc = min(Xc.shape)

        Xsvd = Svd(Xc)

        # select npc loadings & compute scores
        T = np.dot(Xsvd.U.data[:, 0:npc], np.diag(Xsvd.s)[0:npc, 0:npc])
        T = NDDataset(T)
        T.name = 'scores (T) of ' + Xc.name
        T.axes = Axes(Axis(X.axes[0]),
                      Axis(None,
                           labels=['# %s' % i for i in range(len(Xsvd.s))],
                           title='PC')
                      )
        T.description = 'scores (T) of ' + Xc.name
        T.history = str(T.modified) + ': created by Pca \n'

        Pt = Xsvd.Vt[0:npc]
        Pt.name = 'Loadings (P.t) of ' + Xc.name
        Pt.history = str(T.modified) + ': created by Pca \n'

        # scores (T) and loading (Pt) matrixes
        self.T = T
        self.Pt = Pt
        # eigenvalues of the covariance matrix
        self._ev = (Xsvd.s * Xsvd.s) / (np.size(Xsvd.s) - 1)

        return

    # Properties
    #------------
    ev_ratio = Property

    def _get_ev_ratio(self):
        '''% Explained Variance per PC'''
        return 100 * self._ev / sum(self._ev)

    ev_cum = Property

    def _get_ev_cum(self):
        '''% Cummulative Explained Variance'''
        return np.cumsum(self.ev_ratio)

    # special methods
    #-----------------
    def __str__(self, npc=10):

        s =  '\nPC\t\tEigenvalue\t\t%variance\t' \
             '%cumulative\n'
        s += '   \t\tof cov(X)\t\t per PC\t' \
             '     variance\n'
        for i in np.arange(npc):
            tuple= (i, self._ev[i], self.ev_ratio[i], self.ev_cum[i])
            s += '#{}  \t{:8.3e}\t\t {:6.3f}\t      {:6.3f}\n'.format(*tuple)

        return s

    # Public methods
    #-----------------

    def construct(self, npc=5):
        '''reconstructs a dataset with npc PC's

        Parameters
        ----------
        npc : `int`, optional, default=10

            The number of PC to use for the reconstruction

        '''
        X = self.center + np.dot(self.T.data[:, 0:npc], self.Pt.data[0:npc, :])
        X = NDDataset(X)
        X.name = 'PCA constructed Dataset with {} PCs'.format(npc)
        X.axes = Axes(self.T.coords(0).copy(), self.Pt.coords(1).copy())
        return X


    def printev(self, npc=10):
        '''prints figures of merit: eigenvalues and explained variance for the first npc PS's

        Parameters
        ----------
        npc : `int`, optional, default=10

          The number of PC to print

        '''
        print( self.__str__(npc))


    def screeplot(self, npc=5, nfig=None):
        '''scree plot of explained variance + cummulative variance by pca or svd
        :param npc: number of components to plot
        :type npc: int
        :param nfig: figure number. If None (default), a new figure is made
        :type nfig: int'''

        if nfig == None:
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

        if  not _DO_NOT_BLOCK:
            plt.show()
        return

    def scoreplot(self, pcs, nfig=None):
        '''2D or 3D scoreplot of samples
        :param pcs: set of  or 3 pcs
        :type npc: list or tuplet of int with 2 or 3 elements
        :param nfig: figure number. If None (default), a new figure is made
        :type nfig: int'''

        if nfig == None:
            fig = plt.figure()
        else:
            fig = plt.figure(nfig)

        col = self.T.axes[0]

        plt.title('Score plot')
        if len(pcs) == 2:
            plt.xlabel('PC# {} ({:.3f}%)'.format(pcs[0], self.ev_ratio[pcs[0]]))
            plt.ylabel('PC# {} ({:.3f}%)'.format(pcs[1], self.ev_ratio[pcs[1]]))
            plt.scatter(self.T.data[:, pcs[0]], self.T.data[:, pcs[1]], s=30,
                        c=col)

        if len(pcs) == 3:
            ax = fig.add_subplot(111, projection='3d')  #FIXME: projection does not work
            ax.set_title('Score plot')
            ax.set_xlabel(
                'PC# {} ({:.3f}%)'.format(pcs[0], self.ev_ratio[pcs[0]]))
            ax.set_ylabel(
                'PC# {} ({:.3f}%)'.format(pcs[1], self.ev_ratio[pcs[1]]))
            ax.set_zlabel(
                'PC# {} ({:.3f}%)'.format(pcs[2], self.ev_ratio[pcs[2]]))
            ax.scatter(self.T.data[:, pcs[0]], self.T.data[:, pcs[1]],
                       self.T.data[:, pcs[2]], zdir=u'z', s=30, c=col,
                       depthshade=True)

        if not _DO_NOT_BLOCK:
            plt.show()

        return


