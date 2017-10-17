# -*- coding: utf-8; tab-width: 4; indent-tabs-mode: t; python-indent: 4 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
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

__all__ = ['Efa']
_classes = ['Efa']

import numpy as np
from scipy.linalg import svdvals

from ..core.api import NDDataset


class Efa:
    """Evolving Factor Analysis"""

    def __init__(self, X, npc=None):
        """  Performs an Evolving Factor Analysis of the dataset X
        :param X: dataset
        :param plots: display plots of forward and backward analysis
        :param npc: number of singular values to display (if None: display all \
        singular values
        :param axis: the axis on which the evolving sv are displayed (useful \
        only if the dataset has several axis in the 1st dim)

        attributes
        f : Eigenvalues of the forward analysis
        b : Eigenvalues of the backward analysis
        c : estimated concentration profile (first in - first out)
        """

        NaNColumns = np.any(np.isnan(X.data), axis=0)
        packeddata = X.data[:, ~ NaNColumns]

        n = X.shape[0]
        m = packeddata.shape[1]
        k = min(n, m)

        # forward analysis
        f = np.ones((n, k)) * np.nan
        for i in range(n):
            s = svdvals(packeddata[0:i + 1, :], False, False)
            f[i, 0:i + 1] = s * s  # s[0:i+1]*s[0:i+1]

        # backward analysis
        b = np.ones((n, k)) * np.nan
        for i in np.arange(n - 1, -1, -1):
            s = svdvals(packeddata[i:n, :], False, False)
            b[i, 0:n - i] = s * s

        self.__datasetName = X.name
        self.f = NDDataset(f)
        self.f.name = 'forward efa of ' + X.name
        self.f.dims[0] = X.dims[0].copy()

        self.b = NDDataset(b)
        self.b.name = 'backward efa of ' + X.name
        self.b.dims[0] = X.dims[0].copy()

    def c(self, npc=3, order='fifo'):
        """ Computes abstract concentration profile """

        n, k = np.shape(self.f.data)
        c = np.zeros((n, npc))
        f = np.nan_to_num(self.f.data)
        b = np.nan_to_num(self.b.data)
        for i in range(k):
            for j in range(npc):
                c[i, j] = min([f[i, j], b[i, npc - j - 1]])
                if c[i, j] == 0:
                    c[i, j] = 1e-30

        c = NDDataset(c)
        c.name = 'C_EFA[' + self.__datasetName + ']'
        c.dims[0] = self.b.dims[0].copy()
        return c


        #    if plots == True:
        #        if npc == None:
        #            npcp = k
        #        else:
        #            npcp = npc
        #        plt.figure()
        #        plt.subplot(3, 1, 1)
        #        if len(X.dims[0].axes[axis]) ==0 :
        #            plt.plot(np.log10(f[:,0:npcp]))
        #            plt.xlabel('row(X)')
        #        else:
        #            plt.plot(X.dims[0].axes[axis].values, np.log10(f[:,0:npcp]))
        #            plt.xlabel(X.dims[0].axes[axis].name)
        #        plt.title('Forward EFA')
        #        plt.ylabel('Log(sv)')
        #
        #        plt.subplot(3, 1, 2)
        #        if len(X.dims[0].axes[axis]) == 0:
        #            plt.plot(np.log10(b[:,0:npcp]))
        #            plt.xlabel('row(X)')
        #        else:
        #            plt.plot(X.dims[0].axes[axis].values, np.log10(b[:,0:npcp]))
        #            plt.xlabel(X.dims[0].axes[axis].name)
        #        plt.title('Backward EFA')
        #        plt.ylabel('Log(sv)')
        #
        #        plt.subplot(3, 1, 3)
        #        if len(X.dims[0].axes[axis]) == 0:
        #            plt.plot(c)
        #            plt.xlabel('row(X)')
        #        else:
        #            plt.plot(X.dims[0].axes[axis].values, c)
        #            plt.xlabel(X.dims[0].axes[axis].name)
        #        plt.title('Abstract concentration profile')
        #        plt.ylabel('sv')
        #
        #        plt.show()
        #
        #    return f, b, c
