# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2019 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================

"""
This module implement the EFA (Evolving Factor Analysis) class.
"""

__all__ = ['EFA']

__dataset_methods__ = ['EFA']

# ----------------------------------------------------------------------------
# third party imports
# ----------------------------------------------------------------------------

import numpy as np
from traitlets import HasTraits, Instance

# ----------------------------------------------------------------------------
# localimports
# ----------------------------------------------------------------------------

from spectrochempy.dataset.nddataset import NDDataset
from spectrochempy.dataset.ndcoords import Coord
from spectrochempy.utils import docstrings, masked
from spectrochempy.core.analysis.svd import SVD
from spectrochempy.core.plotters.plot1d import plot_multiple
from spectrochempy.utils import show


class EFA(HasTraits):
    """
    Evolving Factor Analysis

    Performs an Evolving Factor Analysis (forward and reverse) of
    the |NDDataset| `X`.

    """

    fefa = Instance(NDDataset)
    """|NDDataset| - Eigenvalues for the forward analysis"""

    befa = Instance(NDDataset)
    """|NDDataset| - Eigenvalues for the backward analysis"""

    docstrings.keep_params('SVD.parameters', 'dataset')

    @docstrings.dedent
    def __init__(self, dataset):
        """
        Parameters
        ----------
        %(SVD.parameters.dataset)s

        """
        # check if we have the correct input
        # ----------------------------------

        X = dataset

        if isinstance(X, NDDataset):
            # As seen below, we cannot performs SVD on the masked array
            # so let's take the ndarray only
            self._X = X
            M, N = X.shape
        else:
            raise TypeError('A dataset of type NDDataset is  '
                               'expected as a dataset of data, but an object'
                               ' of type {} has been provided'.format(
                               type(X).__name__))

        # max number of components
        K = min(M, N)

        # in case some row are masked, we need to take this into account
        masked_rows = np.all(X.mask, axis=-1)

        K = min(K, len(np.where(~ masked_rows)[0]))

        # --------------------------------------------------------------------
        # forward analysis
        # --------------------------------------------------------------------

        self.fefa = f = NDDataset(np.zeros((M, K)),
                      coordset=[X.y.copy(), Coord(range(K))],
                      title= 'EigenValues',
                      name='Forward EFA of ' + X.name)

        # in case some row are masked, take this into account, by masking
        # the corresponding rows of f
        f[masked_rows] = masked

        # performs the analysis
        for i in range(M):
            # if some rows are masked, we must skip them
            if not masked_rows[i]:
                fsvd = SVD(X[:i+1], compute_uv=False)
                k = fsvd.s.size
                #print(i, k)
                f[i, :k] = fsvd.s.data ** 2
                f[i, k:] = masked
            else:
                f[i] = masked

        # --------------------------------------------------------------------
        # backward analysis
        # --------------------------------------------------------------------

        self.befa = b = NDDataset(np.zeros((M, K)),
                      coordset=[X.y.copy(), Coord(range(K))],
                      title='EigenValues',
                      name='Backward EFA of ' + X.name)

        b[masked_rows] = masked

        for i in range(M - 1, -1, -1):
            # if some rows are masked, we must skip them
            if not masked_rows[i]:
                bsvd = SVD(X[i:M], compute_uv=False)
                k = bsvd.s.size
                b[i, :k] = bsvd.s.data ** 2
                b[i, k:] = masked
            else:
                b[i] = masked

    def get_forward(self, npc=None, cutoff=None, plot=False, clear=True,
                    legend='best'):
        """

        Parameters
        ----------
        npc
        plot

        Returns
        -------

        """
        M, K = self.fefa.shape
        if npc is None:
            npc = K
        npc = min(K, npc)

        f = self.fefa
        if cutoff is not None:
            f.data = np.max((f.data, np.ones_like(f.data)*cutoff), axis=0)

        if plot:
            self._plot(f, npc, clear=clear, legend=legend)

        return f

    def get_backward(self, npc=None, cutoff=None, plot=False, clear=True,
                     legend='best'):
        """

        Parameters
        ----------
        npc
        plot

        Returns
        -------

        """
        M, K = self.befa.shape
        if npc is None:
            npc = K
        npc = min(K, npc)

        b = self.befa
        if cutoff is not None:
            b.data = np.max((b.data, np.ones_like(b.data)*cutoff), axis=0)

        if plot:
            self._plot(b, npc, clear=clear, legend=legend)

        return b

    def get_conc(self, npc=None, cutoff = None, order='fifo', plot=True):
        """
        Computes abstract concentration profile (first in - first out)

        Parameters
        ----------
        npc : int, optional, default:3
            Number of components for which the concentration profile must be
            computed.
        order : str, [not used]
        plot : bool, optional, default:True
            whether or not to display a graph with the concentration profiles

        Returns
        --------
        |NDDataset|
            Concentration profile

        """
        M, K = self.fefa.shape
        if npc is None:
            npc = K
        npc = min(K, npc)

        f = self.get_forward(npc, cutoff)
        b = self.get_backward(npc, cutoff)

        xcoord = Coord(range(npc), title='PC#')
        c = NDDataset(np.zeros((M, npc)),
                      coordset=[self._X.y.copy(), xcoord],
                      name = 'C_EFA[{}]'.format(self._X.name),
                      title = 'relative concentration',
                      )
        masked_rows = np.all(self._X.mask, axis=-1)

        for i in range(M):
            if masked_rows[i]:
                c[i] = masked
                continue
            # c[i] = np.min((f[i,:npc].data, b.data[i,-npc-1::-1]), axis=0)
            c[i] = np.min((f[i,:npc].data, b.data[i,:npc][::-1]), axis=0)

        if plot:
            self._plot(c, npc)

        return c

    def _plot(self, c, npc, clear=True, legend='best'):
        """

        Parameters
        ----------
        c
        npc

        Returns
        -------

        """

        profiles = [c.T[j] for j in range(npc)]

        labels = ["PC#%d" % i for i in range(npc)]

        plot_multiple(profiles, labels=labels, yscale='log',
                      clear=clear, legend=legend)


# ============================================================================
if __name__ == '__main__':

    pass




