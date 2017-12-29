# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
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


        # --------------------------------------------------------------------
        # forward analysis
        # --------------------------------------------------------------------

        # max number of components
        K = min(M, N)

        self.fefa = f = NDDataset(np.zeros((M, K)),
                      coordset = [X.y.copy(), Coord(range(K))],
                      name = 'Forward EFA of ' + X.name)

        # in case some row are masked, take this into account, by masking
        # the corresponding rows of f
        masked_rows = np.all(X.mask, axis=-1)
        f[masked_rows] = masked

        # performs the analysis
        for i in range(M):
            # if some rows are masked, we must skip them
            if not masked_rows[i]:
                fsvd = SVD(X[:i+1], compute_uv=False)
                # but if some rows are masked there is another complication
                # as the size of svd.s will be less than the size of the
                # destination! try to correct this.
                # create a temporary masked array to contains calculated
                # svd.s values
                stemp = np.ma.zeros((K,))
                stemp.mask = masked_rows
                stemp[~masked_rows[:i+1]] = fsvd.s ** 2
                f[i] = stemp
                f[i, i + 1:] = masked
            else:
                f[i] = masked

        # --------------------------------------------------------------------
        # backward analysis
        # --------------------------------------------------------------------

        self.befa = b = NDDataset(np.zeros((M, K)),
                      coordset = [X.y.copy(), Coord(range(K))],
                      name = 'Backward EFA of ' + X.name)

        b[masked_rows] = masked

        for i in range(M - 1, -1, -1):
            # if some rows are masked, we must skip them
            if not masked_rows[i]:
                bsvd = SVD(X[i:M], compute_uv=False)
                # but if some rows are masked there is another complication
                # as the size of svd.s will be less than the size of the
                # destination! try to correct this
                stemp = np.ma.zeros((K,))
                stemp.mask = masked_rows
                stemp[~masked_rows[ :M-i]] = bsvd.s ** 2
                b[i] = stemp
                b[i, :M - i] = masked
            else:
                b[i] = masked


    def get_conc(self, npc=3, order='fifo', plot=True):
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
        npc = min(K, npc)

        f = self.fefa
        b = self.befa
        c = NDDataset(np.zeros((M, npc)),
                      coordset=[self._X.y.copy(), Coord(range(npc),
                                                        title='PC#')],
                      name = 'C_EFA[{}]'.format(self._X.name),
                      title = 'relative concentration',
                      )
        masked_rows = np.all(self._X.mask, axis=-1)

        for i in range(K):
            if masked_rows[i]:
                c[i] = masked
                continue
            for j in range(npc):
                c[i, j] = min([f[i, j], b[i, npc - j - 1]])
                if c[i, j] == 0:
                    c[i, j] = masked

        if plot:

            profiles = [c.T[j] for j in range(npc)]

            labels = ["PC#%d" % i for i in range(npc)]

            plot_multiple(profiles, labels=labels, yscale='log', legend=True)

            show()

        return c


# ============================================================================
if __name__ == '__main__':

    pass




