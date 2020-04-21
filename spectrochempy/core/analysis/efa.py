# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

"""
This module implement the EFA (Evolving Factor Analysis) class.

"""

__all__ = ['EFA']

__dataset_methods__ = []

# ----------------------------------------------------------------------------------------------------------------------
# third party imports
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
from datetime import datetime
from traitlets import HasTraits, Instance

# ----------------------------------------------------------------------------------------------------------------------
# localimports
# ----------------------------------------------------------------------------------------------------------------------

from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.dataset.ndcoordset import CoordSet
from spectrochempy.core.dataset.ndcoord import Coord
from spectrochempy.utils import docstrings, MASKED
from spectrochempy.core.analysis.svd import SVD
from spectrochempy.core.plotters.plot1d import plot_multiple
from spectrochempy.utils import show


class EFA(HasTraits):
    """
    Evolving Factor Analysis

    Performs an Evolving Factor Analysis (forward and reverse) of
    the |NDDataset| `X`.

    """

    f = Instance(NDDataset)
    """|NDDataset| - Eigenvalues for the forward analysis"""

    b = Instance(NDDataset)
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
            raise TypeError(f'An object of type NDDataset is expected as input, but an object of type'
                            f' {type(X).__name__} has been provided')

        # max number of components
        K = min(M, N)

        # in case some row are masked, we need to take this into account
        if X.is_masked:
            masked_rows = np.all(X.mask, axis=-1)
        else:
            masked_rows = np.array([False] * M)

        K = min(K, len(np.where(~ masked_rows)[0]))

        # --------------------------------------------------------------------
        # forward analysis
        # --------------------------------------------------------------------

        f = NDDataset(np.zeros((M, K)),
                               coords=[X.y, Coord(range(K))],
                               title='EigenValues',
                               description='Forward EFA of ' + X.name,
                               history=str(datetime.now()) + ': created by spectrochempy ')

        # in case some row are masked, take this into account, by masking
        # the corresponding rows of f
        f[masked_rows] = MASKED

        # performs the analysis
        for i in range(M):
            # if some rows are masked, we must skip them
            if not masked_rows[i]:
                fsvd = SVD(X[:i + 1], compute_uv=False)
                k = fsvd.s.size
                # print(i, k)
                f[i, :k] = fsvd.s.data ** 2
                f[i, k:] = MASKED
            else:
                f[i] = MASKED

        # --------------------------------------------------------------------
        # backward analysis
        # --------------------------------------------------------------------

        b = NDDataset(np.zeros((M, K)),
                                  coords=[X.y, Coord(range(K))],
                                  title='EigenValues',
                                  name='Backward EFA of ' + X.name,
                                  history=str(datetime.now()) + ': created by spectrochempy ')

        b[masked_rows] = MASKED

        for i in range(M - 1, -1, -1):
            # if some rows are masked, we must skip them
            if not masked_rows[i]:
                bsvd = SVD(X[i:M], compute_uv=False)
                k = bsvd.s.size
                b[i, :k] = bsvd.s.data ** 2
                b[i, k:] = MASKED
            else:
                b[i] = MASKED

        self.f = f
        self.b = b


    def cut_f(self, n_pc=None, cutoff=None):
        """

        Parameters
        ----------
        n_pc
        cutoff

        Returns
        -------

        """
        M, K = self.f.shape
        if n_pc is None:
            n_pc = K
        n_pc = min(K, n_pc)

        f = self.f
        if cutoff is not None:
            f.data = np.max((f.data, np.ones_like(f.data) * cutoff), axis=0)
        return f

    def cut_b(self, n_pc=None, cutoff=None):
        """

        Parameters
        ----------
        n_pc
        cutoff

        Returns
        -------

        """
        M, K = self.b.shape
        if n_pc is None:
            n_pc = K
        n_pc = min(K, n_pc)

        b = self.b
        if cutoff is not None:
            b.data = np.max((b.data, np.ones_like(b.data) * cutoff), axis=0)
        return b

    def get_conc(self, n_pc=None, cutoff=None):
        """
        Computes abstract concentration profile (first in - first out)

        Parameters
        ----------
        n_pc : int, optional, default:3
            Number of pure species for which the concentration profile must be
            computed.
        order : str, [not used]
        plot : bool, optional, default:True
            whether or not to display a graph with the concentration profiles

        Returns
        --------
        |NDDataset|
            Concentration profile

        """
        M, K = self.f.shape
        if n_pc is None:
            n_pc = K
        n_pc = min(K, n_pc)

        f = self.cut_f(n_pc, cutoff)
        b = self.cut_b(n_pc, cutoff)

        xcoord = Coord(range(n_pc), title='PS#')
        c = NDDataset(np.zeros((M, n_pc)),
                      coords=CoordSet(y=self._X.y, x=xcoord),
                      name='C_EFA[{}]'.format(self._X.name),
                      title='relative concentration',
                      description='Concentration profile from EFA',
                      history=str(datetime.now()) + ': created by spectrochempy')
        if self._X.is_masked:
            masked_rows = np.all(self._X.mask, axis=-1)
        else:
            masked_rows = np.array([False] * M)

        for i in range(M):
            if masked_rows[i]:
                c[i] = MASKED
                continue
            c[i] = np.min((f.data[i, :n_pc], b.data[i, :n_pc][::-1]), axis=0)
        return c



# ======================================================================================================================
if __name__ == '__main__':
    pass
