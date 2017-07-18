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


__all__ = ['Svd']

from traitlets import HasTraits, Instance

from spectrochempy.core.api import NDDataset, Axis, Axes

import numpy as np


class Svd(HasTraits):
    """Performs a Singular Value Decomposition of a dataset.
    
    The SVD is commonly written as :math:'X = ur \Sigma V^{T}'. This class \
    has the attributes: ur, s = diag(S) and Vt=V.T.
    If the dataset contains NaN values (e.g. if parts of the spectra have\
    been blanked, the corresponding ranges are ignored in the calculation.

    Parameters
    -----------

    X : :class:`~spectrochempy.core.dataset.nddataset.NDDataset` object of
    shape (``N``, ``M``)

    full_matrices : `bool`, optional, default=`False`.

        If `False` , ur and Vt have the shapes\
        (`N`,  `k`) and (`k`, `M`), respectively, where `k` = min(`N`, `M`). \
        Otherwise the shapes will be (`M`, `M`) and (`N`, `N`), respectively.

    compute_uv: `bool`, optiopnal, default=True.

        Whether or not to compute ur and Vt in addition to s.

    Attributes
    ----------

    ur : :class:`~spectrochempy.core.dataset.nddataset.NDDataset`.

        ur contains the left unitary matrix.
        Its shape depends on `full_matrices`.

    s : `numpy.ndarray`. vector of singular values

    Vt : :class:`~spectrochempy.core.dataset.dataset.NDDataset`.

        Vt contains a transpose matrix of the Loadings.
        Its shape depends on `full_matrices`
        :rtype self.ur: Dataset

    Examples
    ---------
    >>> source = NDDataset.load('mydataset.scp')
    >>> svd = Svd(source)
    >>> print(svd.ev)
    [1.18e+04      634 ...,  0.00109 0.000975]
    >>> print(svd.ev_cum)
    [1.18e+04 1.25e+04 ..., 1.25e+04 1.25e+04]
    >>> print(svd.ev_ratio)
    [   0.945   0.0506 ..., 8.69e-08 7.78e-08]

    """

    U = Instance(NDDataset)
    s = Instance(np.ndarray)
    Vt = Instance(NDDataset)

    def __init__(self, X, full_matrices=False, compute_uv=True):
        ''' constructor of svd object '''

        # retains valid columns
        NaNColumns = np.any(np.isnan(X.data), axis=0)
        data = X.data[:, ~ NaNColumns]

        # makes SVD
        U, s, Vt = np.linalg.svd(data, full_matrices, compute_uv)

        # Put back columns with NaN in Vt
        if any(NaNColumns):
            Vt2 = np.zeros((s.shape[0], X.shape[1]))
            j = 0
            for i in np.arange(len(NaNColumns)):
                if ~ NaNColumns[i]:
                    Vt2[:, i] = Vt[:, j]
                    j = j + 1
                else:
                    Vt2[:, i] = float('nan') * np.ones((Vt2.shape[0],))
            Vt = Vt2

            # Returns ur as a Dataset object
        U = NDDataset(U)
        U.name = 'left singular vectors of ' + X.name
        U.axes = Axes(Axis(X.axes[0]),
                      Axis(None,
                           labels=['# %s' % i for i in range(len(s))],
                           title='Unitary vectors')
                      )
        U.description = 'left singular vectors of ' + X.name
        U.history = str(U.modified) + ': created by Svd \n'

        # Returns the loadings (Vt) as a Dataset object
        Vt = NDDataset(Vt)
        Vt.name = 'Loadings (V.t) of ' + X.name
        Vt.axes = Axes(Axis(None,
                            labels=['# %s' % i for i in range(len(s))],
                            title='Unitary vectors'),
                       Axis(X.axes[1])
                       )
        Vt.description = (
        'Loadings obtained by singular value decomposition of ' + X.name)
        Vt.history = (str(Vt.modified) + ': created by Svd \n')

        self.U = U
        self.s = s
        self.Vt = Vt

    # special methods
    # -----------------

    def __repr__(self):
        return '<spectrochempy.svd: ur%s , s(%s), Vt%s>' % (
        self.U.shape, len(self.s), self.Vt.shape)

    # Properties
    # ----------

    @property
    def ev(self):
        """`numpy.ndarray`,  eigenvalues of the covariance matrix """
        return (self.s * self.s) / (np.size(self.s) - 1)

    @property
    def ev_cum(self):
        """`numpy.ndarray`,  Cummulative Explained Variance """
        return np.cumsum(self.ev)

    @property
    def ev_ratio(self):
        """`numpy.ndarray`,  Explained Variance per singular values """
        return self.ev / np.sum(self.ev)
