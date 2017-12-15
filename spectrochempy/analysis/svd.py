# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================




__all__ = ['Svd']

from traitlets import HasTraits, Instance
from spectrochempy.dataset.nddataset import NDDataset, CoordSet
import numpy as np

from spectrochempy.dataset.ndcoords import Coord


class Svd(HasTraits):
    """Performs a Singular Value Decomposition of a dataset.
    
    The SVD is commonly written as :math:'X = ur \Sigma V^{T}'. This class \
    has the attributes: ur, s = diag(S) and Vt=V.T.
    If the dataset contains NaN values (e.g. if parts of the spectra have\
    been blanked, the corresponding ranges are ignored in the calculation.

    Parameters
    -----------

    X : :class:`~spectrochempy.dataset.nddataset.NDDataset` object of
    shape (``N``, ``M``)

    full_matrices : `bool`, optional, default=`False`.

        If `False` , ur and Vt have the shapes\
        (`N`,  `k`) and (`k`, `M`), respectively, where `k` = min(`N`, `M`). \
        Otherwise the shapes will be (`M`, `M`) and (`N`, `N`), respectively.

    compute_uv: `bool`, optiopnal, default=True.

        Whether or not to compute ur and Vt in addition to s.

    Attributes
    ----------

    ur : :class:`~spectrochempy.dataset.nddataset.NDDataset`.

        ur contains the left unitary matrix.
        Its shape depends on `full_matrices`.

    s : `numpy.ndarray`. vector of singular values

    Vt : :class:`~spectrochempy.dataset.dataset.NDDataset`.

        Vt contains a transpose matrix of the Loadings.
        Its shape depends on `full_matrices`
        :rtype self.ur: Dataset

    Examples
    ---------
    >>> source = NDDataset.load('mydataset.scp')
    >>appvd = Svd(source)
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

    def __init__(self, source, full_matrices=False, compute_uv=True):
        """ constructor of svd object """

        # check if we have the correct input
        if isinstance(source, NDDataset):
            data = source.data
        else:
            raise TypeError('A dataset of type NDDataset is  '
                               'expected as a source of data, but an object'
                               ' of type {} has been provided'.format(
                               type(source).__name__))

        # retains valid columns #TODO: mask can do this already
        NaNColumns = np.any(np.isnan(data), axis=0)
        data = data[:, ~ NaNColumns]

        # makes SVD
        U, s, Vt = np.linalg.svd(data, full_matrices, compute_uv)

        # Put back columns with NaN in Vt
        if any(NaNColumns):
            Vt2 = np.zeros((s.shape[0], data.shape[1]))
            j = 0
            for i in np.arange(len(NaNColumns)):
                if ~ NaNColumns[i]:
                    Vt2[:, i] = Vt[:, j]
                    j = j + 1
                else:
                    Vt2[:, i] = float('nan') * np.ones((Vt2.shape[0],))
            Vt_ = Vt2

        # Returns U as a NDDataset object
        U = NDDataset(U)
        U.title = 'left singular vectors of ' + source.name
        U.coordset = CoordSet(source.coordset[0],
                              Coord([i+1 for i in range(len(s))],
                                labels=['#%d' % (i+1) for i in range(len(s))],
                                title='Unitary vectors')
                          )
        U.description = 'left singular vectors of ' + source.name
        U.history = 'created by Svd \n'

        # Returns the loadings (Vt) as a NDDataset object
        Vt = NDDataset(Vt)
        Vt.title = 'Loadings (V.t) of ' + source.name
        Vt.coordset = CoordSet(Coord([i+1 for i in range(len(s))],
                                 labels=['#%d' % (i+1) for i in range(len(s))],
                                 title='Unitary vectors'),
                           source.coordset[1])
        Vt.description = (
            'Loadings obtained by singular value decomposition of ' + source.name)
        Vt.history = (str(Vt.modified) + ': created by Svd \n')

        self.U = U
        self.s = s
        self.Vt = Vt

    # special methods
    # -----------------

    def __repr__(self):
        return '<svd: ur%s, s(%s), Vt%s>' % (
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
