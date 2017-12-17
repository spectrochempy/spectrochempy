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
import numpy as np

from spectrochempy.dataset.ndcoords import Coord
from spectrochempy.dataset.nddataset import NDDataset, CoordSet


# ----------------------------------------------------------------------------
class Svd(HasTraits):
    """Performs a Singular Value Decomposition of a dataset.
    
    The SVD is commonly written as :math:'X = ur \Sigma V^{T}'. This class \
    has the attributes: ur, s = diag(S) and Vt=V.T.

    If the dataset contains NaN values (e.g. if parts of the spectra have\
    been blanked, the corresponding ranges are ignored in the calculation.

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

    def __init__(self, X, full_matrices=False, compute_uv=True):
        """
    Parameters
    -----------
    X : :class:`~spectrochempy.dataset.nddataset.NDDataset` object.
        This nddataset must have a 2 shape (``N``, ``M``).
    full_matrices : bool, optional, default=``False``.
        If ``False`` , `ur` and `Vt` have the shapes (``N``,  ``k``) and
        (``k``, ``M``), respectively, where ``k`` = min(``N``, ``M``).
        Otherwise the shapes will be (``M``, ``M``) and (``N`, ``N``),
        respectively.
    compute_uv: bool, optional, default=``True``.
        Whether or not to compute `ur` and `Vt` in addition to `s`.

    Attributes
    ----------
    ur : :class:`~spectrochempy.dataset.nddataset.NDDataset`.
        `ur` contains the left unitary matrix.
        Its shape depends on `full_matrices`.
    s : `numpy.ndarray`.
        Vector of singular values
    Vt : :class:`~spectrochempy.dataset.dataset.NDDataset`.
        `Vt` contains a transpose matrix of the Loadings.
        Its shape depends on `full_matrices`
        """

        # check if we have the correct input
        if isinstance(X, NDDataset):
            data = X.data
        else:
            raise TypeError('A dataset of type NDDataset is  '
                               'expected as a source of data, but an object'
                               ' of type {} has been provided'.format(
                               type(X).__name__))

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
        U.title = 'left singular vectors of ' + X.name
        U.coordset = CoordSet(X.coordset[0],
                              Coord([i+1 for i in range(len(s))],
                                labels=['#%d' % (i+1) for i in range(len(s))],
                                title='Unitary vectors')
                              )
        U.description = 'left singular vectors of ' + X.name
        U.history = 'created by Svd \n'

        # Returns the loadings (Vt) as a NDDataset object
        Vt = NDDataset(Vt)
        Vt.title = 'Loadings (V.t) of ' + X.name
        Vt.coordset = CoordSet(Coord([i+1 for i in range(len(s))],
                                 labels=['#%d' % (i+1) for i in range(len(s))],
                                 title='Unitary vectors'),
                               X.coordset[1])
        Vt.description = (
            'Loadings obtained by singular value decomposition of ' + X.name)
        Vt.history = (str(Vt.modified) + ': created by Svd \n')

        self.U = U
        self.s = s
        self.Vt = Vt

    # ------------------------------------------------------------------------
    # special methods
    # ------------------------------------------------------------------------

    def __repr__(self):
        return '<svd: ur%s, s(%s), Vt%s>' % (
            self.U.shape, len(self.s), self.Vt.shape)

    # ------------------------------------------------------------------------
    #  Properties
    # ------------------------------------------------------------------------

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
