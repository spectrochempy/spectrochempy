# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================


__all__ = ['Svd']

# ----------------------------------------------------------------------------
# third party imports
# ----------------------------------------------------------------------------

from traitlets import HasTraits, Instance
import numpy as np

# ----------------------------------------------------------------------------
# local imports
# ----------------------------------------------------------------------------

from spectrochempy.dataset.ndcoords import Coord
from spectrochempy.dataset.nddataset import NDDataset, CoordSet
from spectrochempy.dataset.ndarray import masked, nomask

# ----------------------------------------------------------------------------
class Svd(HasTraits):
    """Performs a Singular Value Decomposition of a dataset.
    
    The SVD is commonly written as :math:'X = ur \Sigma V^{T}'. This class \
    has the attributes: ur, s = diag(S) and Vt=V.T.

    If the dataset contains masked values, the corresponding ranges are
    ignored in the calculation.

    Examples
    ---------
    >>> source = NDDataset.load('mydataset.scp')
    >>> svd = Svd(source)
    >>> print(svd.ev.data)
    [11848.225  633.980 ...,    0.001    0.001]
    >>> print(svd.ev_cum.data)
    [11848.225 12482.204 ..., 12532.584 12532.585]
    >>> print(svd.ev_ratio.data)
    [  94.539    5.059 ...,    0.000    0.000]

    """

    U = Instance(NDDataset)
    s = Instance(NDDataset)
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

        # Retains valid columns
        # ---------------------

        # unfortunately the present SVD implementation in linalg librairy
        # doesn't support numpy masked array. So we will have to remove the
        # mask ourselves

        masked_columns = np.any(X.mask, axis=0)
        data = data[:, ~ masked_columns]

        # Performs the SVD
        # ----------------

        U, s, Vt = np.linalg.svd(data, full_matrices, compute_uv)

        # Put back masked columns in Vt
        # -----------------------------

        if np.any(masked_columns):
            temp = np.zeros((s.shape[0], X.shape[1]))
            temp[:, ~ masked_columns ] = Vt
            temp[:, masked_columns] = masked
            Vt = temp

        # Returns U as a NDDataset object
        # --------------------------------

        U = NDDataset(U)
        U.name = 'U'
        U.title = 'left singular vectors of ' + X.name
        U.coordset = CoordSet(X.coordset[0],
                              Coord([i+1 for i in range(len(s))],
                                labels=['#%d' % (i+1) for i in range(len(s))],
                                title='Unitary vectors')
                              )
        U.description = 'left singular vectors of ' + X.name
        U.history = 'created by Svd \n'

        # Returns the loadings (Vt) as a NDDataset object
        # ------------------------------------------------

        Vt = NDDataset(Vt)
        Vt.name = 'V.T'
        Vt.title = 'Loadings (V.t) of ' + X.name
        Vt.coordset = CoordSet(Coord([i+1 for i in range(len(s))],
                                 labels=['#%d' % (i+1) for i in range(len(s))],
                                 title='Unitary vectors'),
                               X.coordset[1])
        Vt.description = (
            'Loadings obtained by singular value decomposition of ' + X.name)
        Vt.history = (str(Vt.modified) + ': created by Svd \n')

        # Returns the diagonal sigma matrix as a NDDataset object
        # -------------------------------------------------------

        s = NDDataset(s)
        s.title='Singular values of ' + X.name
        s.name = 'sigma'
        s.history = 'created by Svd \n'
        s.description = (
            'Vector of singular values obtained  by SVD '
            'decomposition of ' + X.name)

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
        """`NDDataset`,  eigenvalues of the covariance matrix """
        ev = (self.s ** 2) / (self.s.size - 1)
        ev.name = 'ev'
        ev.title = 'Eigenvalues'
        return ev

    @property
    def ev_cum(self):
        """`NDDataset`,  Cumulative Explained Variance """
        ev_cum = np.cumsum(self.ev)
        ev_cum.name = 'ev_cum'
        ev_cum.title = 'Cumulative variance'
        return ev_cum

    @property
    def ev_ratio(self):
        """`numpy.ndarray`,  Explained Variance per singular values """
        ratio = self.ev *100. / np.sum(self.ev)
        ratio.name = 'ev_ratio'
        ratio.title = 'Explained variance'
        ratio.units = 'percent'
        return ratio


if __name__ == '__main__':

    from tests.conftest import IR_source_2D
    from spectrochempy.api import *

    source = IR_source_2D()

    print(source)

    source[:, 1240.0:920.0] = masked  # do not forget to use float in slicing
    ax = source.plot_stack()

    svd = Svd(source)

    print()
    print((svd.U))
    print((svd.Vt))
    print((svd.s))
    print((svd.ev))
    print((svd.ev_cum))
    print((svd.ev_ratio))

    svd.Vt[:6].plot_stack()

    svd.ev_ratio.plot_scatter(color='red', lines=True, xlim=(-0.1,9.5))
    show()

