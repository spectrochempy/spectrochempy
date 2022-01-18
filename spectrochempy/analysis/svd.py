# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory.
# ======================================================================================================================
"""
This module implements the Singular Value Decomposition (SVD) class.
"""

__all__ = ["SVD"]

__dataset_methods__ = []

from traitlets import HasTraits, Instance
import numpy as np

from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.dataset.ndarray import MASKED


def _svd_flip(U, VT, u_based_decision=True):
    """
    Sign correction to ensure deterministic output from SVD.
    Adjusts the columns of u and the rows of v such that the loadings in the
    columns in u that are largest in absolute value are always positive.

    Parameters
    ----------
    u_based_decision : boolean, (default=True)
        If True, use the columns of u as the basis for sign flipping.
        Otherwise, use the rows of v.

    Notes
    -----
    Copied and modified from scikit-learn.utils.extmath (BSD 3 Licence)
    """

    if u_based_decision:
        # columns of U, rows of VT
        max_abs_cols = np.argmax(np.abs(U), axis=0)
        signs = np.sign(U[max_abs_cols, range(U.shape[1])])
        U *= signs
        VT *= signs[:, np.newaxis]
    else:
        # rows of V, columns of U
        max_abs_rows = np.argmax(np.abs(VT), axis=1)
        signs = np.sign(VT[range(VT.shape[0]), max_abs_rows])
        U *= signs
        VT *= signs[:, np.newaxis]

    return U, VT


# ------------------------------------------------------------------
class SVD(HasTraits):
    """
    Performs a Singular Value Decomposition of a dataset.

    The SVD is commonly written as :math:`X = U \\Sigma V^{T}`.

    This class has the attributes : U, s = diag(S) and VT=V :math:`^T`.

    If the dataset contains masked values, the corresponding ranges are
    ignored in the calculation.
    """

    U = Instance(NDDataset, allow_none=True)
    """|NDDataset| - Contains the left unitary matrix. Its shape depends on `full_matrices`"""

    s = Instance(NDDataset)
    """|NDDataset| - Vector of singular values"""

    VT = Instance(NDDataset, allow_none=True)
    """|NDDataset| - Contains a transpose matrix of the Loadings. Its shape depends on `full_matrices`"""

    def __init__(self, dataset, full_matrices=False, compute_uv=True):
        """
        Parameters
        -----------
        dataset : |NDDataset| object
            The input dataset has shape (M, N). M is the number of
            observations (for examples a series of IR spectra) while N
            is the number of features (for example the wavenumbers measured
            in each IR spectrum).
        full_matrices : bool, optional, default=False.
            If False , U and VT have the shapes (M,  k) and
            (k, N), respectively, where k = min(M, N).
            Otherwise the shapes will be (M, M) and (N, N),
            respectively.
        compute_uv : bool, optional, default: True.
            Whether or not to compute U and VT in addition to s.

        Examples
        --------

        >>> dataset = scp.read('irdata/nh4y-activation.spg')
        >>> svd = SVD(dataset)
        >>> print(svd.ev.data)
        [1.185e+04      634 ... 0.001089 0.000975]
        >>> print(svd.ev_cum.data)
        [   94.54     99.6 ...      100      100]
        >>> print(svd.ev_ratio.data)
        [   94.54    5.059 ... 8.687e-06 7.779e-06]
        """
        super().__init__()

        self._compute_uv = compute_uv

        # check if we have the correct input
        # ----------------------------------

        X = dataset

        if isinstance(X, NDDataset):
            # As seen below, we cannot performs SVD on the masked array
            # so let's take the ndarray only.
            # Warning: we need to take the hidden values, to handle the
            # case of dim 1 axis. Of course, svd will work only for real
            # data, not complex or hypercomplex.
            data = X._data
            M, N = X._data.shape
            units = X.units

        else:
            raise TypeError(
                f"A dataset of type NDDataset is expected as a dataset of data, but an object of type"
                f" {type(X).__name__} has been provided"
            )

        # Retains only valid rows and columns
        # -----------------------------------

        # unfortunately, the present SVD implementation in linalg library
        # doesn't support numpy masked arrays as input. So we will have to
        # remove the masked values ourselves

        # the following however assumes that entire rows or columns are masked,
        # not only some individual data (if this is what you wanted, this
        # will fail)

        if np.any(X._mask):
            masked_columns = np.all(X._mask, axis=-2)
            masked_rows = np.all(X._mask, axis=-1)
        else:
            masked_columns = np.zeros(X._data.shape[-1], dtype=bool)
            masked_rows = np.zeros(X._data.shape[-2], dtype=bool)

        data = data[:, ~masked_columns]
        data = data[~masked_rows]

        # Performs the SVD
        # ----------------

        if data.size == 0 and np.product(data.shape[-2:]) == 0:
            raise np.linalg.LinAlgError(
                "Arrays cannot be empty. You may " "want to check the masked data. "
            )

        res = np.linalg.svd(data, full_matrices, compute_uv)
        if compute_uv:
            U, s, VT = res
        else:
            s = res

        # Returns the diagonal sigma matrix as a NDDataset object
        # -------------------------------------------------------

        s = NDDataset(s)
        s.title = "singular values of " + X.name
        s.name = "sigma"
        s.history = "Created by SVD \n"
        s.description = (
            "Vector of singular values obtained  by SVD " "decomposition of " + X.name
        )
        self.s = s

        if compute_uv:
            # Put back masked columns in  VT
            # ------------------------------
            # Note that it is very important to use here the ma version of zeros
            # array constructor
            KV = VT.shape[0]
            if np.any(masked_columns):
                Vtemp = np.ma.zeros((KV, N))  # note np.ma, not np.
                Vtemp[:, ~masked_columns] = VT
                Vtemp[:, masked_columns] = MASKED
                VT = Vtemp

            # Put back masked rows in U
            # -------------------------
            KU = U.shape[1]
            if np.any(masked_rows):
                Utemp = np.ma.zeros((M, KU))
                Utemp[~masked_rows] = U
                Utemp[masked_rows] = MASKED
                U = Utemp

            # Sign correction to ensure deterministic output from SVD.
            # This doesn't work will full_matrices=True.
            if not full_matrices:
                U, VT = _svd_flip(U, VT)

            # Returns U as a NDDataset object
            # --------------------------------
            U = NDDataset(U)
            U.name = "U"
            U.title = "left singular vectors of " + X.name
            U.set_coordset(
                x=Coord(labels=[f"#{i + 1}" for i in range(KU)], title="Components"),
                y=X.y,
            )
            U.description = "left singular vectors of " + X.name
            U.history = "Created by SVD \n"

            # Returns the loadings (VT) as a NDDataset object
            # ------------------------------------------------

            VT = NDDataset(VT)
            VT.name = "V.T"
            VT.title = "loadings (V.t) of " + X.name
            VT.set_coordset(
                x=X.x,
                y=Coord(labels=[f"#{i + 1}" for i in range(KV)], title="Components"),
            )
            VT.description = (
                "Loadings obtained by singular value decomposition of " + X.name
            )
            VT.history = str(VT.modified) + ": Created by SVD \n"
            # loadings keep the units of the original data
            VT.units = units

            self.U = U
            self.VT = VT
        else:
            self.U = None
            self.VT = None

    # ------------------------------------------------------------------------
    # special methods
    # ------------------------------------------------------------------------

    def __repr__(self):
        if self._compute_uv:
            return f"<svd: U{self.U.shape}, s({self.s.size}), VT{self.VT.shape}>"
        return f"<svd: s({self.s.size}), U, VT:not computed>"

    # ------------------------------------------------------------------------
    #  Properties
    # ------------------------------------------------------------------------

    @property
    def sv(self):
        """
        Singular values (|NDDataset|).
        """
        size = self.s.size
        sv = self.s.copy()
        sv.name = "sv"
        sv.title = "singular values"
        sv.set_coordset(
            Coord(None, labels=[f"#{(i + 1)}" for i in range(size)], title="Components")
        )
        return sv

    @property
    def ev(self):
        """
        Explained variance (|NDDataset|).
        """
        size = self.s.size
        ev = self.s ** 2 / (size - 1)
        ev.name = "ev"
        ev.title = "explained variance"
        ev.set_coordset(
            Coord(None, labels=[f"#{(i + 1)}" for i in range(size)], title="Components")
        )
        return ev

    @property
    def ev_cum(self):
        """
        Cumulative Explained Variance (|NDDataset|).
        """
        ev_cum = np.cumsum(self.ev_ratio)
        ev_cum.name = "ev_cum"
        ev_cum.title = "cumulative explained variance"
        ev_cum.units = "percent"
        return ev_cum

    @property
    def ev_ratio(self):
        """
        Explained Variance per singular values |NDDataset|).
        """
        ratio = self.ev * 100.0 / np.sum(self.ev)
        ratio.name = "ev_ratio"
        ratio.title = "explained variance"
        ratio.units = "percent"
        return ratio


# ======================================================================================================================
if __name__ == "__main__":
    pass
