# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
This module implements the Singular Value Decomposition (SVD) class.
"""
import numpy as np
import traitlets as tr

from spectrochempy.analysis._base import (
    DecompositionAnalysis,
    _wrap_ndarray_output_to_nddataset,
)
from spectrochempy.utils.docstrings import _docstring

__all__ = ["SVD"]
__configurables__ = ["SVD"]


# ======================================================================================
# Utilities
# ======================================================================================
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


# ======================================================================================
# class PCA
# ======================================================================================
class SVD(DecompositionAnalysis):

    _docstring.delete_params("DecompositionAnalysis.see_also", "SVD")

    __doc__ = _docstring.dedent(
        r"""
    Singular Value Decomposition (SVD).

    The SVD is commonly written as :math:`X = U \Sigma V^{T}`\ .

    This class has the attributes : U, s = diag(S) and VT=V :math:`^T`\ .

    If the dataset contains masked values, the corresponding ranges are
    ignored in the calculation.

    Parameters
    ----------
    %(AnalysisConfigurable.parameters)s

    See Also
    --------
    %(DecompositionAnalysis.see_also.no_SVD)s

    Examples
    --------
    >>> dataset = scp.read('irdata/nh4y-activation.spg')
    >>> svd = scp.SVD()
    >>> svd.fit(dataset)
    <svd: U(55, 55), s(55), VT(55, 5549)>
    >>> print(svd.ev.data)
    [1.185e+04      634 ... 0.001089 0.000975]
    >>> print(svd.ev_cum.data)
    [   94.54     99.6 ...      100      100]
    >>> print(svd.ev_ratio.data)
    [   94.54    5.059 ... 8.687e-06 7.779e-06]
    """
    )

    # ----------------------------------------------------------------------------------
    # Configuration parameters
    # ----------------------------------------------------------------------------------
    full_matrices = tr.Bool(
        default_value=False,
        help="If False , U and VT have the shapes (M,  k) and (k, N), respectively, "
        "where k = min(M, N). Otherwise the shapes will be (M, M) and (N, N), "
        "respectively.",
    ).tag(config=True)
    compute_uv = tr.Bool(
        default_value=True, help="Whether or not to compute U and VT in addition to s."
    ).tag(config=True)

    # ----------------------------------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------------------------------
    def __init__(
        self,
        *,
        log_level="WARNING",
        warm_start=False,
        **kwargs,
    ):

        # call the super class for initialisation of the configuration parameters
        # to do before anything else!
        super().__init__(
            log_level=log_level,
            warm_start=warm_start,
            **kwargs,
        )

    # ----------------------------------------------------------------------------------
    # Private methods (overloading abstract classes)
    # ----------------------------------------------------------------------------------
    def _fit(self, X, Y=None):
        # this method is called by the abstract class fit.
        # Input X is a np.ndarray
        # Y is ignored in this model
        full_matrices = self.full_matrices
        compute_uv = self.compute_uv
        _outfit = np.linalg.svd(X, full_matrices, compute_uv)
        # Sign correction to ensure deterministic output from SVD.
        # This doesn't work will full_matrices=True.
        if compute_uv and not full_matrices:
            U, s, VT = _outfit
            U, VT = _svd_flip(U, VT)
            _outfit = U, s, VT
        return _outfit

    # ----------------------------------------------------------------------------------
    # special methods
    # ----------------------------------------------------------------------------------
    def __repr__(self):
        if self.compute_uv:
            U, s, VT = self._outfit
            return f"<svd: U{U.shape}, s({s.size}), VT{VT.shape}>"
        s = self._outfit
        return f"<svd: s({s.size}), U and VT not computed>"

    # ----------------------------------------------------------------------------------
    # Public method and properties
    # ----------------------------------------------------------------------------------
    _docstring.keep_params("analysis_fit.parameters", "X")

    @_docstring.dedent
    def fit(self, X):
        """
        Fit the SVD model on X.

        Parameters
        ----------
        %(analysis_fit.parameters.X)s

        Returns
        -------
        %(analysis_fit.returns)s

        See Also
        --------
        %(analysis_fit.see_also)s
        """
        return super().fit(X, Y=None)

    @property
    @_wrap_ndarray_output_to_nddataset(
        units=None, title="Singular values", typesingle="components"
    )
    def singular_values(self):
        """
        Return a NDDataset containing singular values.
        """
        s = self._outfit[1]
        return s

    sv = singular_values

    @property
    def explained_variance(self):
        """
        Return a NDDataset of the explained variance.
        """
        size = self.sv.size
        ev = self.sv**2 / (size - 1)
        ev.name = "ev"
        ev.title = "explained variance"
        return ev

    ev = explained_variance

    @property
    def explained_variance_ratio(self):
        """
        Return Explained Variance per singular values.
        """
        ratio = self.ev * 100.0 / np.sum(self.ev)
        ratio.name = "ev_ratio"
        ratio.title = "explained variance ratio"
        ratio.units = "percent"
        return ratio

    ev_ratio = explained_variance_ratio

    @property
    def cumulative_explained_variance(self):
        """
        Return Cumulative Explained Variance.
        """
        ev_cum = np.cumsum(self.ev_ratio)
        ev_cum.name = "ev_cum"
        ev_cum.title = "cumulative explained variance"
        ev_cum.units = "percent"
        return ev_cum

    ev_cum = cumulative_explained_variance

    # TODO: return masked array for U,s,VT
    @property
    def U(self):
        """
        Return the left unitary matrix.

        Its shape depends on `full_matrices` .
        """
        if self.compute_uv:
            return self._outfit[0]

    @property
    def VT(self):
        """
        Return a transpose matrix of the Loadings.

        Its shape depends on `full_matrices`
        """
        if self.compute_uv:
            return self._outfit[2]

    @property
    def s(self):
        """
        Return Vector of singular values .
        """
        return self._outfit[1]
