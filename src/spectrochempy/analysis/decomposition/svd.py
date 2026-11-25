# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Implements the Singular Value Decomposition (SVD) class."""

import warnings

import numpy as np
import traitlets as tr

from spectrochempy.analysis._base._analysisbase import AnalysisSourceMetadata
from spectrochempy.analysis._base._analysisbase import DecompositionAnalysis
from spectrochempy.analysis._base._analysisbase import NotFittedError
from spectrochempy.analysis._base._analysisbase import _check_n_components
from spectrochempy.analysis._base._analysisbase import _wrap_ndarray_output_to_nddataset
from spectrochempy.analysis._base._result import AnalysisResult
from spectrochempy.core.dataset.nddataset import NDDataset

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
    r"""
    Singular Value Decomposition (SVD).

    The SVD is commonly written as :math:`X = U \Sigma V^{T}`.

    This class has the attributes : U, s = diag(S) and VT=V :math:`^T`.

    Unlike `PCA`, the data are not centered: `transform` returns the scores
    :math:`U \Sigma` (:math:`X V` for new data), `components` the rows of
    :math:`V^{T}`, and `inverse_transform` the rank-:math:`k` reconstruction
    :math:`U_k \Sigma_k V_k^{T}` of the raw bilinear decomposition.

    If the dataset contains masked values, the corresponding ranges are
    ignored in the calculation.

    Parameters
    ----------
    log_level : any of [``"INFO"``, ``"DEBUG"``, ``"WARNING"``, ``"ERROR"``], optional, default: ``"WARNING"``
        The log level at startup. It can be changed later on using the
        `set_log_level` method or by changing the ``log_level`` attribute.
    warm_start : `bool`, optional, default: `False`
        When fitting repeatedly on the same dataset, but for multiple
        parameter values (such as to find the value maximizing performance),
        reuse the solution of the previous call to fit and add more components
        (if available) in a sequential manner.

        When `warm_start` is `True`, the existing fitted model attributes is used to
        initialize the new model in a subsequent call to `fit`.

    See Also
    --------
    fit : Fit the SVD model on X.
    transform : Apply dimensionality reduction.
    fit_transform : Fit the model and apply dimensionality reduction.

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
        default_value=True,
        help="Whether or not to compute U and VT in addition to s.",
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
        result = np.linalg.svd(X, full_matrices, compute_uv)
        # Sign correction to ensure deterministic output from SVD.
        # This doesn't work will full_matrices=True.
        if compute_uv:
            U, s, VT = result
            if not full_matrices:
                U, VT = _svd_flip(U, VT)
            _outfit = (U, s, VT)
        else:
            # compute_uv=False: np.linalg.svd returns only the singular
            # values vector s, not the (U, s, VT) tuple.  Normalise
            # _outfit to a consistent (U, s, VT) tuple so that all
            # downstream properties can safely index _outfit[0|1|2].
            _outfit = (None, result, None)
        # number of singular triplets (k = min(M, N)), also when full_matrices
        # is True and VT has more rows than singular values.
        self._n_components = int(np.size(_outfit[1]))
        return _outfit

    def _right_singular_vectors(self, method):
        if not self._fitted:
            raise NotFittedError()
        # The k = min(M, N) first rows of VT (VT has N rows when
        # full_matrices is True). Check what was fitted, not the current
        # compute_uv trait, which may have been changed since.
        if self._outfit[2] is None:
            raise ValueError(
                f"{method} requires the right singular vectors: "
                "fit the SVD with compute_uv=True."
            )
        return self._outfit[2][: self._n_components]

    def _transform(self, X):
        # Scores in the basis of the right singular vectors: X @ V.
        # SVD does not center or scale the data, so no preprocessing is
        # applied here; for the fitted X this equals U @ diag(s).
        VT = self._right_singular_vectors("transform")
        if X.shape[-1] != VT.shape[1]:
            raise ValueError(
                f"X has {X.shape[-1]} features, but SVD is expecting "
                f"{VT.shape[1]} features as input."
            )
        return X @ VT.T

    def _inverse_transform(self, X_transform):
        # Truncated reconstruction X_hat = T[:, :k] @ VT[:k], without any
        # mean added back since SVD does not center the data.
        VT = self._right_singular_vectors("inverse_transform")
        if X_transform.ndim == 1:
            X_transform = X_transform.reshape(-1, 1)
        k = min(X_transform.shape[1], VT.shape[0])
        return X_transform[:, :k] @ VT[:k]

    def _get_components(self):
        return self._right_singular_vectors("components")

    # ----------------------------------------------------------------------------------
    # special methods
    # ----------------------------------------------------------------------------------
    def __repr__(self):
        if self.compute_uv:
            U, s, VT = self._outfit
            return f"<svd: U{U.shape}, s({s.size}), VT{VT.shape}>"
        # _outfit is always a (U, s, VT) tuple after _fit normalisation
        s = self._outfit[1]
        return f"<svd: s({s.size}), U and VT not computed>"

    # ----------------------------------------------------------------------------------
    # Public method and properties
    # ----------------------------------------------------------------------------------
    def fit(self, X):
        """
        Fit the SVD model on X.

        Parameters
        ----------
        X : `NDDataset` or :term:`array-like` of shape (:term:`n_observations`, :term:`n_features`)
            Training data.

        Returns
        -------
        self
            The fitted SVD instance.

        See Also
        --------
        fit_transform : Fit the model and apply dimensionality reduction.
        transform : Apply dimensionality reduction.

        """
        return super().fit(X, Y=None)

    def transform(self, X=None, **kwargs):
        r"""
        Apply dimensionality reduction to `X`.

        Project `X` onto the right singular vectors :math:`V` of the fitted
        model: :math:`T = X V`. Unlike `PCA`, SVD does not center (nor scale)
        the data: the scores are those of the raw bilinear decomposition
        :math:`X = U \Sigma V^{T}`, so for the fitted dataset
        :math:`T = U \Sigma`.

        Parameters
        ----------
        X : `NDDataset` or :term:`array-like` of shape (:term:`n_observations`, :term:`n_features`), optional
            New data with the same features as the fitted dataset. If not
            provided, the dataset used in `fit` is transformed.
        **kwargs : keyword parameters, optional
            See Other Parameters.

        Returns
        -------
        `NDDataset`
            Scores with shape (:term:`n_observations`, :term:`n_components`).
            Like all latent factors (`PCA` scores, `MCRALS` profiles), the
            scores have no units.

        Other Parameters
        ----------------
        n_components : `int`, optional
            The number of components to keep (a positive integer). If `None`
            (default), all the :math:`\min(M, N)` components are kept.

        Raises
        ------
        ValueError
            If the model was fitted with ``compute_uv=False``, or if `X` does
            not have the number of (unmasked) features of the fitted dataset.

        See Also
        --------
        fit_transform : Fit the model and apply dimensionality reduction.

        Examples
        --------
        >>> dataset = scp.read('irdata/nh4y-activation.spg')
        >>> svd = scp.SVD().fit(dataset)
        >>> scores = svd.transform(n_components=2)
        >>> scores.shape
        (55, 2)

        """
        if not self._fitted:
            raise NotFittedError()
        training = self.X.copy()
        if X is not None:
            incoming = X if isinstance(X, NDDataset) else NDDataset(X)
            expected = self._analysis_full_columns(training.mask)
            actual = self._analysis_full_columns(incoming.mask)
            expected = (
                np.zeros(training.shape[-1], dtype=bool)
                if expected is None
                else expected
            )
            actual = (
                np.zeros(incoming.shape[-1], dtype=bool) if actual is None else actual
            )
            if incoming.shape[-1] != training.shape[-1]:
                raise ValueError(
                    f"X must have {training.shape[-1]} features as in the fitted dataset."
                )
            if not np.array_equal(actual, expected):
                raise ValueError(
                    "Fully masked feature columns must match those used during fit (features differ)."
                )
        try:
            return super().transform(X, **kwargs)
        finally:
            self._X = training

    def inverse_transform(self, X_transform=None, **kwargs):
        r"""
        Transform data back to its original space.

        Compute the rank-:math:`k` reconstruction
        :math:`\hat{X} = T_k V_k^{T}` from the scores :math:`T`. As SVD does not
        center the data (unlike `PCA`), no mean is added back: for the fitted
        dataset, :math:`\hat{X} = U_k \Sigma_k V_k^{T}`, which is `X` itself
        when all the components are kept.

        Parameters
        ----------
        X_transform : `NDDataset` or :term:`array-like` of shape (:term:`n_observations`, :term:`n_components`), optional
            Scores to transform back. If not provided, the scores of the
            dataset used in `fit` are computed first.
        **kwargs : keyword parameters, optional
            See Other Parameters.

        Returns
        -------
        `NDDataset`
            Reconstructed dataset with shape (:term:`n_observations`, :term:`n_features`).

        Other Parameters
        ----------------
        n_components : `int`, optional
            The number of components :math:`k` used for the reconstruction (a
            positive integer). If `None` (default), all the components of
            `X_transform` are used.

        Raises
        ------
        ValueError
            If the model was fitted with ``compute_uv=False``.

        See Also
        --------
        transform : Apply dimensionality reduction.

        Examples
        --------
        >>> dataset = scp.read('irdata/nh4y-activation.spg')
        >>> svd = scp.SVD().fit(dataset)
        >>> X_hat = svd.inverse_transform(n_components=2)
        >>> X_hat.shape
        (55, 5549)

        """
        if not self._fitted:
            raise NotFittedError
        # Fail early, naming this method, when U and VT were not computed
        # (otherwise the implicit transform() call below would raise first).
        self._right_singular_vectors("inverse_transform")

        # Unlike the base class, n_components is honoured in all cases: the
        # reconstruction uses the first n_components scores only.
        n_components = _check_n_components(kwargs.pop("n_components", None))
        if n_components is not None and n_components > self._n_components:
            warnings.warn(
                "The number of components required for reduction "
                "cannot be greater than the fitted model components : "
                f"{self._n_components}. We then use this latter value.",
                stacklevel=2,
            )
            n_components = self._n_components

        source_scores = X_transform if isinstance(X_transform, NDDataset) else None
        supplied = X_transform is not None
        if X_transform is None:
            kw = {} if n_components is None else {"n_components": n_components}
            X_transform = self.transform(**kw).data
        elif isinstance(X_transform, NDDataset):
            X_transform = X_transform.data
        else:
            X_transform = np.asarray(X_transform)
        if X_transform.ndim == 1:
            X_transform = X_transform.reshape(-1, 1)
        if n_components is not None:
            if n_components > X_transform.shape[1]:
                warnings.warn(
                    "The number of components required for reduction "
                    "cannot be greater than the X_transform size : "
                    f"{X_transform.shape[1]}. We then use this latter value.",
                    stacklevel=2,
                )
            X_transform = X_transform[:, :n_components]

        data = self._inverse_transform(X_transform)
        if source_scores is not None:
            training = self.X
            source = source_scores.copy()
            source._data = np.zeros((source_scores.shape[0], training.shape[-1]))
            source._coordset = None
            dims = [source_scores.dims[0], training.dims[-1]]
            source.dims = dims if dims[0] != dims[1] else list(NDDataset(data).dims)
            source.set_coordset(
                {
                    source.dims[0]: self._copy_axis_coord(source_scores, 0),
                    source.dims[-1]: self._copy_axis_coord(training, -1),
                }
            )
            rows = self._analysis_full_rows(source_scores.mask)
            cols = self._analysis_full_columns(training.mask)
            mask = np.zeros(source.shape, dtype=bool)
            if rows is not None:
                mask[rows, :] = True
            if cols is not None:
                mask[:, cols] = True
            source._mask = mask
            source.units = training.units
            restored = self._analysis_restore_full_geometry(
                NDDataset(data), AnalysisSourceMetadata(source)
            )
            result = self._wrap_reconstruction(restored.data, X=training)
            return self._analysis_copy_exact_geometry(
                result, AnalysisSourceMetadata(source)
            )
        if supplied:
            return self._wrap_reconstruction(data, X=X_transform)
        return self._wrap_reconstruction(data)

    @staticmethod
    def _copy_axis_coord(dataset, axis):
        if dataset.coordset is None:
            return None
        dim = dataset.dims[axis]
        dim = dataset.coordset.references.get(dim, dim)
        try:
            coord = dataset.coordset[dim]
        except KeyError:
            return None
        return coord.copy() if coord is not None else None

    def _wrap_reconstruction(self, data, X=None):
        result = NDDataset(data)
        context = {
            "role_id": "reconstruction",
            "meta_from": "_X",
            "direct_x": X,
            "direct_y": None,
            "direct_x_kind": "none"
            if X is None
            else "dataset"
            if isinstance(X, NDDataset)
            else "arraylike",
            "direct_y_kind": "none",
        }
        result = self._apply_analysis_output_geometry(result, **context)
        if isinstance(X, NDDataset):
            context.update(direct_x=None, direct_x_kind="none")
        return self._apply_analysis_output_metadata(result, **context)

    def fit_transform(self, X, Y=None, **kwargs):
        r"""
        Fit the SVD model on `X` and apply the dimensionality reduction on `X`.

        This is equivalent to ``fit(X).transform()``. Unlike `PCA`, the data are
        not centered: the scores are :math:`U \Sigma` of the raw decomposition
        :math:`X = U \Sigma V^{T}`.

        Parameters
        ----------
        X : `NDDataset` or :term:`array-like` of shape (:term:`n_observations`, :term:`n_features`)
            Training data.
        Y : any, optional
            Ignored; present for API consistency.
        **kwargs : keyword parameters, optional
            See Other Parameters.

        Returns
        -------
        `NDDataset`
            Scores :math:`U \Sigma` with shape (:term:`n_observations`, :term:`n_components`).

        Other Parameters
        ----------------
        n_components : `int`, optional
            The number of components to keep (a positive integer). If `None`
            (default), all the :math:`\min(M, N)` components are kept.

        See Also
        --------
        transform : Apply dimensionality reduction.

        Examples
        --------
        >>> dataset = scp.read('irdata/nh4y-activation.spg')
        >>> scores = scp.SVD().fit_transform(dataset, n_components=3)
        >>> scores.shape
        (55, 3)

        """
        self.fit(X)
        return self.transform(**kwargs)

    @property
    @_wrap_ndarray_output_to_nddataset(
        units=None,
        title="Singular values",
        typesingle="components",
        analysis_role="singular_values",
    )
    def singular_values(self):
        """Return a NDDataset containing singular values."""
        return self._outfit[1]

    sv = singular_values

    @property
    def explained_variance(self):
        """Return a NDDataset of the explained variance."""
        size = self.sv.size
        ev = self.sv**2 / (size - 1)
        return self._apply_analysis_output_metadata(
            ev,
            role_id="explained_variance",
            meta_from="_X",
            direct_x=None,
            direct_y=None,
            direct_x_kind="none",
            direct_y_kind="none",
        )

    ev = explained_variance

    @property
    def explained_variance_ratio(self):
        """Return Explained Variance per singular values."""
        ratio = self.ev * 100.0 / np.sum(self.ev)
        ratio.units = "percent"
        return self._apply_analysis_output_metadata(
            ratio,
            role_id="explained_variance_ratio",
            meta_from="_X",
            direct_x=None,
            direct_y=None,
            direct_x_kind="none",
            direct_y_kind="none",
        )

    ev_ratio = explained_variance_ratio

    @property
    def cumulative_explained_variance(self):
        """Return Cumulative Explained Variance."""
        ev_cum = np.cumsum(self.ev_ratio)
        ev_cum.units = "percent"
        return self._apply_analysis_output_metadata(
            ev_cum,
            role_id="cumulative_explained_variance",
            meta_from="_X",
            direct_x=None,
            direct_y=None,
            direct_x_kind="none",
            direct_y_kind="none",
        )

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
        return None

    @property
    def VT(self):
        """
        Return a transpose matrix of the Loadings.

        Its shape depends on `full_matrices`
        """
        if self.compute_uv:
            return self._outfit[2]
        return None

    @property
    def s(self):
        """Return Vector of singular values ."""
        return self._outfit[1]

    @property
    def result(self):
        """
        Return the SVD result object.

        Returns
        -------
        AnalysisResult
            Result object containing outputs (U, s, VT)
            and diagnostics (singular_values, explained_variance,
            explained_variance_ratio).
        """
        if not self._fitted:
            raise NotFittedError(
                "The fit method must be used before accessing the result",
            )

        # NOTE: a new AnalysisResult is created on every access.
        # Caching is deliberately deferred to keep the implementation
        # simple and aligned with the PCA result behaviour.

        diagnostics = {
            "singular_values": self.singular_values,
            "explained_variance": self.explained_variance,
            "explained_variance_ratio": self.explained_variance_ratio,
        }

        return AnalysisResult(
            estimator="SVD",
            parameters={
                "full_matrices": self.full_matrices,
                "compute_uv": self.compute_uv,
            },
            outputs={
                "U": self.U,
                "s": self.s,
                "VT": self.VT,
            },
            diagnostics=diagnostics,
        )
