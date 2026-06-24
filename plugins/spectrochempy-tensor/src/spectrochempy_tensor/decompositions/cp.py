# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Implementation of CP/PARAFAC decomposition using TensorLy."""

import numpy as np
import traitlets as tr

from spectrochempy.analysis._base._analysisbase import DecompositionAnalysis
from spectrochempy.analysis._base._analysisbase import NotFittedError
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.extern.traittypes import Array
from spectrochempy.utils.decorators import signature_has_configurable_traits
from spectrochempy.utils.optional import import_optional_dependency

__all__ = ["CP"]
__configurables__ = ["CP"]


# ======================================================================================
# Lazy TensorLy import helper
# ======================================================================================
def _import_tensorly():
    """Lazily import tensorly, raising a clear error if not installed."""
    return import_optional_dependency(
        "tensorly",
        extra="Install it with `pip install spectrochempy-tensor`.",
        errors="raise",
    )


# ======================================================================================
# Utility functions for core consistency
# ======================================================================================
def _kronecker_mat_ten(matrices, X):
    """Apply kronecker product of matrices along each mode of tensor X."""
    tl = _import_tensorly()
    for k in range(len(matrices)):
        M = matrices[k]
        X = tl.tenalg.mode_dot(X, M, k)
    return X


def _invert_sing(s):
    """Invert singular values, returning diagonal matrix."""
    return np.diag(1.0 / s)


# ======================================================================================
# class CP
# ======================================================================================
@signature_has_configurable_traits
class CP(DecompositionAnalysis):
    """
    CP/PARAFAC decomposition of 3D datasets using TensorLy.

    CP (Canonical Polyadic) decomposition, also known as PARAFAC, factorizes a 3D tensor
    into a sum of rank-1 tensors. By default, this implementation uses TensorLy's
    `parafac` function. When constraints or penalties are active (e.g., `non_negative`,
    `l1_reg`, `smoothness`, etc.), it automatically switches to TensorLy's
    `constrained_parafac` function with AO-ADMM optimization.

    Note: `fixed_modes` is supported by both `parafac` and `constrained_parafac`,
    so using `fixed_modes` alone does not trigger the constrained path.

    Only 3D datasets are supported. For 2D data, use PCA, NMF, or SVD instead.

    Parameters
    ----------
    n_components : int
        Number of components (rank) for the decomposition.
    n_iter_max : int, default=100
        Maximum number of outer iterations for the ALS algorithm.
    n_iter_max_inner : int, default=10
        Number of iterations for inner loop (ADMM optimization).
    init : {"random", "svd", "CPTensor"}, default="svd"
        Type of factor matrix initialization. If a CPTensor is passed, it is used directly.
    svd : {"numpy_svd", "truncated_svd", "randomized_svd"}, default="truncated_svd"
        Function to use for SVD computation during initialization.
    tol_outer : float, default=1e-8
        Relative reconstruction error tolerance for outer loop convergence.
    tol_inner : float, default=1e-6
        Absolute reconstruction error tolerance for inner loop (ADMM optimization).
    random_state : int or np.random.RandomState, optional
        Seed for random number generator or RandomState instance for reproducibility.
    verbose : int, default=0
        Level of verbosity for iteration logging.
    return_errors : bool, default=False
        Whether to store iteration errors after fitting. Errors are only
        available when using constrained_parafac (i.e., when constraints
        are active). For unconstrained parafac, errors will be None.
    non_negative : bool or dict, default=False
        If True, applies non-negative constraint to all modes. Can also be a dict
        specifying constraints per mode.
    l1_reg : float, list, or dict, optional
        L1 norm regularization parameter for sparsity. Applied to factors.
    l2_reg : float, list, or dict, optional
        L2 norm regularization parameter. Applied to factors.
    l2_square_reg : float, list, or dict, optional
        L2 square norm regularization parameter. Applied to factors.
    unimodality : bool or dict, optional
        If True, enforces unimodality constraint on all modes.
    normalize : bool or dict, optional
        If True, normalizes factors by dividing by maximum value.
    simplex : float, list, or dict, optional
        Projects factors onto the simplex with the given parameter.
    normalized_sparsity : float, list, or dict, optional
        Normalizes factors with L1 norm after hard thresholding.
    soft_sparsity : float, list, or dict, optional
        Imposes L1 norm bound on factor columns.
    smoothness : float, list, or dict, optional
        Optimizes factors by solving a banded system for smoothness.
    monotonicity : bool or dict, optional
        If True, projects factor columns to monotonically decreasing distribution.
    hard_sparsity : float, list, or dict, optional
        Applies hard thresholding with the given threshold.
    cvg_criterion : {"abs_rec_error", "rec_error"}, default="abs_rec_error"
        Stopping criterion for ALS. "abs_rec_error" uses absolute difference,
        "rec_error" uses relative difference.
    fixed_modes : list of int, optional
        Modes for which initial values are not modified during optimization.
        The last mode cannot be fixed.

    Attributes
    ----------
    A : NDDataset
        Factor matrix for mode 0 with shape (mode_0_size, n_components).
    B : NDDataset
        Factor matrix for mode 1 with shape (mode_1_size, n_components).
    C : NDDataset
        Factor matrix for mode 2 with shape (mode_2_size, n_components).
    loadings : tuple
        Tuple of factor matrices (A, B, C).
    weights : ndarray
        Weights from CP decomposition.
    errors : list or None
        Iteration errors during fitting. Available when TensorLy returns them
        (typically with constrained_parafac). Returns None if unavailable
        or if not fitted yet.
    SSE : float
        Sum of Squared Errors of the reconstruction.
    explained_variance : float
        Percentage of variance explained by the model.
    core_consistency : float
        CORCONDIA (Core Consistency) diagnostic value. Can be negative if overfactoring.

    Methods
    -------
    fit(X)
        Fit the CP model to a 3D dataset.
    fit_transform(X)
        Fit the model and return the reconstructed tensor.
    inverse_transform()
        Return the reconstructed tensor from fitted factors.

    Notes
    -----
    This method requires the optional dependency ``tensorly``.
    Install it with::

        pip install spectrochempy-tensor

    CP decomposition is sensitive to initialization and rank choice. The results
    may vary with different random seeds. Use ``random_state`` for reproducibility.

    Core consistency (CORCONDIA) helps assess model validity. Values close to 100%
    indicate a good fit, while negative values suggest overfactoring.

    See Also
    --------
    PCA : Principal Component Analysis for 2D data.
    SVD : Singular Value Decomposition for 2D data.

    Examples
    --------
    >>> import spectrochempy as scp
    >>> import numpy as np
    >>> X = np.random.rand(6, 8, 10)
    >>> ds = scp.NDDataset(X)
    >>> cp = scp.tensor.CP(n_components=2)
    >>> cp.fit(ds)
    >>> A, B, C = cp.loadings
    >>> Xr = cp.inverse_transform()
    """

    # ----------------------------------------------------------------------------------
    # Configuration parameters
    # ----------------------------------------------------------------------------------

    n_components = tr.Integer(
        allow_none=False,
        help="Number of components (this is the 'rank' parameter used in TensorLy).",
    ).tag(config=True)

    n_iter_max = tr.Integer(
        allow_none=True,
        default_value=100,
        help="Maximum number of outer iterations.",
    ).tag(config=True)

    n_iter_max_inner = tr.Integer(
        allow_none=True,
        default_value=10,
        help="Number of iterations for inner loop (ADMM optimization).",
    ).tag(config=True)

    init = tr.Enum(
        ["random", "svd", "CPTensor"],
        default_value="svd",
        help="""Type of factor matrix initialization.

        If a CPTensor is passed, this is directly used for initialization.
        See `initialize_factors`.""",
    ).tag(config=True)

    svd = tr.Enum(
        ["numpy_svd", "truncated_svd", "randomized_svd"],
        default_value="truncated_svd",
        help="Function to use to compute the SVD. Maps to tensorly SVD functions.",
    ).tag(config=True)

    tol_outer = tr.Float(
        default_value=1.0e-8,
        help="""Relative reconstruction error tolerance for outer loop.

        The algorithm is considered to have found a local minimum when the
        reconstruction error is less than `tol_outer`.""",
    ).tag(config=True)

    tol_inner = tr.Float(
        default_value=1.0e-6,
        help="""Absolute reconstruction error tolerance for factor update during inner loop,
        i.e., ADMM optimization.""",
    ).tag(config=True)

    random_state = tr.Union(
        (tr.Integer(), tr.Instance(np.random.RandomState)),
        allow_none=True,
        default_value=None,
        help="""If int, used to set the seed of the random number generator.
        If numpy.random.RandomState, used to initialize factor matrices
        with uniform distribution.""",
    ).tag(config=True)

    verbose = tr.Integer(
        default_value=0,
        help="Level of verbosity.",
    ).tag(config=True)

    return_errors = tr.Bool(
        default_value=False,
        help="Activate return of iteration errors.",
    ).tag(config=True)

    non_negative = tr.Union(
        (tr.Bool(), tr.Dict()),
        default_value=False,
        help="""This constraint clips negative values to '0'.

        If True, non-negative constraint is applied to all modes.""",
    ).tag(config=True)

    l1_reg = tr.Union(
        (tr.Float(), tr.List(), tr.Dict()),
        allow_none=True,
        default_value=None,
        help="""Penalizes the factor with the l1 norm using the input value as regularization parameter.""",
    ).tag(config=True)

    l2_reg = tr.Union(
        (tr.Float(), tr.List(), tr.Dict()),
        allow_none=True,
        default_value=None,
        help="""Penalizes the factor with the l2 norm using the input value as regularization parameter.""",
    ).tag(config=True)

    l2_square_reg = tr.Union(
        (tr.Float(), tr.List(), tr.Dict()),
        allow_none=True,
        default_value=None,
        help="""Penalizes the factor with the l2 square norm using the input value as regularization parameter.""",
    ).tag(config=True)

    unimodality = tr.Union(
        (tr.Bool(), tr.Dict()),
        allow_none=True,
        default_value=None,
        help="""If True, enforces unimodality constraint on all modes.
        Applied to each mode separately.""",
    ).tag(config=True)

    normalize = tr.Union(
        (tr.Bool(), tr.Dict()),
        allow_none=True,
        default_value=None,
        help="""This constraint divides all the values by maximum value of the input array.
        If True, normalize constraint is applied to all modes.""",
    ).tag(config=True)

    simplex = tr.Union(
        (tr.Float(), tr.List(), tr.Dict()),
        default_value=None,
        help="""Projects on the simplex with the given parameter.
        Applied to each column separately.""",
    ).tag(config=True)

    normalized_sparsity = tr.Union(
        (tr.Float(), tr.List(), tr.Dict()),
        default_value=None,
        help="Normalizes with the norm after hard thresholding.",
    ).tag(config=True)

    soft_sparsity = tr.Union(
        (tr.Float(), tr.List(), tr.Dict()),
        default_value=None,
        help="Impose that the columns of factors have L1 norm bounded by a user-defined threshold.",
    ).tag(config=True)

    smoothness = tr.Union(
        (tr.Float(), tr.List(), tr.Dict()),
        default_value=None,
        help="Optimizes the factors by solving a banded system.",
    ).tag(config=True)

    monotonicity = tr.Union(
        (tr.Bool(), tr.Dict()),
        default_value=None,
        help="""Projects columns to monotonically decreasing distribution.
        Applied to each column separately.
        If True, monotonicity constraint is applied to all modes.""",
    ).tag(config=True)

    hard_sparsity = tr.Union(
        (tr.Float(), tr.List(), tr.Dict()),
        default_value=None,
        help="Hard thresholding with the given threshold.",
    ).tag(config=True)

    cvg_criterion = tr.Enum(
        ["abs_rec_error", "rec_error"],
        default_value="abs_rec_error",
        help="""Stopping criterion for ALS, works if `tol` is not None.

        If 'rec_error', ALS stops at current iteration if
        ``(previous rec_error - current rec_error) < tol``.
        If 'abs_rec_error', ALS terminates when
        ``|previous rec_error - current rec_error| < tol``.""",
    ).tag(config=True)

    fixed_modes = tr.List(
        default_value=None,
        help="""A list of modes for which the initial value is not modified.
        The last mode cannot be fixed due to error computation.""",
    ).tag(config=True)

    # ----------------------------------------------------------------------------------
    # Runtime Parameters
    # ----------------------------------------------------------------------------------
    _weights = Array(help="Weights from CP decomposition.")
    _loadings = tr.List(help="List of factor matrices from CP decomposition.")
    _errors = tr.List(help="Iteration errors if return_errors=True.")
    _A = tr.Instance(NDDataset, help="Factor A (mode 0 loadings).")
    _B = tr.Instance(NDDataset, help="Factor B (mode 1 loadings).")
    _C = tr.Instance(NDDataset, help="Factor C (mode 2 loadings).")
    _SSE = tr.Float(allow_none=True, help="Sum of Squared Errors.")
    _explained_variance = tr.Float(
        allow_none=True, help="Explained variance percentage."
    )

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
    # Private validation methods
    # ----------------------------------------------------------------------------------
    @tr.observe("_X")
    def _preprocess_as_X_changed(self, change):
        """Validate input data and preprocess."""
        X = change.new

        # Check for 3D input
        if X.ndim != 3:
            raise ValueError(
                f"CP decomposition requires 3D input. Got {X.ndim}D input. "
                "For 2D data, use PCA, NMF, or SVD instead."
            )

        # Validate n_components
        if self.n_components is None:
            raise ValueError("n_components must be specified for CP decomposition.")
        if self.n_components <= 0:
            raise ValueError(f"n_components must be positive. Got {self.n_components}.")
        if self.n_components > min(X.shape):
            raise ValueError(
                f"n_components ({self.n_components}) cannot exceed "
                f"min(data shape) ({min(X.shape)})."
            )

        # Validate fixed_modes
        if self.fixed_modes:
            for mode in self.fixed_modes:
                if mode < 0 or mode >= len(X.shape):
                    raise ValueError(f"fixed_modes contains invalid mode {mode}.")
                if mode == len(X.shape) - 1:
                    raise ValueError(
                        "Last mode cannot be fixed (needed for error computation)."
                    )

        # Preprocess: get numpy array (masked rows/cols removed by base class)
        self._X_preprocessed = X.data

    # ----------------------------------------------------------------------------------
    # Constraint detection
    # ----------------------------------------------------------------------------------
    def _has_constraints(self):
        """Return True if AO-ADMM constraints/penalties are active."""
        return (
            self.non_negative is not False
            or self.l1_reg is not None
            or self.l2_reg is not None
            or self.l2_square_reg is not None
            or self.unimodality is not None
            or self.normalize is not None
            or self.simplex is not None
            or self.normalized_sparsity is not None
            or self.soft_sparsity is not None
            or self.smoothness is not None
            or self.monotonicity is not None
            or self.hard_sparsity is not None
        )

    # ----------------------------------------------------------------------------------
    # Private methods (overloading abstract classes)
    # ----------------------------------------------------------------------------------
    def _fit(self, X, Y=None):
        """Fit the CP model to X (np.ndarray)."""
        tl = _import_tensorly()

        if self._has_constraints():
            result = tl.decomposition.constrained_parafac(
                X,
                rank=self.n_components,
                n_iter_max=self.n_iter_max,
                n_iter_max_inner=self.n_iter_max_inner,
                init=self.init,
                svd=self.svd,
                tol_outer=self.tol_outer,
                tol_inner=self.tol_inner,
                random_state=self.random_state,
                verbose=self.verbose,
                cvg_criterion=self.cvg_criterion,
                fixed_modes=self.fixed_modes if self.fixed_modes else [],
                non_negative=self.non_negative,
                l1_reg=self.l1_reg,
                l2_reg=self.l2_reg,
                l2_square_reg=self.l2_square_reg,
                unimodality=self.unimodality,
                normalize=self.normalize,
                simplex=self.simplex,
                normalized_sparsity=self.normalized_sparsity,
                soft_sparsity=self.soft_sparsity,
                smoothness=self.smoothness,
                monotonicity=self.monotonicity,
                hard_sparsity=self.hard_sparsity,
                return_errors=True,
            )
        else:
            result = tl.decomposition.parafac(
                X,
                rank=self.n_components,
                n_iter_max=self.n_iter_max,
                init=self.init,
                svd=self.svd,
                tol=self.tol_outer,
                random_state=self.random_state,
                verbose=self.verbose,
                fixed_modes=self.fixed_modes if self.fixed_modes else [],
                return_errors=True,
            )

        # Unpack result robustly: may be (cp_tensor, errors) or cp_tensor
        if isinstance(result, tuple):
            cp_tensor, errors = result
        else:
            cp_tensor = result
            errors = None
        self._weights = cp_tensor.weights
        self._loadings = cp_tensor.factors  # List of factor matrices
        self._errors = errors if errors else None

        # Create NDDataset objects for factors
        self._get_A()
        self._get_B()
        self._get_C()

        # Reset cached SSE and explained variance
        self._SSE = None
        self._explained_variance = None

        return result

    def _transform(self, X):
        """CP does not support transform for new data."""
        raise NotImplementedError(
            "CP decomposition does not support transform() for new data. "
            "Use fit() to decompose data, then access A, B, C properties."
        )

    def _get_components(self):
        """Return components (factor B following PCA convention)."""
        if not self._fitted:
            raise NotFittedError("Model must be fitted first.")
        return self._loadings[1]  # Return B factor

    # ----------------------------------------------------------------------------------
    # Factor creation methods
    # ----------------------------------------------------------------------------------
    def _get_A(self):
        """Create NDDataset for factor A (mode 0)."""
        self._A = NDDataset(self._loadings[0])
        # Use 'a' as the components dimension (valid SpectroChemPy dim name)
        self._A.dims = [self._X.dims[0], "a"]
        # Create components coordinate with labels
        comp_coord = Coord(
            np.arange(self.n_components),
            title="components",
            labels=[f"#{i}" for i in range(self.n_components)],
        )
        self._A.set_coordset(
            {
                self._X.dims[0]: self._X.coordset[self._X.dims[0]],
                "a": comp_coord,
            }
        )
        self._A.name = f"Factor A of CP({self._X.name})"
        self._A.history = "Created by SpectroChemPy CP"
        return self._A

    def _get_B(self):
        """Create NDDataset for factor B (mode 1)."""
        self._B = NDDataset(self._loadings[1])
        self._B.dims = [self._X.dims[1], "a"]
        # Create components coordinate with labels
        comp_coord = Coord(
            np.arange(self.n_components),
            title="components",
            labels=[f"#{i}" for i in range(self.n_components)],
        )
        self._B.set_coordset(
            {
                self._X.dims[1]: self._X.coordset[self._X.dims[1]],
                "a": comp_coord,
            }
        )
        self._B.name = f"Factor B of CP({self._X.name})"
        self._B.history = "Created by SpectroChemPy CP"
        return self._B

    def _get_C(self):
        """Create NDDataset for factor C (mode 2)."""
        if len(self._loadings) > 2:
            self._C = NDDataset(self._loadings[2])
            self._C.dims = [self._X.dims[2], "a"]
            # Create components coordinate with labels
            comp_coord = Coord(
                np.arange(self.n_components),
                title="components",
                labels=[f"#{i}" for i in range(self.n_components)],
            )
            self._C.set_coordset(
                {
                    self._X.dims[2]: self._X.coordset[self._X.dims[2]],
                    "a": comp_coord,
                }
            )
            self._C.name = f"Factor C of CP({self._X.name})"
            self._C.history = "Created by SpectroChemPy CP"
            return self._C
        self._C = None
        return None

    # ----------------------------------------------------------------------------------
    # Core consistency computation
    # ----------------------------------------------------------------------------------
    def _compute_core_consistency(self):
        """
        Compute CORCONDIA (Core Consistency) diagnostic.

        Adapted from https://github.com/willshiao/pycorcondia, MIT License.
        """
        _import_tensorly()

        k = self.n_components
        A, B, C = self._loadings

        Ua, Sa, Va = np.linalg.svd(A, full_matrices=False)
        Ub, Sb, Vb = np.linalg.svd(B, full_matrices=False)
        Uc, Sc, Vc = np.linalg.svd(C, full_matrices=False)

        inverted = [_invert_sing(x) for x in (Sa, Sb, Sc)]

        part1 = _kronecker_mat_ten([Ua.T, Ub.T, Uc.T], self._X.data)
        part2 = _kronecker_mat_ten(inverted, part1)
        G = _kronecker_mat_ten([Va.T, Vb.T, Vc.T], part2)

        for i in range(k):
            G[:, :, i] = G[:, :, i] / G[i, i, i]

        T = np.zeros((k, k, k))
        for i in range(k):
            T[i, i, i] = 1

        return 100.0 * (1 - ((G - T) ** 2).sum() / float(k))

    # ----------------------------------------------------------------------------------
    # Public methods
    # ----------------------------------------------------------------------------------
    def fit(self, X):
        """
        Fit the CP model on X.

        Parameters
        ----------
        X : NDDataset
            3D dataset to decompose.

        Returns
        -------
        self : CP
            The fitted CP instance.
        """
        return super().fit(X, Y=None)

    def fit_transform(self, X, **kwargs):
        """
        Fit the CP model on X and return the factors.

        Parameters
        ----------
        X : NDDataset
            3D dataset to decompose.
        **kwargs
            Additional keyword arguments passed to fit.

        Returns
        -------
        tuple of NDDataset
            The factor matrices (A, B, C).
        """
        self.fit(X)
        return self.A, self.B, self.C

    def inverse_transform(self, X_transform=None, **kwargs):
        """
        Transform data back to its original space.

        Reconstruct the original tensor from the CP factors.

        Parameters
        ----------
        X_transform : None
            Ignored. Present for API compatibility.
        **kwargs
            Additional keyword arguments (ignored).

        Returns
        -------
        NDDataset
            Reconstructed dataset with shape matching the input.
        """
        if not self._fitted:
            raise NotFittedError(
                "The fit method must be used before using inverse_transform()"
            )

        tl = _import_tensorly()
        X_hat = NDDataset(tl.cp_to_tensor((self._weights, self._loadings)))
        X_hat.dims = self._X.dims
        X_hat.coordset = self._X.coordset
        X_hat.units = self._X.units
        X_hat.title = self._X.title
        X_hat.name = f"rank-{self.n_components} CP inverse transform of {self._X.name}"
        X_hat.history = "Created by SpectroChemPy CP"

        return X_hat

    @property
    def result(self):
        """
        Return the CP analysis result.

        Returns
        -------
        AnalysisResult
            Result containing the primary CP outputs and fit diagnostics.

        Raises
        ------
        NotFittedError
            If the estimator has not been fitted yet.
        """
        if not self._fitted:
            raise NotFittedError(
                "The fit method must be used before accessing the result"
            )

        from spectrochempy.analysis._base._result import AnalysisResult  # noqa: PLC0415

        return AnalysisResult(
            estimator="CP",
            parameters={
                "n_components": self.n_components,
                "n_iter_max": self.n_iter_max,
                "n_iter_max_inner": self.n_iter_max_inner,
                "init": self.init,
                "svd": self.svd,
                "tol_outer": self.tol_outer,
                "tol_inner": self.tol_inner,
                "random_state": self.random_state,
                "verbose": self.verbose,
                "return_errors": self.return_errors,
                "non_negative": self.non_negative,
                "l1_reg": self.l1_reg,
                "l2_reg": self.l2_reg,
                "l2_square_reg": self.l2_square_reg,
                "unimodality": self.unimodality,
                "normalize": self.normalize,
                "simplex": self.simplex,
                "normalized_sparsity": self.normalized_sparsity,
                "soft_sparsity": self.soft_sparsity,
                "smoothness": self.smoothness,
                "monotonicity": self.monotonicity,
                "hard_sparsity": self.hard_sparsity,
                "cvg_criterion": self.cvg_criterion,
                "fixed_modes": self.fixed_modes,
            },
            outputs={
                "factors": self.loadings,
                "weights": self.weights,
            },
            diagnostics={
                "errors": self.errors,
                "SSE": self.SSE,
                "explained_variance": self.explained_variance,
                "core_consistency": self.core_consistency,
            },
        )

    # ----------------------------------------------------------------------------------
    # Public properties
    # ----------------------------------------------------------------------------------
    @property
    def A(self):
        """Return factor matrix A (mode 0 loadings)."""
        if not self._fitted:
            raise NotFittedError(
                "The fit method must be used to get the loading matrix A"
            )
        return self._A

    @property
    def B(self):
        """Return factor matrix B (mode 1 loadings)."""
        if not self._fitted:
            raise NotFittedError(
                "The fit method must be used to get the loading matrix B"
            )
        return self._B

    @property
    def C(self):
        """Return factor matrix C (mode 2 loadings)."""
        if not self._fitted:
            raise NotFittedError(
                "The fit method must be used to get the loading matrix C"
            )
        if self._C is None:
            raise AttributeError("The loading matrix C is not available")
        return self._C

    @property
    def components(self):
        """
        Return factor B (mode 1) as components, following PCA convention.

        Returns
        -------
        NDDataset
            Factor B with shape (mode_1_size, n_components).
        """
        if not self._fitted:
            raise NotFittedError("Model must be fitted first.")
        return self.B  # B is mode 1 factor

    @property
    def loadings(self):
        """Return tuple of factor matrices (A, B, C)."""
        return (self.A, self.B, self.C)

    @property
    def weights(self):
        """Return the weights from CP decomposition."""
        if not self._fitted:
            raise NotFittedError("The fit method must be used to get weights")
        return self._weights

    @property
    def errors(self):
        """Return iteration errors if return_errors was True."""
        if not self._fitted:
            raise NotFittedError("The fit method must be used to get errors")
        return self._errors

    @property
    def SSE(self):
        """Return Sum of Squared Errors."""
        if not self._fitted:
            raise NotFittedError("The fit method must be used first to get SSE")
        if not hasattr(self, "_SSE") or self._SSE is None:
            X = self._X_preprocessed
            tl = _import_tensorly()
            X_hat = tl.cp_to_tensor((self._weights, self._loadings))
            self._SSE = float(np.sum((X - X_hat) ** 2))
        return self._SSE

    @property
    def explained_variance(self):
        """Return explained variance percentage."""
        if not self._fitted:
            raise NotFittedError(
                "The fit method must be used first to get explained variance"
            )
        if not hasattr(self, "_explained_variance") or self._explained_variance is None:
            X = self._X_preprocessed
            total_variance = float(np.sum(X**2))
            self._explained_variance = 100.0 * (1.0 - self.SSE / total_variance)
        return self._explained_variance

    @property
    def core_consistency(self):
        """
        Return CORCONDIA (Core Consistency) diagnostic.

        Returns
        -------
        float
            Core consistency value. Can be negative if overfactoring occurred.
        """
        if not self._fitted:
            raise NotFittedError(
                "The fit method must be used first to get Core Consistency"
            )
        if not hasattr(self, "_core_consistency"):
            self._core_consistency = self._compute_core_consistency()
        return self._core_consistency
