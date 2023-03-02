# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Implementation of fast ICA algorithm (using scikit-learn library)
"""
import traitlets as tr
from numpy.random import RandomState
from sklearn import decomposition

from spectrochempy.analysis._analysisutils import NotFittedError
from spectrochempy.analysis.abstractanalysis import DecompositionAnalysis
from spectrochempy.extern.traittypes import Array

__all__ = ["FastICA"]
__configurables__ = ["FastICA"]


# ======================================================================================
# class FastICA
# ======================================================================================
class FastICA(DecompositionAnalysis):
    """FastICA: a fast algorithm for Independent Component Analysis.

    Use scikit-learn FastICA

    Parameters
    ----------
    log_level : ["INFO", "DEBUG", "WARNING", "ERROR"], optional, default:"WARNING"
        The log level at startup
    config : Config object, optional
        By default the configuration is determined by the MCRALS.py
        file in the configuration directory. A traitlets.config.Config() object can
        eventually be used here.
    warm_start : bool, optional, default: false
        When fitting with SIMPLISMA repeatedly on the same dataset, but for multiple
        parameter values (such as to find the value maximizing performance),
        it may be possible to reuse previous model learned from the previous parameter
        value, saving time.
        When warm_start is true, the existing fitted model attributes is used to
        initialize the new model in a subsequent call to fit.
    **kwargs
        Optional configuration  parameters.

    See Also
    --------
    PCA : Perform Principal Components Analysis.
    NNMF : Perform a Non-Negative Matrix Factorization.
    EFA : Perform an Evolving Factor Analysis (forward and reverse).
    MCRALS : Perform MCR-ALS of a dataset knowing the initial C or St matrix.
    SVD : Perform a Singular Value Decomposition.
    SIMPLISMA : SIMPLe to use Interactive Self-modeling Mixture Analysis.
    """

    name = tr.Unicode("FastICA")
    description = tr.Unicode("Scikit-learn FastICA model")

    # ----------------------------------------------------------------------------------
    # Runtime Parameters,
    # only those specific to PCA, the other being defined in AnalysisConfigurable.
    # ----------------------------------------------------------------------------------
    # define here only the variable that you use in fit or transform functions
    _ica = tr.Instance(
        decomposition.FastICA,
        help="The instance of sklearn.decomposition.FastICA used in this model",
    )

    # ----------------------------------------------------------------------------------
    # Configuration parameters
    # ----------------------------------------------------------------------------------
    used_components = tr.Int(
        default_value=None,
        allow_none=True,
        help="Number of components to use. If None is passed, all are used.",
    ).tag(config=True)

    algorithm = tr.Enum(
        ["parallel", "deflation"],
        default_value="parallel",
        help="Specify which algorithm to use for FastICA.",
    ).tag(config=True)

    whiten = tr.Enum(
        ["arbitrary-variance", "unit-variance"],
        default_value="arbitrary-variance",
        help="""Specify the whitening strategy to use.
- If 'arbitrary-variance' (default), a whitening with variance
  arbitrary is used.
- If 'unit-variance', the whitening matrix is rescaled to ensure that
  each recovered source has unit variance.
- If False, the data is already considered to be whitened, and no
  whitening is performed.""",
    ).tag(config=True)
    fun = tr.Union(
        (tr.Enum(["logcosh", "exp", "cube"]), tr.Callable()),
        default_value="logcosh",
        help="""The functional form of the G function used in the
approximation to neg-entropy. Could be either 'logcosh', 'exp',
or 'cube'.
You can also provide your own function. It should return a tuple
containing the value of the function, and of its derivative, in the
point. The derivative should be averaged along its last dimension.
Example::

 def my_g(x):
     return x ** 3, (3 * x ** 2).mean(axis=-1)
""",
    ).tag(config=True)

    fun_args = tr.Dict(
        default_value=None,
        allow_none=True,
        help="""Arguments to send to the functional form.
If empty or None and if fun='logcosh', fun_args will take value {'alpha' : 1.0}.""",
    ).tag(config=True)

    max_iter = tr.Int(
        default_value=200, help="Maximum number of iterations during fit."
    ).tag(config=True)

    tol = tr.Float(
        default_value=1e-4,
        help="A positive scalar giving the tolerance at which the un-mixing matrix "
        "is considered to have converged.",
    ).tag(config=True)

    w_init = Array(
        default_value=None,
        allow_none=True,
        help="array-like of shape (n_components, n_components): "
        "Initial un-mixing array. If `w_init=None`, then an array of values "
        "drawn from a normal distribution is used.",
    ).tag(config=True)

    whiten_solver = tr.Enum(
        ["eigh", "svd"],
        default_value="svd",
        help="""The solver to use for whitening.
 - "svd" is more stable numerically if the problem is degenerate, and
   often faster when `n_observations <= n_features`.

 - "eigh" is generally more memory efficient when
   `n_samples >= n_features`, and can be faster when
   `n_samples >= 50 * n_features`.""",
    ).tag(config=True)

    random_state = tr.Union(
        (tr.Int(), tr.Instance(RandomState)),
        allow_none=True,
        default_value=None,
        help="""Used to initialize ``w_init`` when not specified, with a
normal distribution. Pass an int, for reproducible results across multiple function
calls.""",
    ).tag(config=True)

    # ----------------------------------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------------------------------
    def __init__(
        self,
        *,
        log_level="WARNING",
        config=None,
        warm_start=False,
        copy=True,
        **kwargs,
    ):
        # we have changed the name n_components use in sklearn by
        # used_components (in order  to avoid conflict with the rest of the progrma)
        # warn th user:
        if "n_components" in kwargs:
            raise KeyError(
                "`n_components` is not a valid parameter. Did-you mean "
                "`used_components`?"
            )

        # call the super class for initialisation of the configuration parameters
        # to do before anything else!
        super().__init__(
            log_level=log_level,
            warm_start=warm_start,
            config=config,
            copy=copy,
            **kwargs,
        )

        # initialize sklearn FastICA
        self._ica = decomposition.FastICA(
            n_components=self.used_components,
            algorithm=self.algorithm,
            whiten=self.whiten,
            fun=self.fun,
            fun_args=self.fun_args,
            max_iter=self.max_iter,
            tol=self.tol,
            w_init=self.w_init,
            whiten_solver=self.whiten_solver,
            random_state=self.random_state,
        )

    # ----------------------------------------------------------------------------------
    # Private methods (overloading abstract classes)
    # ----------------------------------------------------------------------------------
    def _fit(self, X, Y=None):
        # this method is called by the abstract class fit.
        # Input X is a np.ndarray
        # Y is ignored in this model

        # call the sklearn _fit function on data
        # _outfit is a tuple handle the eventual output of _fit for further processing.

        # The _outfit members are np.ndarrays
        _outfit = self._ica._fit_transform(X, compute_sources=False)
        self._n_components = int(
            self._ica.n_components
        )  # cast the returned int64 to int
        return _outfit

    def _fit_transform(self, X, Y=None):
        _outfit = self._ica._fit_transform(X, compute_sources=True)
        self._n_components = int(
            self._ica.n_components
        )  # cast the returned int64 to int
        return _outfit

    def _transform(self, X):
        return self._ica.transform(X)

    def _inverse_transform(self, X_transform):
        # we need to  set self._ica.components_ to a compatible size but without
        # destroying the full matrix:
        store_components_ = self._ica.components_
        self._ica.components_ = self._ica.components_[: X_transform.shape[1]]
        X = self._ica.inverse_transform(X_transform)
        # restore
        self._ica.components_ = store_components_
        return X

    def _get_components(self):
        self._components = self._ica.components_
        return self._components

    # ----------------------------------------------------------------------------------
    # Public methods and properties specific to FastICA
    # ----------------------------------------------------------------------------------
    @property
    def n_components(self):
        """
        Return the number of components that were fitted.
        """
        if self._fitted:
            return self._n_components
        else:
            raise NotFittedError("n_components")
