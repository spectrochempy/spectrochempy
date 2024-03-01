# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Implementation of FastICA model (using scikit-learn library)
"""

import traitlets as tr
from numpy.random import RandomState
from sklearn import decomposition

from spectrochempy.analysis._base._analysisbase import (
    DecompositionAnalysis,
    _wrap_ndarray_output_to_nddataset,
)
from spectrochempy.utils.decorators import signature_has_configurable_traits
from spectrochempy.utils.docstrings import _docstring
from spectrochempy.utils.traits import NDDatasetType

__all__ = ["FastICA"]
__configurables__ = ["FastICA"]


# ======================================================================================
# class FastICA
# ======================================================================================
@signature_has_configurable_traits
class FastICA(DecompositionAnalysis):
    _docstring.delete_params("DecompositionAnalysis.see_also", "FastICA")

    __doc__ = _docstring.dedent(
        r"""
    Fast algorithm for Independent Component Analysis (FastICA).

    A wrapper of `sklearn.decomposition.FastICA`\ .

    :term:`ICA` (Independent Component Analysis) extracts the underlying sources of
    the variability of a set of spectra :math:`X` into the spectral profiles :math:`S^t`
    of the underlying sources and a mixing matrix :math:`A`\ .

    In terms of matrix equation:

    .. math:: X = \bar{X} + A \cdot S^t + E

    where :math:`\bar{X}` is the mean of the dataset and :math:`E` is the matrix of
    residuals.

    Parameters
    ----------
    %(AnalysisConfigurable.parameters)s

    See Also
    --------
    %(DecompositionAnalysis.see_also.no_FastICA)s
    """
    )

    # ----------------------------------------------------------------------------------
    # Runtime Parameters,
    # only those specific to FastICA, the other being defined in AnalysisConfigurable.
    # ----------------------------------------------------------------------------------
    # define here only the variable that you use in fit or transform functions
    _fastica = tr.Instance(
        decomposition.FastICA,
        help="The instance of sklearn.decomposition.FastICA used in this model",
    )

    # ----------------------------------------------------------------------------------
    # Configuration parameters
    # ----------------------------------------------------------------------------------

    n_components = tr.Integer(
        default_value=None,
        allow_none=True,
        help="Number of components (sources) to use.",
    ).tag(config=True)

    algorithm = tr.Enum(
        ["parallel", "deflation"],
        default_value="parallel",
        help=("""Specify which algorithm to use for FastICA."""),
    ).tag(config=True)

    whiten = tr.Union(
        (tr.Enum(["arbitrary-variance", "unit-variance"]), tr.Bool(False)),
        default_value="unit-variance",
        allow_none=True,
        help=(
            """Specify the whitening strategy to use.

- ``"arbitrary-variance"``\ : a whitening with variance arbitrary is used.
- "unit-variance" : the whitening matrix is rescaled to ensure that each recovered
  source has unit variance.
- False : the data is already considered to be whitened, and no whitening is performed.
"""
        ),
    ).tag(config=True)

    fun = tr.Union(
        (tr.Enum(["logcosh", "exp", "cube"]), tr.Callable(), tr.Unicode()),
        default_value="logcosh",
        allow_none=True,
        help=(
            r"""The functional form of the function used in the approximation to neg-entropy.

- ``string``\ : could be either ``"logcosh"``\ , ``"exp"``\ , or ``"cube"``\ .
- ``callable``\ : You can provide your own function. It should return a tuple containing
  the value of the function, and of its derivative, in the point. The derivative should
  be averaged along its last dimension.

.. code-block::

    def my_g(x):
        return x ** 3, (3 * x ** 2).mean(axis=-1)

.. note::
    ``fun`` can be also a serialized function created using dill and base64
    python libraries. Normally not used directly, it is here for internal
    process. """
        ),
    ).tag(config=True)

    fun_args = tr.Dict(
        default_value=None,
        allow_none=True,
        help=(
            """Arguments to send to the functional form.

If empty or None and if ``fun=="logcosh"`` , `fun_args` will take value
``{alpha : 1.0}``."""
        ),
    ).tag(config=True)

    tol = tr.Float(
        default_value=1e-4,
        help="Tolerance at which the un-mixing matrix is considered to have converged.",
    ).tag(config=True)

    max_iter = tr.Integer(
        default_value=200, help="Maximum number of iterations before timing out."
    ).tag(config=True)

    w_init = tr.Union(
        (NDDatasetType(),),
        default_value=None,
        allow_none=True,
        help=(
            r"""Initial un-mixing array.

NDDataset or array-like of shape (n_components, n_components). If w_init=None, then an
array of values drawn from a normal distribution is used."""
        ),
    ).tag(config=True)

    whiten_solver = tr.Enum(
        ["svd", "eigh"],
        default_value="svd",
        help=(
            """The solver to use for whitening.

- ``"svd"``\ : is more stable numerically if the problem is degenerate, and often faster
  when :term:`n_observations` <= :term:`n_features`\ .
- ``"eigh"``\ : is generally more memory efficient when
  :term:`n_observations` >= :term:`n_features`\ , and can be faster when
  :term:`n_observations` >= 50 * :term:`n_features`\ . """
        ),
    ).tag(config=True)

    random_state = tr.Union(
        (tr.Integer(), tr.Instance(RandomState)),
        allow_none=True,
        default_value=None,
        help=(
            "Used to initialize ``w_init`` when not specified, with a normal"
            "distribution. Pass an ``int``, for reproducible results across "
            "multiple function calls."
        ),
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

        # initialize sklearn Fast!ICA
        self._fast_ica = decomposition.FastICA(
            n_components=self.n_components,
            algorithm=self.algorithm,
            whiten=self.whiten,
            fun=self.fun,
            fun_args=self.fun_args,
            tol=self.tol,
            max_iter=self.max_iter,
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
        _outfit = self._fast_ica.fit(X)
        self._n_components = int(
            self._fast_ica.n_components
        )  # cast the returned int64 to int
        return _outfit

    def _transform(self, X):
        return self._fast_ica.transform(X)

    def _inverse_transform(self, X_transform):
        # we need to  set self._fast_ica.components_ to a compatible size but without
        # destroying the full matrix:
        store_components_ = self._fast_ica.components_
        self._fast_ica.components_ = self._fast_ica.components_[: X_transform.shape[1]]
        X = self._fast_ica.inverse_transform(X_transform)
        # restore
        self._fast_ica.components_ = store_components_
        return X

    def _get_components(self):
        self._components = self._fast_ica.components_
        return self._components

    _docstring.keep_params("analysis_fit.parameters", "X")

    @_docstring.dedent
    def fit(self, X):
        """
        Fit the FastICA model on X.

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
        units=None, title=None, typey="features", typex="components"
    )
    def mixing(self):
        """
        The pseudo inverse of components.

        NDDataset of size (`n_features`\ , `n_components`\ ). It is the linear operator
        that maps independent sources to the data, and the transpose of `St`\ .
        """
        return self._fast_ica.mixing_

    @property
    @_wrap_ndarray_output_to_nddataset(
        units=None,
        title=None,
        typey="components",
    )
    def St(self):
        """
        The spectral profiles of the independant sources.

        NDDataset of size (`n_components`\ , `n_features`\ ). It is the transpose of the
        ``mixing_`` matrix returned by Scikit-Learn.
        """
        return self._fast_ica.mixing_.T

    @property
    @_wrap_ndarray_output_to_nddataset(
        units=None,
        title=None,
        typex="components",
    )
    def A(self):
        """
        The mixing system A.

        NDDataset of size (`n_observations`\ , `n_components`\ ). It is the matrix
        returned by the `transform()` method.
        """
        return self._fast_ica.transform(self.X.data)

    @property
    @_wrap_ndarray_output_to_nddataset()
    def mean(self):
        """
        The mean of X over features.

        Only set if `whiten` is True, it is needed (and used) to reconstruct a dataset
        by ``inverse_transform(A)``\ .
        """
        return self._fast_ica.mean_

    @property
    def n_iter(self):
        """
        Number of iterations.

        If the algorithm is “deflation”, n_iter is the maximum number of iterations run
        across all components. Else they are just the number of iterations taken to
        converge.
        """
        return self._fast_ica.n_iter_

    @property
    @_wrap_ndarray_output_to_nddataset(
        units=None,
        title=None,
        typey="components",
    )
    def whitening(self):
        """
        NDDataset of shape (n_components, n_features).

        Only set if whiten is not None. This is the pre-whitening matrix that projects
        data onto the first n_components principal components.
        """
        if self.whiten:
            return self._fast_ica.whitening_
