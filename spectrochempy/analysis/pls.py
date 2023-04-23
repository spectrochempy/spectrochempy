# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Implementation of Partial Least Square regression (using scikit-learn library)
"""

import traitlets as tr
from sklearn import cross_decomposition

from spectrochempy.analysis._base import (
    CrossDecompositionAnalysis,
    _wrap_ndarray_output_to_nddataset,
)
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.utils.decorators import signature_has_configurable_traits
from spectrochempy.utils.docstrings import _docstring

__all__ = ["PLS"]
__configurables__ = ["PLS"]


# ======================================================================================
# class PLS
# ======================================================================================
_docstring.delete_params("AnalysisConfigurable.parameters", "copy")


@signature_has_configurable_traits
class PLS(CrossDecompositionAnalysis):
    _docstring.delete_params("DecompositionAnalysis.see_also", "PLS")

    __doc__ = _docstring.dedent(
        """
    Partial Least Squares regression (PLS).

    The  Partial Least Squares regression wraps the
    `sklearn.cross_decomposition.PLSRegression` model, with few
    additional methods.

    Parameters
    ----------
    %(AnalysisConfigurable.parameters.no_copy)s

    """
    )

    # ----------------------------------------------------------------------------------
    # Runtime Parameters,
    # only those specific to PLS, the other being defined in AnalysisConfigurable.
    # ----------------------------------------------------------------------------------
    # define here only the variable that you use in fit or transform functions
    _pls = tr.Instance(
        cross_decomposition.PLSRegression,
        help="The instance of sklearn.cross_decomposition.PLSRegression used in this model",
    )

    # ----------------------------------------------------------------------------------
    # Configuration parameters
    # ----------------------------------------------------------------------------------

    used_components = tr.Int(
        default_value=2,
        help="Number of components to keep. Should be in the range [1, min(n_samples, "
        "n_features, n_targets)].",
    ).tag(config=True)

    scale = tr.Bool(default_value=True, help="Whether to scale X and Y.").tag(
        config=True
    )

    max_iter = tr.Int(
        default_value=500,
        help="The maximum number of iterations of the power method when "
        "algorithm='nipals'. Ignored otherwise.",
    ).tag(config=True)

    tol = tr.Float(
        default_value=1.0e-6,
        help="The tolerance used as convergence criteria in the power method:"
        "the algorithm stops whenever the squared norm of u_i - u_{i-1} "
        "is less than tol, where u corresponds to the left singular vector.",
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
        # we have changed the name n_components use in sklearn by
        # used_components (in order  to avoid conflict with the rest of the program)
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
            **kwargs,
        )

        # initialize sklearn PLS
        self._pls = cross_decomposition.PLSRegression(
            n_components=self.used_components,
            scale=self.scale,
            max_iter=self.max_iter,
            tol=self.tol,
        )

    # ----------------------------------------------------------------------------------
    # Private methods (overloading abstract classes)
    # ----------------------------------------------------------------------------------

    def _fit(self, X, Y):
        # this method is called by the abstract class fit.
        # Input X and Y are np.ndarray
        self._pls.fit(X, Y)
        self._x_weights = self._pls.x_weights_.T
        self._y_weights = self._pls.y_weights_.T
        self._x_loadings = self._pls.x_loadings_.T
        self._y_loadings = self._pls.y_loadings_.T
        self._x_scores = self._pls.x_scores_
        self._y_scores = self._pls.y_scores_
        self._x_rotations = self._pls.x_rotations_.T
        self._y_rotations = self._pls.y_rotations_.T

        self._coef = self._pls.coef_.T
        self._intercept = self._pls.intercept_
        self._n_iter = self._pls.n_iter_
        self._n_feature_in = self._pls.n_features_in_

        # todo: check self.feature_names_in = self._pls.feature_names_in_

        # for compatibility with superclass methods
        self._n_components = self.used_components

    def _fit_transform(self, X, Y=None):
        # Learn and apply the dimension reduction on the train data.
        #
        return self._pls.fit_transform(X, Y)

    # todo ?: def _get_feature_names_out([input_features])

    # todo ? : def _get_params([deep])

    def _inverse_transform(self, X_transform, Y_transform=None):
        # Transform data back to its original space.
        return self._pls.inverse_transform(X_transform, Y=Y_transform)

    def _transform(self, X, Y=None, copy=True):
        # Apply the dimension reduction.
        return self._pls.transform(X, Y, copy)

    def _predict(self, X):
        # Predict targets of given samples.
        return self._pls.predict(X)

    def _score(self, X, Y, sample_weight=None):
        # this method is called by the abstract class score.
        # Input X, Y, sample_weights are np.ndarray
        return self._pls.score(X, Y, sample_weight=sample_weight)

    # ----------------------------------------------------------------------------------
    # Public methods and properties specific to PLS
    # ----------------------------------------------------------------------------------
    _docstring.keep_params("analysis_fit.parameters", "X")

    @_docstring.dedent
    def fit(self, X, Y):
        """
        Fit the PLS model on X and Y.

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
        return super().fit(X, Y)

    @property
    @_wrap_ndarray_output_to_nddataset(
        units=None,
        title=None,
        typey="components",
    )
    def x_loadings(self):
        return self._x_loadings

    @property
    @_wrap_ndarray_output_to_nddataset(
        meta_from="_Y",
        units=None,
        title=None,
        typey="components",
    )
    def y_loadings(self):
        return self._y_loadings

    @property
    @_wrap_ndarray_output_to_nddataset(
        units=None,
        title=None,
        typex="components",
    )
    def x_scores(self):
        return self._x_scores

    @property
    @_wrap_ndarray_output_to_nddataset(
        meta_from="_Y",
        units=None,
        title=None,
        typex="components",
    )
    def y_scores(self):
        return self._y_scores

    @property
    @_wrap_ndarray_output_to_nddataset(
        units=None,
        title=None,
        typey="components",
    )
    def x_rotations(self):
        return self._x_rotations

    @property
    @_wrap_ndarray_output_to_nddataset(
        meta_from="_Y",
        typey="components",
    )
    def y_rotations(self):
        return self._y_rotations

    @property
    @_wrap_ndarray_output_to_nddataset(
        typey="components",
    )
    def x_weights(self):
        return self._x_weights

    @property
    @_wrap_ndarray_output_to_nddataset(
        meta_from="_Y",
        typey="components",
    )
    def y_weights(self):
        return self._y_weights

    @property
    def coef(self):
        coef = NDDataset(self._coef)
        coef.set_coordset(
            y=self._Y.x,
            x=self._X.x,
        )
        return coef

    @property
    @_wrap_ndarray_output_to_nddataset(
        meta_from="_Y",
        typesingle="targets",
    )
    def intercept(self):
        return self._intercept

    @property
    def n_iter(self):
        return self._n_iter_

    # ----------------------------------------------------------------------------------
    # Reporting specific to PCA
    # ----------------------------------------------------------------------------------

    # ----------------------------------------------------------------------------------
    # Plot methods specific to PLS
    # ----------------------------------------------------------------------------------


if __name__ == "__main__":
    pass
