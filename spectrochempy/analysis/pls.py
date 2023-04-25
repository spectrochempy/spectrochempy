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

__all__ = ["PLSRegression"]
__configurables__ = ["PLSRegression"]


# ======================================================================================
# class PLSRegression
# ======================================================================================
@signature_has_configurable_traits
class PLSRegression(CrossDecompositionAnalysis):
    _docstring.delete_params("DecompositionAnalysis.see_also", "PLSRegression")

    __doc__ = _docstring.dedent(
        """
    Partial Least Squares regression (PLSRegression).

    The  Partial Least Squares regression wraps the
    `sklearn.cross_decomposition.PLSRegression` model, with few
    additional methods.

    Parameters
    ----------
    %(AnalysisConfigurable.parameters)s

    """
    )

    # ----------------------------------------------------------------------------------
    # Runtime Parameters,
    # only those specific to PLSRegression, the other being defined in AnalysisConfigurable.
    # ----------------------------------------------------------------------------------
    # define here only the variable that you use in fit or transform functions
    _plsregression = tr.Instance(
        cross_decomposition.PLSRegression,
        help="The instance of sklearn.cross_decomposition.PLSRegression used in this model",
    )

    # ----------------------------------------------------------------------------------
    # Configuration parameters
    # ----------------------------------------------------------------------------------

    n_components = tr.Int(
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

        # call the super class for initialisation of the configuration parameters
        # to do before anything else!
        super().__init__(
            log_level=log_level,
            warm_start=warm_start,
            **kwargs,
        )

        # initialize sklearn PLSRegression
        self._plsregression = cross_decomposition.PLSRegression(
            n_components=self.n_components,
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
        self._plsregression.fit(X, Y)
        self._x_weights = self._plsregression.x_weights_.T
        self._y_weights = self._plsregression.y_weights_.T
        self._x_loadings = self._plsregression.x_loadings_.T
        self._y_loadings = self._plsregression.y_loadings_.T
        self._x_scores = self._plsregression.x_scores_
        self._y_scores = self._plsregression.y_scores_
        self._x_rotations = self._plsregression.x_rotations_.T
        self._y_rotations = self._plsregression.y_rotations_.T

        self._coef = self._plsregression.coef_.T
        self._intercept = self._plsregression.intercept_
        self._n_iter = self._plsregression.n_iter_
        self._n_feature_in = self._plsregression.n_features_in_

        # todo: check self.feature_names_in = self._plsregression.feature_names_in_

        # for compatibility with superclass methods
        self._n_components = self.n_components

    def _fit_transform(self, X, Y=None):
        # Learn and apply the dimension reduction on the train data.
        #
        return self._plsregression.fit_transform(X, Y)

    def _inverse_transform(self, X_transform, Y_transform=None):
        # Transform data back to its original space.
        return self._plsregression.inverse_transform(X_transform, Y=Y_transform)

    def _transform(self, X, Y=None):
        # Apply the dimension reduction.
        return self._plsregression.transform(X, Y)

    def _predict(self, X):
        # Predict targets of given samples.
        return self._plsregression.predict(X)

    def _score(self, X, Y, sample_weight=None):
        # this method is called by the abstract class score.
        # Input X, Y, sample_weights are np.ndarray
        return self._plsregression.score(X, Y, sample_weight=sample_weight)

    # ----------------------------------------------------------------------------------
    # Public methods and properties specific to PLSRegression
    # ----------------------------------------------------------------------------------
    _docstring.keep_params("analysis_fit.parameters", "X")

    @_docstring.dedent
    def fit(self, X, Y):
        """
        Fit the PLSRegression model on X and Y.

        Parameters
        ----------
        %(analysis_fit.parameters.X)s
        Y :  :term:`array-like` of shape (n_samples,) or (n_samples, n_targets)
            Target vectors, where n_samples is the number of samples and n_targets is the number of response variables.

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
    # Plot methods specific to PLSRegression
    # ----------------------------------------------------------------------------------


if __name__ == "__main__":
    pass
