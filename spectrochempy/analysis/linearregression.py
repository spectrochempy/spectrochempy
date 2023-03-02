# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Implementation of least squares Linear Regression.
"""
import numpy as np
import traitlets as tr
from sklearn import linear_model

from spectrochempy.analysis._analysisutils import NotFittedError
from spectrochempy.analysis.abstractanalysis import AnalysisConfigurable
from spectrochempy.core.dataset.basearrays.ndarray import NDArray
from spectrochempy.extern.traittypes import Array
from spectrochempy.utils.docstrings import _docstring
from spectrochempy.utils.traits import NDDatasetType

__all__ = ["LSTSQ", "NNLS"]
__configurables__ = ["LSTSQ", "NNLS"]


# ======================================================================================
# Base class LinearRegressionAnalysis
# ======================================================================================
class LinearRegressionAnalysis(AnalysisConfigurable):
    __doc__ = _docstring.dedent(
        """
    Ordinary least squares Linear Regression.

    Use :class:`~sklearn.linear_model.LinearRegression`

    LinearRegression fits a linear model with coefficients w = (w1, ..., wp)
    to minimize the residual sum of squares between the observed targets in
    the dataset, and the targets predicted by the linear approximation.

    Parameters
    ----------
    %(AnalysisConfigurable.parameters)s
    """
    )

    # ----------------------------------------------------------------------------------
    # Configuration parameters (mostly defined in subclass
    # as they depend on the model estimator)
    # ----------------------------------------------------------------------------------
    fit_intercept = tr.Bool(
        default_value=True,
        help="Whether to calculate the intercept for this model. If set to False, "
        "no intercept will be used in calculations (i.e. data is expected to be "
        "centered).",
    ).tag(config=True)

    positive = tr.Bool(
        default_value=False,
        help="When set to ``True``, forces the coefficients to be positive. This"
        "option is only supported for dense arrays.",
    ).tag(config=True)

    # ----------------------------------------------------------------------------------
    # Runtime Parameters (in addition to those of AnalysisConfigurable)
    # ----------------------------------------------------------------------------------
    _Y = NDDatasetType()
    _Y_preprocessed = Array(help="preprocessed Y")

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

        # call the super class for initialisation of the configuration parameters
        # to do before anything else!
        super().__init__(
            log_level=log_level,
            warm_start=warm_start,
            config=config,
            copy=copy,
            **kwargs,
        )

        # initialize sklearn LinearRegression
        self._linear_regression = linear_model.LinearRegression(
            fit_intercept=self.fit_intercept,
            copy_X=copy,
            n_jobs=None,  # not used for the moment (XXX: should we add this?)
            positive=self.positive,
        )

        # unlike decomposition methods, we output ndarray when the input
        # is not a dataset
        self._output_type == "ndarray"

    # ----------------------------------------------------------------------------------
    # Private validation and default getter methods
    # ----------------------------------------------------------------------------------
    @tr.validate("_Y")
    def _Y_validate(self, proposal):
        # validation of the _Y attribute: fired when self._Y is assigned
        Y = proposal.value

        # we need a dataset or a list of NDDataset with eventually  a copy of the
        # original data (default being to copy them)

        Y = self._make_dataset(Y)
        return Y

    @property
    def _Y_is_missing(self):
        # check wether or not Y has been already defined
        try:
            if self._Y is None:
                return True
        except NotFittedError:
            return True
        return False

    # ----------------------------------------------------------------------------------
    # Private methods
    # ----------------------------------------------------------------------------------
    @tr.observe("_Y")
    def _preprocess_as_Y_changed(self, change):
        # to be optionally replaced by user defined function (with the same name)
        Y = change.new
        # optional preprocessing as scaling, centering, ...
        # return a np.ndarray
        self._Y_preprocessed = Y.data

    def _fit(self, X, Y=None, sample_weight=None):
        # this method is called by the abstract class fit.
        _outfit = self._linear_regression.fit(X, Y, sample_weight=sample_weight)
        return _outfit

    # ----------------------------------------------------------------------------------
    # Public methods
    # ----------------------------------------------------------------------------------
    def fit(self, X, Y=None, sample_weight=None):
        """
        Fit linear model

        Parameters
        ----------
        X : NDDataset or array-like of shape (n_observations, n_features)
            Training data, where `n_observations` is the number of observations
            and `n_features` is the number of features.
        Y : array_like of shape (n_observations,) or (n_observations, n_targets)
            Target values. Will be cast to X's dtype if necessary.
        sample_weight : array-like of shape (n_observations,), default=None
            Individual weights for each sample.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._fitted = False  # reiniit this flag

        # store if the original input type is a dataset (or at least a subclass instance
        # of NDArray)
        self._is_dataset = isinstance(X, NDArray)

        # fire the X and Y validation and preprocessing.
        if Y is not None:
            self._X = X
            self._Y = Y
        else:
            # X should contain the X and Y information (X being the coord and Y the data)
            if X.coordset is None:
                raise ValueError(
                    "The passed argument must have a x coordinates,"
                    "or X input and Y target must be passed separately"
                )
            self._X = X.coord(0)
            self._Y = X

        # _X_preprocessed has been computed when X was set, as well as _Y_preprocessed.
        # At this stage they should be simple ndarrays
        newX = self._X_preprocessed
        newY = self._Y_preprocessed

        # call to the actual _fit method (overloaded in the subclass)
        # warning : _fit must take ndarray arguments not NDDataset arguments.
        # when method must return NDDataset from the calculated data,
        # we use the decorator _wrap_ndarray_output_to_nddataset, as below or in the PCA
        # model for example.
        self._outfit = self._fit(newX, newY, sample_weight=sample_weight)

        # if the process was succesful,_fitted is set to True so that other method which
        # needs fit will be possibly used.
        self._fitted = True
        return self

    @property
    def Y(self):
        """
        Return the Y input dataset
        """
        # We use Y property only to show this information to the end user. Internally
        # we use _Y attribute to refer to the input data
        if self._Y_is_missing:
            raise NotFittedError
        Y = self._Y
        if self._is_dataset or self._output_type == "NDDataset":
            return Y
        else:
            return np.asarray(Y)

    @property
    def coef(self):
        """Estimated coefficients for the linear regression problem.

        If multiple targets are passed during the fit (Y 2D), this is a 2D array of
        shape (n_targets, n_features), while if only one target is passed,
        this is a 1D array of length n_features.
        """
        if self._linear_regression.coef_.size == 1:
            # this is the result of the single equation, so only one value
            # should be returned
            A = float(self._linear_regression.coef_)
            if self._is_dataset and self._Y.has_units and self._X.has_units:
                A = A * self._Y.units / self._X.units
        elif self._is_dataset:

            unitsX = self._X.units if self._X.units is not None else 1.0
            unitsY = self._Y.units if self._Y.units is not None else 1.0
            if unitsX != 1 or unitsY != 1:
                units = self._Y.units / self._X.units
            else:
                units = None

            A = type(self._X)(
                data=self._linear_regression.coef_,
                coordset=self._Y.coordset,
                dims=self._Y.T.dims,
                units=units,
                title=f"{self._Y.title} / {self._X.title}",
                history="Computed from the LSTSQ model",
            )
        return A

    @property
    def intercept(self):
        """
        Return a float or na array of shape (n_targets,).

        Independent term in the linear model. Set to 0.0 if fit_intercept = False.
        If Y has units, then intercept has the same units.
        """
        if self._linear_regression.intercept_.size == 1:
            # A single value, return the associated quantity
            B = self._linear_regression.intercept_
            if self._is_dataset and self._Y.has_units:
                B = B * self._Y.units
        elif self._is_dataset:
            # else, return a NDDataset with the same units has Y
            B = type(self._X)(
                data=self._linear_regression.intercept_,
                coordset=self._Y.coordset,
                dims=self._Y.dims,
                units=self._Y.units,
                title=f"{self._Y.title} at origin",
                history="Computed from the LSTSQ model",
            )
        return B

    def predict(self, X=None):
        """
        Predict using the linear model

        Parameters
        ----------
        X : NDDataset or array-like matrix, shape (n_observations, n_features)
            Observations. If X is not set, the input X for fit is used.

        Returns
        -------
        C : array, shape (n_observations,)
            Returns predicted values.
        """
        if not self._fitted:
            raise NotFittedError()

        # Fire the validation and preprocessing
        if X is not None:
            # _is_dataset = hasattr(X, "_implements") and X._implements("NDDataset")
            self._X = X

        # Get the processed ndarray data
        newX = self._X_preprocessed

        predicted = self._linear_regression.predict(newX)

        if self._is_dataset:
            predicted = type(self._X)(
                predicted,
                coordset=self._Y.coordset,
                dims=self._Y._dims,
                units=self._Y.units,
                title=self._Y.title,
                history="Computed from a LSTSQ model",
            )

        return predicted

    def score(self, X=None, Y=None, sample_weight=None):
        """Return the coefficient of determination of the prediction.

        The coefficient of determination :math:`R^2` is defined as
        :math:`(1 - \\frac{u}{v})`, where :math:`u` is the residual
        sum of squares ``((y_true - y_pred)** 2).sum()`` and :math:`v`
        is the total sum of squares ``((y_true - y_true.mean()) ** 2).sum()``.
        The best possible score is 1.0 and it can be negative (because the
        model can be arbitrarily worse). A constant model that always predicts
        the expected value of `y`, disregarding the input features, would get
        a :math:`R^2` score of 0.0.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_features)
            Test samples. For some estimators this may be a precomputed
            kernel matrix or a list of generic objects instead with shape
            ``(n_observations, n_observations_fitted)``, where ``n_observations_fitted``
            is the number of observations used in the fitting for the estimator.

        Y : array-like of shape (n_observations,) or (n_observations, n_outputs)
            True values for `X`.

        sample_weight : array-like of shape (n_observations,), default=None
            Sample weights.

        Returns
        -------
        score : float
            :math:`R^2` of ``self.predict(X)`` wrt. `Y`.
        """
        if not self._fitted:
            raise NotFittedError()

        # Fire the validation and preprocessing
        self._X = X if X is not None else self.X
        self._Y = Y if X is not None else self.Y

        # Get the processed ndarray data
        newX = self._X_preprocessed
        newY = self._Y_preprocessed

        return self._linear_regression.score(newX, newY, sample_weight=sample_weight)


# ======================================================================================
# class LSTSQ
# ======================================================================================
class LSTSQ(LinearRegressionAnalysis):

    name = "LSTSQ"
    description = "Ordinary Least Squares Linear Regression"


# ======================================================================================
# class NNLS
# ======================================================================================
class NNLS(LinearRegressionAnalysis):
    name = "NNLS"
    description = "Non-Negative Least Squares Linear Regression"
