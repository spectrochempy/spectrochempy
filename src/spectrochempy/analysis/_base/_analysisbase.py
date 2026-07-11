# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Module implementing the base abstract classes to define estimators such as PCA, ..."""

import logging
import warnings

import numpy as np
import traitlets as tr
from sklearn import linear_model

from spectrochempy.application.application import app
from spectrochempy.core.dataset.basearrays.ndarray import NDArray
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.extern.traittypes import Array
from spectrochempy.utils.baseconfigurable import BaseConfigurable
from spectrochempy.utils.decorators import _wrap_ndarray_output_to_nddataset
from spectrochempy.utils.decorators import deprecated
from spectrochempy.utils.exceptions import NotFittedError
from spectrochempy.utils.exceptions import SpectroChemPyError
from spectrochempy.utils.traits import NDDatasetType


# ======================================================================================
# Base class AnalysisConfigurable
# ======================================================================================
class AnalysisConfigurable(BaseConfigurable):
    """
    Abstract class to write analysis model estimators.

    Analysis model class must subclass this to get a minimal structure.

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

    """

    # Get doc sections for reuse in subclass

    # ----------------------------------------------------------------------------------
    # Runtime Parameters
    # ----------------------------------------------------------------------------------
    _fitted = tr.Bool(False, help="False if the model was not yet fitted")
    _outfit = tr.Any(help="the output of the _fit method - generally a tuple")

    # ----------------------------------------------------------------------------------
    # Configuration parameters (mostly defined in subclass
    # as they depend on the model estimator)
    # ----------------------------------------------------------------------------------

    # Write here traits like e.g.,
    #     A = Unicode("A", help='description").tag(config=True)

    # ----------------------------------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------------------------------
    def __init__(
        self,
        *,
        log_level=logging.WARNING,
        warm_start=False,
        **kwargs,
    ):
        self._warm_start = warm_start

        super().__init__(log_level=log_level, **kwargs)

        if not warm_start:
            # We should not be able to use any methods requiring fit results
            # until the fit method has been executed
            self._fitted = False

    # ----------------------------------------------------------------------------------
    # Private validation and default getter methods
    # ----------------------------------------------------------------------------------
    @tr.default("_X")
    def _X_default(self):
        raise NotFittedError

    @property
    def _X_is_missing(self):
        # check whether X has been already defined
        try:
            if self._X is None:
                return True
        except NotFittedError:
            return True
        return False

    # ----------------------------------------------------------------------------------
    # Private methods that should be, most of the time, overloaded in subclass
    # ----------------------------------------------------------------------------------
    def _fit(self, X, Y=None):  # pragma: no cover
        #  Intended to be replaced in the subclasses by user defined function
        #  (with the same name)
        raise NotImplementedError("fit method has not yet been implemented")

    # ----------------------------------------------------------------------------------
    # Public methods and property
    # ----------------------------------------------------------------------------------
    def fit(self, X, Y=None):
        r"""
        Fit the model with ``X`` as input dataset.

        Parameters
        ----------
        X : `NDDataset` or :term:`array-like` of shape (:term:`n_observations`, :term:`n_features`)
            Training data.
        Y : any
            Depends on the model.

        Returns
        -------
        self
            The fitted instance itself.

        See Also
        --------
        fit_transform :  Fit the model with an input dataset ``X`` and apply the dimensionality reduction on ``X``.
        fit_reduce : Alias of `fit_transform` (Deprecated).

        """
        self._fitted = False  # reinit this flag

        # fire the X and eventually Y validation and preprocessing.
        # X and Y are expected to be resp. NDDataset and NDDataset or list of NDDataset.
        self._X = X
        if Y is not None:
            self._Y = Y

        # _X_preprocessed has been computed when X was set, as well as _Y_preprocessed.
        # At this stage they should be simple ndarrays
        newX = self._X_preprocessed
        newY = self._Y_preprocessed if Y is not None else None

        # Call to the actual _fit method (overloaded in the subclass)
        # warning : _fit must take ndarray arguments not NDDataset arguments.
        # when method must return NDDataset from the calculated data,
        # we use the decorator _wrap_ndarray_output_to_nddataset, as in the PCA
        # model for example.
        try:
            self._outfit = self._fit(newX, newY)
        except TypeError:
            # in case Y s not used in _fit
            self._outfit = self._fit(newX)

        # if the process was successful, _fitted is set to True so that other method
        # which needs fit will be possibly used.
        self._fitted = True
        return self

    # we do not use this method as a decorator as in this case signature of subclasses
    # extract useful individual parameters doc

    @property
    def log(self):
        """Return ``log`` output."""
        # A string handler (#1) is defined for the Spectrochempy logger,
        # thus we will return it's content
        return app.log.handlers[1].stream.getvalue().rstrip()

    @property
    def X(self):
        """Return the X input dataset (eventually modified by the model)."""
        if self._X_is_missing:
            raise NotFittedError
        # We use X property only to show this information to the end user. Internally
        # we use _X attribute to refer to the input data
        X = self._X.copy()
        if np.any(self._X_mask):
            # restore masked row and column if necessary
            X = self._restore_masked_data(X, axis="both")
        if self._is_dataset or self._output_type == "NDDataset":
            return X
        return np.asarray(X)

    def get_params(self, deep=True):
        r"""
        Get the configuration parameters of this estimator.

        Parameters
        ----------
        deep : `bool`, optional, default:`True`
            Ignored.  Present for compatibility with scikit-learn conventions.

        Returns
        -------
        `dict`
            Mapping of parameter name -> current value.

        """
        return dict(self.params())

    def set_params(self, **params):
        r"""
        Set configuration parameters on this estimator.

        Returns `self` so that calls can be chained.

        Parameters
        ----------
        **params
            Parameter names and values to update.

        Returns
        -------
        self
            The estimator instance.

        Raises
        ------
        SpectroChemPyError
            If a parameter name does not correspond to a configurable trait.

        """
        for key, value in params.items():
            if not hasattr(self, key):
                raise SpectroChemPyError(
                    f"Invalid parameter '{key}' for {self.__class__.__name__}."
                )
            setattr(self, key, value)
        return self

    def __repr__(self):
        cls = self.__class__.__name__
        params = self.get_params()
        # Show a concise subset for readability
        display = {k: v for k, v in params.items() if not k.startswith("_")}
        if not display:
            return f"{cls}()"
        items = ", ".join(f"{k}={v!r}" for k, v in display.items())
        return f"{cls}({items})"


# ======================================================================================
# Base class DecompositionAnalysis
# ======================================================================================
class DecompositionAnalysis(AnalysisConfigurable):
    """
    Abstract class to write analysis decomposition models such as `PCA`, ...

    Subclass this to get a minimal structure

    See Also
    --------
    EFA : Perform an Evolving Factor Analysis (forward and reverse).
    FastICA : Perform Independent Component Analysis with a fast algorithm.
    IRIS : Integral inversion solver for spectroscopic data.
    MCRALS : Perform MCR-ALS of a dataset knowing the initial :math:`C` or :math:`S^T` matrix.
    NMF : Non-Negative Matrix Factorization.
    PCA : Perform Principal Components Analysis.
    SIMPLISMA : SIMPLe to use Interactive Self-modeling Mixture Analysis.
    SVD : Perform a Singular Value Decomposition.

    """

    # This class is subclass AnalysisConfigurable, so we define only additional
    # attributes and methods necessary for decomposition model.

    # Get doc sections for reuse in subclass

    # ----------------------------------------------------------------------------------
    # Runtime Parameters (in addition to those of AnalysisConfigurable)
    # ----------------------------------------------------------------------------------
    _Y = tr.Union(
        (
            tr.Tuple(NDDatasetType(), NDDatasetType()),
            NDDatasetType(),
        ),
        default_value=None,
        allow_none=True,
        help="Target/profiles taken into account to fit a model",
    )
    _Y_preprocessed = tr.Union((tr.List(Array()), Array()), help="preprocessed Y")
    _n_components = tr.Integer(help="""The actual number of components.""")
    _components = Array(help="the array of (n_components, n_features) components")

    # ----------------------------------------------------------------------------------
    # Private validation and default getter methods
    # ----------------------------------------------------------------------------------
    @tr.validate("_Y")
    def _Y_validate(self, proposal):
        # validation of the _Y attribute: fired when self._Y is assigned
        Y = proposal.value

        # we need a dataset or a list of NDDataset
        return self._make_dataset(Y)

    @property
    def _Y_is_missing(self):
        # check whether or not Y has been already defined
        try:
            if self._Y is None:
                return True
        except NotFittedError:
            return True
        return False

    @tr.default("_n_components")
    def _n_components_default(self):
        # ensure model fitted before using this value
        if not self._fitted:
            raise NotFittedError("_n_components")

    # ----------------------------------------------------------------------------------
    # Private methods that should be most of the time overloaded in subclass
    # ----------------------------------------------------------------------------------
    @tr.observe("_Y")
    def _preprocess_as_Y_changed(self, change):
        # to be optionally replaced by user defined function (with the same name)
        Y = change.new
        # optional preprocessing as scaling, centering, ...
        # return a np.ndarray
        self._Y_preprocessed = Y.data

    def _transform(self, *args, **kwargs):  # pragma:  no cover
        # to be overridden in subclass such as PCA, MCRALS, ...
        raise NotImplementedError("transform has not yet been implemented")

    def _inverse_transform(self, *args, **kwargs):  # pragma:  no cover
        # to be overridden in subclass such as PCA, MCRALS, ...
        raise NotImplementedError("inverse_transform has not yet been implemented")

    def _get_components(self, n_components=None):  # pragma:  no cover
        # to be overridden in subclass such as PCA, MCRALS, ...
        raise NotImplementedError("get_components has not yet been implemented")

    # ----------------------------------------------------------------------------------
    # Public methods
    # ----------------------------------------------------------------------------------
    @_wrap_ndarray_output_to_nddataset(units=None, title=None, typex="components")
    def transform(self, X=None, **kwargs):
        r"""
        Apply dimensionality reduction to `X`.

        Parameters
        ----------
        X : `NDDataset` or :term:`array-like` of shape (:term:`n_observations`, :term:`n_features`), optional
            New data, where :term:`n_observations` is the number of observations
            and :term:`n_features` is the number of features.
            if not provided, the input dataset of the `fit` method will be used.
        **kwargs : keyword parameters, optional
            See Other Parameters.

        Returns
        -------
        `NDDataset`
            Dataset with shape (:term:`n_observations`, :term:`n_components`).

        Other Parameters
        ----------------
        n_components : `int`, optional
            The number of components to use for the reduction. If not given
            the number of components is eventually the one specified or determined
            in the `fit` process.

        """
        if not self._fitted:
            raise NotFittedError()

        # Fire the validation and preprocessing
        self._X = X if X is not None else self.X.copy()

        # Get the processed ndarray data
        newX = self._X_preprocessed

        X_transform = self._transform(newX)

        # Slice according to n_components
        n_components = kwargs.pop("n_components", self._n_components)
        if n_components > self._n_components:
            warnings.warn(
                "The number of components required for reduction "
                "cannot be greater than the fitted model components : "
                f"{self._n_components}. We then use this latter value.",
                stacklevel=2,
            )
        if n_components < self._n_components:
            X_transform = X_transform[:, :n_components]

        return X_transform

    # Get doc sections for reuse in subclass

    @_wrap_ndarray_output_to_nddataset
    def inverse_transform(self, X_transform=None, **kwargs):
        r"""
        Transform data back to its original space.

        In other words, return an input `X_original` whose reduce/transform would
        be `X_transform`.

        Parameters
        ----------
        X_transform : array-like of shape (:term:`n_observations`, :term:`n_components`), optional
            Reduced `X` data, where `n_observations` is the number of observations
            and `n_components` is the number of components. If `X_transform` is not
            provided, a transform of `X` provided in `fit` is performed first.
        **kwargs : keyword parameters, optional
            See Other Parameters.

        Returns
        -------
        `NDDataset`
            Dataset with shape (:term:`n_observations`, :term:`n_features`).

        Other Parameters
        ----------------
        n_components : `int`, optional
            The number of components to use for the reconstruction.

        See Also
        --------
        reconstruct : Alias of inverse_transform (Deprecated).

        """
        if not self._fitted:
            raise NotFittedError

        # get optional n_components
        n_components = kwargs.pop("n_components", self._n_components)
        if n_components > self._n_components:
            warnings.warn(
                "The number of components required for reduction "
                "cannot be greater than the fitted model components : "
                f"{self._n_components}. We then use this latter value.",
                stacklevel=2,
            )

        if isinstance(X_transform, NDDataset):
            X_transform = X_transform.data
            if n_components > X_transform.shape[1]:
                warnings.warn(
                    "The number of components required for reduction "
                    "cannot be greater than the X_transform size : "
                    f"{X_transform.shape[1]}. We then use this latter value.",
                    stacklevel=2,
                )
        elif X_transform is None:
            X_transform = self.transform(**kwargs).data

        return self._inverse_transform(X_transform)

    def fit_transform(self, X, Y=None, **kwargs):
        r"""
        Fit the model with `X` and apply the dimensionality reduction on `X`.

        Parameters
        ----------
        X : `NDDataset` or :term:`array-like` of shape (:term:`n_observations`, :term:`n_features`)
            Training data.
        Y : any
            Depends on the model.
        **kwargs : keyword arguments, optional
            Additional keyword arguments passed to the underlying implementation.

        Returns
        -------
        `NDDataset`
            Dataset with shape (:term:`n_observations`, :term:`n_components`).

        Other Parameters
        ----------------
        n_components : `int`, optional
            The number of components to use for the reduction.

        """
        try:
            self.fit(X, Y)
        except TypeError:
            # the current model does not use Y
            self.fit(X)
        return self.transform(X, **kwargs)

    @_wrap_ndarray_output_to_nddataset(units=None, title=None, typey="components")
    def get_components(self, n_components=None):
        r"""
        Return the component's dataset: (selected :term:`n_components`, :term:`n_features`).

        Parameters
        ----------
        n_components : `int`, optional, default: `None`
            The number of components to keep in the output dataset.
            If `None`, all calculated components are returned.

        Returns
        -------
        `~spectrochempy.core.dataset.nddataset.NDDataset`
            Dataset with shape (:term:`n_components`, :term:`n_features`)

        """
        if n_components is None or n_components > self._n_components:
            n_components = self._n_components

        # we call the specific _get_components method defined in subclasses
        return self._get_components()[:n_components]

    @property
    @_wrap_ndarray_output_to_nddataset(units=None, title="keep", typey="components")
    def components(self):
        r"""
        `NDDataset` with components in feature space (:term:`n_components`, :term:`n_features`).

        See Also
        --------
        get_components : Retrieve only the specified number of components.

        """
        return self._get_components()

    @property
    def n_components(self):
        """Number of components that were fitted."""
        if self._fitted:
            return self._n_components
        raise NotFittedError("n_components")

    def _get_component_labels(self, n):
        """
        Return default labels for *n* components.

        Subclasses may override this to provide domain-specific labels
        (e.g. ``PC1``, ``PC2``, ... for PCA).
        """
        return [f"#{i}" for i in range(n)]

    # ----------------------------------------------------------------------------------
    # Plot methods
    # ----------------------------------------------------------------------------------
    def plot_merit(self, X=None, X_hat=None, **kwargs):
        r"""
        Plot the input (`X`), reconstructed (`X_hat`) and residuals.

        :math:`X` and :math:`\hat{X}` can be passed as arguments. If not,
        the `X` attribute is used for :math:`X`and :math:`\hat{X}`is computed by
        the `inverse_transform` method

        Parameters
        ----------
        X : `NDDataset`, optional
            Original dataset. If is not provided (default), the `X`
            attribute is used and X_hat is computed using `inverse_transform`.
        X_hat : `NDDataset`, optional
            Inverse transformed dataset. if `X` is provided, `X_hat`
            must also be provided as compuyed externally.

        Returns
        -------
        `~matplotlib.axes.Axes`
            Matplotlib subplot axe.

        Other Parameters
        ----------------
        exp_c : color, colormap, or list of colors, optional
            Color(s) for experimental spectra.
            - None: use unified semantic resolver (auto-detect categorical/sequential)
            - Single color: use for all experimental spectra
            - Colormap name/object: sample colors from colormap
            - List/tuple: use as explicit color cycle
        calc_c : color, colormap, or list of colors, optional
            Color(s) for calculated spectra.
            - None: use default blue "#2a6fbb"
            - Single color: use for all calculated spectra
            - Colormap name/object: sample colors from colormap
            - List/tuple: use as explicit color cycle
        resid_c : color, colormap, or list of colors, optional
            Color(s) for residual spectra.
            - None: use default grey "0.4"
            - Single color: use for all residual spectra
            - Colormap name/object: sample colors from colormap
            - List/tuple: use as explicit color cycle
        exp_linestyle : str, optional
            Line style for experimental spectra. Default: "-".
        calc_linestyle : str, optional
            Line style for calculated spectra. Default: "--".
        resid_linestyle : str, optional
            Line style for residual spectra. Default: "-".
        exp_linewidth : float, optional
            Line width for experimental spectra. Default: 1.2.
        calc_linewidth : float, optional
            Line width for calculated spectra. Default: 1.0.
        resid_linewidth : float, optional
            Line width for residual spectra. Default: 1.0.
        min_contrast : float, optional
            Minimum contrast ratio for sequential colormaps. Default: 1.5.
        offset : `float`, optional, default: `None`
            Specify the separation (in percent) between the
            :math:`X` , :math:`X_hat` and :math:`E`.
        nb_traces : `int` or ``'all'``, optional
            Number of lines to display. Default is ``'all'``.
        **others : Other keywords parameters
            Parameters passed to the internal `plot` method of the `X` dataset.
            Common options include ``color``, ``linewidth``, ``linestyle``,
            ``alpha``, and standard Matplotlib kwargs.

        """
        from spectrochempy.plotting.composite.plotmerit import plot_merit

        return plot_merit(
            analysis_object=self,
            X=X,
            X_hat=X_hat,
            **kwargs,
        )

    # Backward compatibility alias
    @deprecated(replace="plot_merit", removed="0.12")
    def plotmerit(self, X=None, X_hat=None, **kwargs):
        """
        Backward-compatible alias for :meth:`plot_merit`. Deprecated.

        Returns
        -------
        `~matplotlib.axes.Axes`
            Matplotlib axes containing the plot.
        """
        return self.plot_merit(X, X_hat, **kwargs)

    @property
    def Y(self):
        r"""The `Y` input."""
        # We use Y property only to show this information to the end-user. Internally
        # we use _Y attribute to refer to the input data
        if self._Y_is_missing:
            raise NotFittedError
        return self._Y


# ======================================================================================
# Base class CrossDecompositionAnalysis
# ======================================================================================
class CrossDecompositionAnalysis(DecompositionAnalysis):
    """
    Abstract class to write analysis cross decomposition models such as `PLSRegression`, ...

    Subclass this to get a minimal structure

    See Also
    --------
    PLSRegression : Perform a Partial Least Square Regression .

    """

    # This class is a subclass of DecompositionAnalysis, so we define only additional
    # attributes and methods necessary for cross decomposition model.

    # Get doc sections for reuse in subclass

    # ----------------------------------------------------------------------------------
    # Private methods that should be most of the time overloaded in subclass
    # ----------------------------------------------------------------------------------
    def _predict(self, *args, **kwargs):  # pragma:  no cover
        # to be overridden in subclass such as PLSRegression, ...
        raise NotImplementedError("predict has not yet been implemented")

    # ----------------------------------------------------------------------------------
    # Public methods
    # ----------------------------------------------------------------------------------

    @_wrap_ndarray_output_to_nddataset(meta_from="_Y", title=None)
    def predict(self, X=None):
        r"""
        Predict targets of given observations.

        Parameters
        ----------
        X : `NDDataset` or :term:`array-like` of shape (:term:`n_observations`, :term:`n_features`), optional
            New data, where :term:`n_observations` is the number of observations
            and :term:`n_features` is the number of features.
            if not provided, the input dataset of the `fit` method will be used.

        Returns
        -------
        `NDDataset`
            Datasets with shape (:term:`n_observations`,) or ( :term:`n_observations`, :term:`n_targets`).

        """
        if not self._fitted:
            raise NotFittedError()

        if X is None:
            X = self._X_preprocessed
        elif isinstance(X, NDDataset):
            X = X.data

        return self._predict(X)

    def score(self, X=None, Y=None, sample_weight=None):
        r"""
        Return the coefficient of determination of the prediction.

        The coefficient of determination :math:`R^2` is defined as
        :math:`(1 - \frac{u}{v})` , where :math:`u` is the residual
        sum of squares ``((y_true - y_pred)** 2).sum()`` and :math:`v`
        is the total sum of squares ``((y_true - y_true.mean()) ** 2).sum()``.
        The best possible score is ``1.0`` and it can be negative (because the
        model can be arbitrarily worse). A constant model that always predicts
        the expected value of `Y`, disregarding the input features, would get
        a :math:`R^2` score of 0.0.

        Parameters
        ----------
        X : `NDDataset` or :term:`array-like` of shape (:term:`n_observations`, :term:`n_features`), optional
            Test samples. If not given, the X attribute is used.
        Y : `NDDataset` or :term:`array-like` of shape (:term:`n_observations`, :term:`n_targets`), optional
            True values for `X`.
        sample_weight : `NDDataset` or :term:`array-like` of shape (:term:`n_samples`,), default: `None`
            Sample weights.

        Returns
        -------
        `float`
            :math:`R^2` of `predict`(X) w.r.t `Y`.

        """
        if not self._fitted:
            raise NotFittedError()

        if X is None:
            X = self._X_preprocessed
        elif isinstance(X, NDDataset):
            X = X.data

        if Y is None:
            Y = self._Y_preprocessed
        elif isinstance(Y, NDDataset):
            Y = Y.data

        if isinstance(sample_weight, NDDataset):
            sample_weight = sample_weight.data

        return self._score(X, Y, sample_weight)

    @_wrap_ndarray_output_to_nddataset(
        units=None,
        title=None,
        meta_from=("_X", "_Y"),
        typex="components",
    )
    def transform(self, X=None, Y=None, both=False, **kwargs):
        r"""
        Apply dimensionality reduction to `X`and `Y`.

        Parameters
        ----------
        X : `NDDataset` or :term:`array-like` of shape (:term:`n_observations`, :term:`n_features`), optional
            New data, where :term:`n_observations` is the number of observations
            and :term:`n_features` is the number of features.
            if not provided, the input dataset of the `fit` method will be used.
        Y : `NDDataset` or :term:`array-like` of shape (:term:`n_observations`, :term:`n_targets`), optional
            New data, where :term:`n_targets` is the number of variables to predict.
            if not provided, the input dataset of the `fit` method will be used.
        both : `bool`, default: `False`
            Whether to also apply the dimensionality reduction to Y when neither X nor Y are provided.

        Returns
        -------
        x_score, y_score: `NDDataset` or tuple of `NDDataset`
            Datasets with shape (:term:`n_observations`, :term:`n_components`).

        """
        if not self._fitted:
            raise NotFittedError()

        # Fire the validation and preprocessing
        self._X = X if X is not None else self.X
        self._Y = Y if Y is not None else self.Y

        # Get the processed ndarray data
        newX = self._X_preprocessed
        newY = self._Y_preprocessed

        if both or (Y is not None):
            return self._transform(newX, newY)
        return self._transform(newX, None)

    # Get doc sections for reuse in subclass

    @_wrap_ndarray_output_to_nddataset(meta_from=("_X", "_Y"))
    def inverse_transform(
        self,
        X_transform=None,
        Y_transform=None,
        both=False,
        **kwargs,
    ):
        r"""
        Transform data back to its original space.

        In other words, return reconstructed `X` and `Y` whose reduce/transform would
        be `X_transform` and `Y_transform`.

        Parameters
        ----------
        X_transform : array-like of shape (:term:`n_observations`, :term:`n_components`), optional
            Reduced `X` data, where `n_observations` is the number of observations
            and `n_components` is the number of components. If `X_transform` is not
            provided, a transform of `X` provided in `fit` is performed first.
        Y_transform : `NDDataset` or :term:`array-like` of shape (:term:`n_observations`, `n_components`), optional
            New data, where :term:`n_targets` is the number of variables to predict. If `Y_transform` is not
            provided, a transform of `Y` provided in `fit` is performed first.
        **kwargs : keyword parameters, optional
            See Other Parameters.

        Returns
        -------
        `NDDataset`
            Dataset with shape (:term:`n_observations`, :term:`n_components`).

        Other Parameters
        ----------------
        n_components : `int`, optional
            The number of components to use for the reduction.

        See Also
        --------
        reconstruct : Alias of inverse_transform (Deprecated).

        """
        if not self._fitted:
            raise NotFittedError

        if isinstance(X_transform, NDDataset):
            X_transform = X_transform.data

        elif X_transform is None:
            X_transform = self.transform(**kwargs).data

        if isinstance(Y_transform, NDDataset):
            Y_transform = Y_transform.data

        elif Y_transform is None and both is True:
            Y_transform = self.transform(**kwargs).data

        if Y_transform is None:
            return self._inverse_transform(X_transform)
        X, Y = self._inverse_transform(X_transform, X_transform)
        return X, Y

    def fit_transform(self, X, Y, both=False):
        r"""
        Fit the model with `X` and `Y` and apply the dimensionality reduction on `X` and optionally on `Y`.

        Parameters
        ----------
        X : `NDDataset` or :term:`array-like` of shape (:term:`n_observations`, :term:`n_features`)
            Training data.
        Y : `NDDataset` or :term:`array-like` of shape (:term:`n_observations`, :term:`n_features`)
            Training data.
        both : `bool`, optional
            Whether to also apply the dimensionality reduction to Y when neither X nor Y are provided.

        Returns
        -------
        `NDDataset` or tuple of `NDDataset`
            Transformed data.

        """
        try:
            result = self.fit(X, Y).transform(X, Y, both=both)
            # fit_transform should return only x_scores by default (not a tuple)
            if both:
                return result
            # result could be a tuple from _transform - return only x_scores
            if isinstance(result, tuple):
                return result[0]
            return result
        except NotFittedError:
            # If transform failed, return None
            return None

    def plot_parity(
        self,
        Y=None,
        Y_hat=None,
        *,
        ax=None,
        clear=True,
        show=True,
        **kwargs,
    ):
        r"""
        Plot the predicted (:math:`\hat{Y}`) vs measured (:math:`Y`) values.

        :math:`Y` and :math:`\hat{Y}` can be passed as arguments. If not,
        the `Y` attribute is used for :math:`\hat{Y}` computed by
        the `predict` method.

        Parameters
        ----------
        Y : `NDDataset`, optional
            Measured values. If not provided, uses ``self.Y`` and computes
            ``Y_hat`` via ``self.predict(self.X)``.
        Y_hat : `NDDataset`, optional
            Predicted values. If ``Y`` is provided, ``Y_hat`` must also be
            provided as computed externally.
        ax : `~matplotlib.axes.Axes`, optional
            Axes to plot on. If None, a new figure is created.
        clear : `bool`, optional
            Whether to clear the axes before plotting. Default: True.
            Only used when ``ax`` is provided.
        show : `bool`, optional
            Whether to display the figure. Default: True.
        **kwargs : keyword arguments, optional
            Additional keyword arguments passed to
            `~matplotlib.axes.Axes.scatter`. Includes ``s``, ``c``, ``marker``,
            ``cmap``, ``norm``, ``vmin``, ``vmax``, ``alpha``, ``linewidths``,
            ``edgecolors``, ``plotnonfinite``.

        Returns
        -------
        `~matplotlib.axes.Axes`
            Matplotlib axes containing the parity plot.

        See Also
        --------
        parityplot : Deprecated alias for this method.
        """
        if Y is None:
            Y = self.Y
            if Y_hat is None:
                Y_hat = self.predict(self.X)
        elif Y_hat is None:
            raise ValueError(
                "If Y is provided, an externally computed Y_hat dataset "
                "must be also provided.",
            )

        from spectrochempy.plotting.composite.parity import plot_parity as _plot_parity

        return _plot_parity(Y, Y_hat, ax=ax, clear=clear, show=show, **kwargs)

    # Backward compatibility alias
    @deprecated(replace="plot_parity", removed="0.12")
    def parityplot(
        self, Y=None, Y_hat=None, *, ax=None, clear=True, show=True, **kwargs
    ):
        """
        Backward-compatible alias for :meth:`plot_parity`. Deprecated.

        Returns
        -------
        `~matplotlib.axes.Axes`
            Matplotlib axes containing the parity plot.
        """
        return self.plot_parity(Y, Y_hat, ax=ax, clear=clear, show=show, **kwargs)


# ======================================================================================
# Base class LinearRegressionAnalysis
# ======================================================================================
class LinearRegressionAnalysis(AnalysisConfigurable):
    # ----------------------------------------------------------------------------------
    # Configuration parameters (mostly defined in subclass
    # as they depend on the model estimator)
    # ----------------------------------------------------------------------------------
    fit_intercept = tr.Bool(
        default_value=True,
        help="Whether to calculate the `intercept` for this model. If set to `False`, "
        "no `intercept` will be used in calculations (*i.e.,* data is expected to be "
        "centered).",
    ).tag(config=True)

    positive = tr.Bool(
        default_value=False,
        help=r"When set to `True` , forces the coefficients (`coef`) "
        r"to be positive.",
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

        # initialize sklearn LinearRegression
        self._linear_regression = linear_model.LinearRegression(
            fit_intercept=self.fit_intercept,
            n_jobs=None,  # not used for the moment (XXX: should we add this?)
            positive=self.positive,
        )

        # unlike decomposition methods, we output ndarray when the input
        # is not a dataset
        self._output_type = "ndarray"

    # ----------------------------------------------------------------------------------
    # Private validation and default getter methods
    # ----------------------------------------------------------------------------------
    @tr.validate("_Y")
    def _Y_validate(self, proposal):
        # validation of the _Y attribute: fired when self._Y is assigned
        Y = proposal.value

        # we need a dataset or a list of NDDataset
        return self._make_dataset(Y)

    @property
    def _Y_is_missing(self):
        # check whether or not Y has been already defined
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
        return self._linear_regression.fit(X, Y, sample_weight=sample_weight)

    # ----------------------------------------------------------------------------------
    # Public methods
    # ----------------------------------------------------------------------------------
    def fit(self, X, Y=None, sample_weight=None):
        r"""
        Fit linear model.

        Parameters
        ----------
        X : `NDDataset` or :term:`array-like` of shape (:term:`n_observations`,:term:`n_features`)
            Training data, where `n_observations` is the number of observations
            and `n_features` is the number of features.
        Y : :term:`array-like` of shape (:term:`n_observations`,) or (:term:`n_observations`,:term:`n_targets`)
            Target values. Will be cast to `X`'s dtype if necessary.
        sample_weight : :term:`array-like` of shape (:term:`n_observations`,), default: `None`
            Individual weights for each observation.

        Returns
        -------
        self
            Returns the instance itself.

        """
        self._fitted = False  # reiniit this flag

        # store if the original input type is a dataset (or at least a subclass instance
        # of NDArray)
        self._is_dataset = isinstance(X, NDArray)

        def _make2D(X):
            # For regression analysis we need X as a NDDataset with two dimensions
            # IF X is 1D, then we add a dimension at the end.
            X = NDDataset(X)
            if X.ndim == 1:
                coordset = X.coordset
                X._data = X._data[:, np.newaxis]
                if np.any(X.mask):
                    X._mask = X._mask[:, np.newaxis]
                X.dims = ["x", "a"]
                coordx = coordset[0] if coordset is not None else None
                X.set_coordset(x=coordx, a=None)
            return X

        # fire the X and Y validation and preprocessing.
        if Y is not None:
            self._X = _make2D(X)
            self._Y = Y
        else:
            # X should contain the X and Y information (X being the coord and Y the data)
            if X.coordset is None:
                raise ValueError(
                    "The passed argument must have a x coordinates,"
                    "or X input and Y target must be passed separately",
                )
            self._X = _make2D(X.coord(0))
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

        # if the process was successful,_fitted is set to True so that other method which
        # needs fit will be possibly used.
        self._fitted = True
        return self

    @property
    def Y(self):
        """Return the `Y` input dataset."""
        # We use Y property only to show this information to the end user. Internally
        # we use _Y attribute to refer to the input data
        if self._Y_is_missing:
            raise NotFittedError
        Y = self._Y
        if self._is_dataset or self._output_type == "NDDataset":
            return Y
        return np.asarray(Y)

    @property
    def coef(self):
        r"""
        Estimated coefficients for the linear regression problem.

        If multiple targets are passed during the fit (Y 2D), this is a 2D array of
        shape (:term:`n_targets`, :term:`n_features`), while if only one target
        is passed, this is a 1D array of length :term:`n_features`.
        """
        if self._linear_regression.coef_.size == 1:
            # this is the result of the single equation, so only one value
            # should be returned
            if self._linear_regression.coef_.ndim == 0:
                A = float(self._linear_regression.coef_)
            else:
                A = float(self._linear_regression.coef_[0])
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
        r"""
        Return a float or an array of shape (:term:`n_targets`,).

        Independent term in the linear model. Set to ``0.0`` if `fit_intercept` is `False`.
        If `Y` has units, then `intercept` has the same units.
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
        r"""
        Predict features using the linear model.

        Parameters
        ----------
        X : `NDDataset` or :term:`array-like` matrix, shape (:term:`n_observations`,:term:`n_features`)
            Observations. If `X` is not set, the input `X` for `fit` is used.

        Returns
        -------
        `~spectrochempy.core.dataset.nddataset.NDDataset`
            Predicted values (object of type of the input) using a ahape (:term:`n_observations`,).

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
        r"""
        Return the coefficient of determination of the prediction.

        The coefficient of determination :math:`R^2` is defined as
        :math:`(1 - \frac{u}{v})` , where :math:`u` is the residual
        sum of squares ``((y_true - y_pred)** 2).sum()`` and :math:`v`
        is the total sum of squares ``((y_true - y_true.mean()) ** 2).sum()`` .
        The best possible score is ``1.0`` and it can be negative (because the
        model can be arbitrarily worse). A constant model that always predicts
        the expected value of `Y`, disregarding the input features, would get
        a :math:`R^2` score of 0.0.

        Parameters
        ----------
        X : `NDDataset` or :term:`array-like` of shape (:term:`n_observations`, :term:`n_features`)
            Test samples.

        Y : `NDDataset` or :term:`array-like` of shape (:term:`n_observations`,)
            True values for `X`.

        sample_weight : :term:`array-like` of shape (:term:`n_observations`,), default: `None`
            Sample weights.

        Returns
        -------
        `float`
            :math:`R^2` of `predict` (`X` ) wrt. `Y` .

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
