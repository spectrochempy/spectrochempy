# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
This module implements the base abstract class to define estimators such as PCA, ...
"""

import logging
import warnings
from copy import copy

import numpy as np
import traitlets as tr
from sklearn import linear_model

from spectrochempy.analysis._analysisutils import (
    NotFittedError,
    _wrap_ndarray_output_to_nddataset,
)
from spectrochempy.core import app, set_loglevel
from spectrochempy.core.dataset.baseobjects.meta import Meta
from spectrochempy.core.dataset.baseobjects.ndarray import NDArray
from spectrochempy.core.dataset.coordset import CoordSet
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.extern.traittypes import Array
from spectrochempy.utils import exceptions
from spectrochempy.utils.constants import MASKED, NOMASK
from spectrochempy.utils.docstrings import _docstring
from spectrochempy.utils.traits import MetaConfigurable, NDDatasetType


# ======================================================================================
# Base class AnalysisConfigurable
# ======================================================================================
class AnalysisConfigurable(MetaConfigurable):
    """
    Abstract class to write analysis estimators.

    Subclass this to get a minimal structure

    Parameters
    ----------
    log_level : ["INFO", "DEBUG", "WARNING", "ERROR"], optional, default:"WARNING"
        The log level at startup
    copy : bool, optional, default:True
        Whether to copy input data to avoid overriding.
    config : Config object, optional
        By default the configuration is determined by the object configuration
        file in the configuration directory. A traitlets.config.Config() object can
        eventually be used here.
    warm_start : bool, optional, default:False
        When fitting repeatedly on the same dataset, but for multiple
        parameter values (such as to find the value maximizing performance),
        it may be possible to reuse previous model learned from the previous parameter
        value, saving time.
        When warm_start is true, the existing fitted model attributes is used to
        initialize the new model in a subsequent call to fit.
    **kwargs
        Optional configuration  parameters.
    """

    name = tr.Unicode(help="name of the implemented model")
    # name must be defined in subclass with the name of the model: PCA, MCRALS, ...
    description = tr.Unicode(help="optional description of the implemented model")

    # get doc sections for reuse
    _docstring.get_sections(_docstring.dedent(__doc__), base="AnalysisConfigurable")

    # ----------------------------------------------------------------------------------
    # Runtime Parameters
    # ----------------------------------------------------------------------------------
    _warm_start = tr.Bool(False, help="If True previous execution state is reused")
    _fitted = tr.Bool(False, help="False if the model was not yet fitted")
    _masked_rc = tr.Tuple(allow_none=True, help="List of masked rows and columns")
    _X = NDDatasetType(allow_none=True, help="Data to fit a model")
    _X_mask = Array(allow_none=True, help="Mask information of the input X data")
    _X_preprocessed = Array(help="Preprocessed inital input X data")
    _X_shape = tr.Tuple(
        help="Original shape of the input X data, " "before any transformation"
    )
    _X_coordset = tr.Instance(CoordSet, allow_none=True)
    _is_dataset = tr.Bool(help="True if the input X data is a NDDataset")
    _outfit = tr.Any(help="the output of the _fit method - generally a tuple")
    _copy = tr.Bool(default_value=True, help="If True, input X data are copied")
    _output_type = tr.Enum(
        ["NDDataset", "ndarray"],
        default_value="NDDataset",
        help="Whether the output is a NDDataset or a ndarray",
    )

    # ----------------------------------------------------------------------------------
    # Configuration parameters (mostly defined in subclass
    # as they depend on the model estimator)
    # ----------------------------------------------------------------------------------

    # write traits like e.g.,  A = Unicode("A", help='description").tag(config=True)

    # ----------------------------------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------------------------------
    def __init__(
        self,
        *,
        log_level=logging.WARNING,
        config=None,
        warm_start=False,
        copy=True,
        **kwargs,
    ):

        # call the super class for initialisation
        super().__init__(section=self.name, config=config, parent=app)

        # set log_level of the console report
        set_loglevel(log_level)

        # initial configuration
        # reset to default if not warm_start
        defaults = self.parameters(default=True)
        configkw = {} if warm_start else defaults
        # eventually take parameters form kwargs
        configkw.update(kwargs)

        for k, v in configkw.items():
            if k in defaults:
                setattr(self, k, v)
            else:
                raise KeyError(
                    f"'{k}' is not a valid configuration parameters. "
                    f"Use the method `parameters()` to check the current "
                    f"allowed parameters and their current value."
                )

        # if warm start we can use the previous fit as starting profiles.
        self._warm_start = warm_start
        if not warm_start:
            # We should not be able to use any methods requiring fit results
            # until the fit method has been executed
            self._fitted = False

        # copy passed data
        self._copy = copy

    # ----------------------------------------------------------------------------------
    # Private methods
    # ----------------------------------------------------------------------------------
    def _make_dataset(self, d):
        # Transform an array-like object to NDDataset (optionally copy data)
        # or a list of array-like to alist of NDQataset
        if d is None:
            return
        if isinstance(d, (tuple, list)):
            d = [self._make_dataset(item) for item in d]
        elif not isinstance(d, NDDataset):
            d = NDDataset(d, copy=self._copy)
        elif self._copy:
            d = d.copy()
        return d

    def _get_masked_rc(self, mask):
        if np.any(mask):
            masked_columns = np.all(mask, axis=-2)  # if mask.ndim == 2 else None
            masked_rows = np.all(mask, axis=-1)
        else:
            masked_columns = np.zeros(self._X_shape[-1], dtype=bool)
            masked_rows = np.zeros(self._X_shape[-2], dtype=bool)
        return masked_rows, masked_columns

    def _remove_masked_data(self, X):
        # Retains only valid rows and columns
        # -----------------------------------
        # unfortunately, the implementation of linalg library
        # doesn't support numpy masked arrays as input. So we will have to
        # remove the masked values ourselves

        # the following however assumes that entire rows or columns are masked,
        # not only some individual data (if this is what you wanted, this
        # will fail)

        if not hasattr(X, "mask"):
            return X

        # remove masked rows and columns
        masked_rows, masked_columns = self._get_masked_rc(X._mask)

        Xc = X[:, ~masked_columns]
        Xrc = Xc[~masked_rows]

        # destroy the mask
        Xrc._mask = NOMASK

        # return the modified X dataset
        return Xrc

    def _restore_masked_data(self, D, axis=-1):
        # by default, we restore columns, put axis=0 to restore rows instead
        # Note that it is very important to use here the ma version of zeros
        # array constructor or both if both axis should be restored
        if not np.any(self._X_mask):
            # return it inchanged as wa had no mask originally
            return D

        rowsize, colsize = self._X_shape
        masked_rows, masked_columns = self._get_masked_rc(self._X_mask)

        Dtemp = None
        if D.ndim == 2:
            # Put back masked columns in D
            # ----------------------------
            M, N = D.shape
            if axis == "both":  # and D.shape[0] == rowsize:
                if np.any(masked_columns) or np.any(masked_rows):
                    Dtemp = np.ma.zeros((rowsize, colsize))  # note np.ma, not np.
                    Dtemp[~self._X_mask] = D.data.flatten()
                    Dtemp[self._X_mask] = MASKED
                    D.data = Dtemp
                    try:
                        D.coordset[D.dims[-1]] = self._X_coordset[D.dims[-1]]
                        D.coordset[D.dims[-2]] = self._X_coordset[D.dims[-2]]
                    except TypeError:
                        # probably no coordset
                        pass
            elif axis == -1 or axis == 1:
                if np.any(masked_columns):
                    Dtemp = np.ma.zeros((M, colsize))  # note np.ma, not np.
                    Dtemp[:, ~masked_columns] = D
                    Dtemp[:, masked_columns] = MASKED
                    D.data = Dtemp
                    try:
                        D.coordset[D.dims[-1]] = self._X_coordset[D.dims[-1]]
                    except TypeError:
                        # probably no coordset
                        pass

            # Put back masked rows in D
            # -------------------------
            elif axis == -2 or axis == 0:
                if np.any(masked_rows):
                    Dtemp = np.ma.zeros((rowsize, N))
                    Dtemp[~masked_rows] = D
                    Dtemp[masked_rows] = MASKED
                    D.data = Dtemp
                    try:
                        D.coordset[D.dims[-2]] = self._X_coordset[D.dims[-2]]
                    except TypeError:
                        # probably no coordset
                        pass
        elif D.ndim == 1:
            # we assume here that the only case it happens is for array as explained
            # variance so that we deal with masked rows
            if np.any(masked_rows):
                Dtemp = np.ma.zeros((rowsize,))  # note np.ma, not np.
                Dtemp[~masked_rows] = D
                Dtemp[masked_rows] = MASKED
                D.data = Dtemp

        elif D.ndim == 3:
            # CASE of IRIS for instance

            # Put back masked columns in D
            # ----------------------------
            J, M, N = D.shape
            if axis == -1 or axis == 2:
                if np.any(masked_columns):
                    Dtemp = np.ma.zeros((J, M, colsize))  # note np.ma, not np.
                    Dtemp[..., ~masked_columns] = D
                    Dtemp[..., masked_columns] = MASKED
                    D.data = Dtemp
                    try:
                        D.coordset[D.dims[-1]] = self._X_coordset[D.dims[-1]]
                    except TypeError:
                        # probably no coordset
                        pass

        # return the D array with restored masked data
        return D

    # ----------------------------------------------------------------------------------
    # Private validation and default getter methods
    # ----------------------------------------------------------------------------------
    @tr.default("name")
    def _name_default(self):
        # this ensures a name has been defined for the subclassed model estimators
        # or an error is returned
        raise NameError("The name of the object was not defined.")

    @tr.default("_X")
    def _X_default(self):
        raise NotFittedError

    @tr.validate("_X")
    def _X_validate(self, proposal):
        # validation fired when self._X is assigned
        X = proposal.value

        # for the following we need X with two dimensions
        # So let's generate the un-squeezed X
        if X.ndim == 1:
            coordset = X.coordset
            X._data = X._data[np.newaxis]
            if np.any(X.mask):
                X._mask = X._mask[np.newaxis]
            X.dims = ["y", "x"]  # "y" is the new dimension
            coordx = coordset[0] if coordset is not None else None
            X.set_coordset(x=coordx, y=None)

        # as in fit methods we often use np.linalg library, we cannot handle directly
        # masked data (so we remove them here and they will be restored at the end of
        # the process during transform or inverse transform methods
        # store the original shape as it will be eventually modified as welle- as the
        # original coordset
        self._X_shape = X.shape
        # store the mask because it may be destroyed
        self._X_mask = X._mask.copy()
        # and the original coordset
        self._X_coordset = copy(X._coordset)

        # remove masked data and return modified dataset
        X = self._remove_masked_data(X)
        return X

    @property
    def _X_is_missing(self):
        # check wether or not X has been already defined
        try:
            if self._X is None:
                return True
        except NotFittedError:
            return True
        return False

    # ----------------------------------------------------------------------------------
    # Private methods that should be most of the time overloaded in subclass
    # ----------------------------------------------------------------------------------
    @tr.observe("_X")
    def _preprocess_as_X_changed(self, change):
        # to be optionally replaced by user defined function (with the same name)
        X = change.new
        # .... preprocessing as scaling, centering, ... must return a ndarray with
        #  same shape a X.data

        # Set a X.data by default
        self._X_preprocessed = X.data

    def _fit(self, X, Y=None):
        #  Intended to be replaced in the subclasses by user defined function
        #  (with the same name)
        raise NotImplementedError("fit method has not yet been implemented")

    # ----------------------------------------------------------------------------------
    # Public methods
    # ----------------------------------------------------------------------------------
    @property
    def X(self):
        """
        Return the X input dataset (eventually modified by the model)
        """
        if self._X_is_missing:
            raise NotFittedError
        # We use X property only to show this information to the end user. Internally
        # we use _X attribute to refer to the input data
        X = self._X.copy()
        if np.any(self._X_mask):
            # restore masked row and column if necessary
            # X.data = self._restore_masked_data(X.data, axis="both")
            X = self._restore_masked_data(X, axis="both")
        if self._is_dataset or self._output_type == "NDDataset":
            return X
        else:
            return np.asarray(X)

    def fit(self, X, Y=None):
        """
        Fit the model with X and optional Y data

        Parameters
        ----------
        X : NDDataset or array-like of shape (n_observations, n_features)
            Training data, where `n_observations` is the number of observations
            and `n_features` is the number of features.

        Y : array_like, optional
            For example Y is not used in PCA, but corresponds to the guess profiles in
            MCRALS

        Returns
        -------
        self : object
            Returns the fitted instance itself.
        """
        self._fitted = False  # reiniit this flag

        # fire the X and eventually Y validation and preprocessing.
        # X and Y are expected to be resp. NDDataset and NDDataset or list of NDDataset.
        self._X = X  # self._make_dataset(X)
        if Y is not None:
            self._Y = Y  # self._make_dataset(Y)

        # _X_preprocessed has been computed when X was set, as well as _Y_preprocessed.
        # At this stage they should be simple ndarrays
        newX = self._X_preprocessed
        newY = self._Y_preprocessed if Y is not None else None

        # call to the actual _fit method (overloaded in the subclass)
        # warning : _fit must take ndarray arguments not NDDataset arguments.
        # when method must return NDDataset from the calculated data,
        # we use the decorator _wrap_ndarray_output_to_nddataset, as below or in the PCA
        # model for example.
        self._outfit = self._fit(newX, newY)

        # if the process was succesful,_fitted is set to True so that other method which
        # needs fit will be possibly used.
        self._fitted = True
        return self

    # ----------------------------------------------------------------------------------
    # Public utility functions
    # ----------------------------------------------------------------------------------
    def parameters(self, default=False):
        """
        Return current or default configuration values

        Parameters
        ----------
        default : Bool, optional, default: False
            If 'default' is True, the default parameters are returned,
            else the current values.

        Returns
        -------
        dict
        """
        d = Meta()
        if not default:
            d.update(self.trait_values(config=True))
        else:
            d.update(self.trait_defaults(config=True))
        return d

    def reset(self):
        """
        Reset configuration to default
        """
        for k, v in self.parameters(default=True).items():
            setattr(self, k, v)

    @classmethod
    @property
    def help(cls):
        """
        Return a description of all configuration parameters with their default value
        """
        return cls.class_config_rst_doc()

    @property
    def log(self):
        """
        Logs output.
        """
        # A string handler (#2) is defined for the Spectrochempy logger,
        # thus we will return it's content
        return app.log.handlers[2].stream.getvalue().rstrip()


# ======================================================================================
# Base class DecompositionAnalysis
# ======================================================================================
class DecompositionAnalysis(AnalysisConfigurable):
    """
    Abstract class to write analysis decomposition model such as PCA, ...

    Subclass this to get a minimal structure
    """

    # This class is subclass AnalysisConfigurable, so we define only additional
    # attributes and methods necessary for decomposition model.

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
    _Y_preprocessed = Array(help="preprocessed Y")
    _n_components = tr.Integer(help="""The actual number of components.""")
    _components = Array(help="the array of (n_components, n_features) components")

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

    def _transform(self, *args, **kwargs):
        # to be overriden in subclass such as PCA, MCRALS, ...
        raise NotImplementedError("transform has not yet been implemented")

    def _inverse_transform(self, *args, **kwargs):
        # to be overriden in subclass such as PCA, MCRALS, ...
        raise NotImplementedError("inverse_transform has not yet been implemented")

    def _get_components(self, n_components=None):
        # to be overriden in subclass such as PCA, MCRALS, ...
        raise NotImplementedError("get_components has not yet been implemented")

    # ----------------------------------------------------------------------------------
    # Public methods
    # ----------------------------------------------------------------------------------
    @property
    def Y(self):
        """
        Return the Y input dataset or a list of dataset

        This describes for example starting Concentration and Spectra in MCRALS
        """
        # We use Y property only to show this information to the end user. Internally
        # we use _Y attribute to refer to the input data
        if self._Y_is_missing:
            raise NotFittedError
        Y = self._Y
        return Y

    @_wrap_ndarray_output_to_nddataset(units=None, title=None, typex="components")
    def transform(self, X=None, **kwargs):
        """
        Apply dimensionality reduction to X.

        X is projected on the first principal components previously extracted
        from a training set.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_features), optional
            New data, where `n_observations` is the number of observations
            and `n_features` is the number of features.
            if not provided, the input dataset of the fit method will be used.

        **kwargs
            Additional keyword parameters. See Other Parameters

        Other Parameters
        ----------------
        n_components : int, optional
            The number of components to use for the transformation. If not given
            The number of compopnents is eventually the one specified or determined
            in the fit process.

        Returns
        -------
        NDDataset(n_observations, n_components)
            Projection of X in the first principal components, where `n_observations`
            is the number of observations and `n_components` is the number of the components.
        """
        if not self._fitted:
            raise NotFittedError()

        # Fire the validation and preprocessing
        self._X = X if X is not None else self.X

        # Get the processed ndarray data
        newX = self._X_preprocessed

        X_transform = self._transform(newX)

        # Slice according to n_components
        n_components = kwargs.pop(
            "n_components", kwargs.pop("n_pc", self._n_components)
        )
        if n_components > self._n_components:
            warnings.warn(
                "The number of components required for reduction "
                "cannot be greater than the fitted model components : "
                f"{self._n_components}. We then use this latter value."
            )
        if n_components < self._n_components:
            X_transform = X_transform[:, :n_components]

        return X_transform

    @_wrap_ndarray_output_to_nddataset
    def inverse_transform(self, X_transform=None, **kwargs):
        """
        Transform data back to its original space.

        In other words, return an input `X_original` whose reduce/transform would be X.

        Parameters
        ----------
        X_transform : array-like of shape (n_observations, n_components), optional
            Reduced X data, where `n_observations` is the number of observations
            and `n_components` is the number of components. If X_transform is not
            provided, a transform of X provided in fit is performed first.
        **kwargs
            Additional keyword parameters. See Other Parameters

        Other Parameters
        ----------------
        n_components : int, optional
            The number of components to use for the inverse_transformation. If not given
            The number of components is eventually the one specified or determined
            in the fit process.

        Returns
        -------
        NDDataset(n_observations, n_features)
            Data with the original X shape
            eventually filtered by the reduce/transform operation.
        """
        if not self._fitted:
            raise NotFittedError(
                "The fit method must be used before using reconstruct/inverse_transform "
                "method"
            )

        # get optional n_components
        n_components = kwargs.pop(
            "n_components", kwargs.pop("n_pc", self._n_components)
        )
        if n_components > self._n_components:
            warnings.warn(
                "The number of components required for reduction "
                "cannot be greater than the fitted model components : "
                f"{self._n_components}. We then use this latter value."
            )

        if isinstance(X_transform, NDDataset):
            X_transform = X_transform.data
            if n_components > X_transform.shape[1]:
                warnings.warn(
                    "The number of components required for reduction "
                    "cannot be greater than the X_transform size : "
                    f"{X_transform.shape[1]}. We then use this latter value."
                )
        elif X_transform is None:
            X_transform = self.transform(**kwargs)

        X = self._inverse_transform(X_transform)

        return X

    def fit_transform(self, X, Y=None, **kwargs):
        """
        Fit the model with X and apply the dimensionality reduction on X.

        Parameters
        ----------
        X : NDDataset
            Input dataset of shape (n_observation, n_feature) to fit
        Y : array_like, optional
            For example Y is not used in PCA, but corresponds to the guess profiles in
            MCRALS
        **kwargs :
            Additional optional keywords parameters as for the `transform` method.

        Returns
        -------
        NDDataset(n_observations, n_components)
        """
        self.fit(X, Y)
        X_transform = self.transform(X, **kwargs)
        return X_transform

    @exceptions.deprecated(replace="transform")
    def reduce(self, X=None, **kwargs):
        return self.transform(X, **kwargs)

    reduce.__doc__ = transform.__doc__

    @exceptions.deprecated(replace="inverse_transform")
    def reconstruct(self, X_transform=None, **kwargs):
        return self.inverse_transform(self, X_transform, **kwargs)

    reconstruct.__doc__ = inverse_transform.__doc__

    @exceptions.deprecated(replace="fit_transform")
    def fit_reduce(self, X, Y=None, **kwargs):
        return self.fit_transform(X, Y, **kwargs)

    fit_reduce.__doc__ = fit_transform.__doc__

    @_wrap_ndarray_output_to_nddataset(units=None, title=None, typey="components")
    def get_components(self, n_components=None):
        """
        Returns the components dataset: (selected n_components, n_features).

        Parameters
        ----------
        n_components : int, optional
            The number of components to keep in the output nddataset
            If None, all calculated components are eturned.

        Returns
        -------
        NDDataset
            A nddataset with shape (n_components, n_features).
        """
        if n_components is None or n_components > self._n_components:
            n_components = self._n_components

        # we call the specific _get_components method defined in subclasses
        components = self._get_components()[:n_components]

        return components

    @property
    @_wrap_ndarray_output_to_nddataset(units=None, title="keep", typey="components")
    def components(self):
        """
        Return a NDDataset with components in feature space (n_components, n_features).

        See Also
        --------
        get_components: retrieve only the specified number of components
        """
        return self._get_components()

    @property
    def n_components(self):
        """
        Return the number of components that were fitted.
        """
        if self._fitted:
            return self._n_components
        else:
            raise NotFittedError("n_components")

    # ----------------------------------------------------------------------------------
    # Plot methods
    # ----------------------------------------------------------------------------------
    @_docstring.get_sections(base="plotmerit")
    @_docstring.dedent
    def plotmerit(self, X, X_hat, **kwargs):
        """
        Plots the input dataset (X), reconstructed (X_hat) and residuals (E) datasets.

        Parameters
        ----------
        X : NDDataset
            Original dataset.
        X_hat : NDDataset
            Inverse_transform (reconstructed) dataset from a decomposition model.
        %(kwargs)s

        Other Parameters
        ----------------
        colors : tuple or array of 3 colors, optional, default: ["blue", "orange", "red"]
            Colors for :math:`X`, :math:`\hat X` and :math:`E`.
            in the case of 2D, The default colormap is used for X.
        offset : float, optional, default: None
            Specify the separation (%%) between the X, X_hat and E.
        nb_traces : int, optional
            Number of lines to display. Default is all
        **others : Other keywords parameters that are passed to
            the internal plot method of the X dataset.

        Returns
        -------
        mpl.Axe
        """
        if not self._fitted:
            raise NotFittedError(
                "The fit method must be used before using this method."
            )

        colX, colXhat, colRes = kwargs.pop("colors", ["blue", "orange", "red"])

        if X._squeeze_ndim == 1:
            # normally this was done before, but if needed.
            X = X.squeeze()
            X_hat = X_hat.squeeze()

        # Number of traces to keep
        nb_traces = kwargs.pop("nb_traces", "all")
        if X.ndim == 2 and nb_traces != "all":
            inc = int(X.shape[0] / nb_traces)
            X = X[::inc]
            X_hat = X_hat[::inc]

        res = X - X_hat

        # separation between traces
        offset = kwargs.pop("offset", 0)

        ma = max(X.max(), X_hat.max())
        mao = ma * offset / 100
        mad = ma * offset / 100 + ma / 10
        _ = (X - X.min()).plot(color=colX, **kwargs)
        _ = (X_hat - X_hat.min() - mao).plot(
            clear=False, ls="dashed", cmap=None, color=colXhat
        )
        ax = (res - res.min() - mad).plot(clear=False, cmap=None, color=colRes)

        #             color=colXhat)
        #     ax.plot(res.T.masked_data - 1.2 * ma,
        #             color=colRes)

        # if X.x is not None and X.x.data is not None:
        #     ax.plot(X.x.data, X_hat.T.masked_data - ma, '-',
        #             color=colXhat)
        #     ax.plot(X.x.data, res.T.masked_data - 1.2 * ma, '-',
        #             color=colRes)
        # else:
        #     ax.plot(X_hat.T.masked_data - ma,
        #             color=colXhat)
        #     ax.plot(res.T.masked_data - 1.2 * ma,
        #             color=colRes)
        ax.autoscale(enable=True, axis="y")
        ax.set_title(f"{self.name} plot of merit")
        ax.yaxis.set_visible(False)
        return ax


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
        self._output_type = "ndarray"

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

        def _make2D(X):
            # For regression analysis we need X as a NDDataset with two dimensions
            # IF X is 1D, then we add a dimension at the end.
            X = self._make_dataset(X)
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
                    "or X input and Y target must be passed separately"
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
