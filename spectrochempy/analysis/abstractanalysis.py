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

import numpy as np
import traitlets as tr
from traittypes import Array

from spectrochempy.analysis._analysisutils import (
    NotFittedError,
    _wrap_ndarray_output_to_nddataset,
)
from spectrochempy.core import app, set_loglevel
from spectrochempy.core.common.meta import Meta
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.utils import MASKED, NOMASK, exceptions
from spectrochempy.utils.traits import MetaConfigurable


class AnalysisConfigurable(MetaConfigurable):
    """
    Abstract class to write analysis estimators.

    Subclass this to get a minimal structure
    """

    name = tr.Unicode(help="name of the implemented model")
    # name must be defined in subclass with the name of the model: PCA, MCRALS, ...
    description = tr.Unicode(help="optional description of the implemented model")

    # ----------------------------------------------------------------------------------
    # Runtime Parameters
    # ----------------------------------------------------------------------------------
    _warm_start = tr.Bool(False, help="If True previous execution state " "is reused")
    _fitted = tr.Bool(False, help="False if the model was not yet fitted")
    _masked_rc = tr.Tuple(allow_none=True, help="List of masked rows and columns")

    _X = tr.Instance(NDDataset, allow_none=True, help="Data to fit a model")
    _X_mask = Array(allow_none=True, help="mask information of the " "input data")
    _X_preprocessed = Array(help="preprocessed inital X input data")
    _shape = tr.Tuple(help="original shape of the data, before any transformation")
    _outfit = tr.Any(help="the output of the _fit method - generally a tuple")
    _copy = tr.Bool(default_value=True, help="If True passed X data are copied")

    # ----------------------------------------------------------------------------------
    # Configuration parameters (mostly defined in subclass
    # as they depends on the model estimator)
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
            masked_columns = np.all(mask, axis=-2)
            masked_rows = np.all(mask, axis=-1)
        else:
            masked_columns = np.zeros(self._shape[-1], dtype=bool)
            masked_rows = np.zeros(self._shape[-2], dtype=bool)
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

        # store the mask because it will be destroyed
        self._X_mask = X._mask

        # remove masked rows and columns
        masked_rows, masked_columns = self._get_masked_rc(X._mask)

        data = X.data[:, ~masked_columns]
        data = data[~masked_rows]

        # destroy the mask
        X._mask = NOMASK

        # return the modified X dataset
        X.data = data
        return X

    def _restore_masked_data(self, D, axis=-1):
        # by default we restore columns, put axis=0 to restore rows instead
        # Note that it is very important to use here the ma version of zeros
        # array constructor or both if both axis should be restored
        if not np.any(self._X_mask):
            # return it inchanged as wa had no mask originally
            return D

        rowsize, colsize = self._shape
        masked_rows, masked_columns = self._get_masked_rc(self._X_mask)
        M, N = D.shape

        Dtemp = None
        # Put back masked columns in D
        # ----------------------------
        if axis == -1 or axis == 1 or axis == "both":  # and D.shape[0] == rowsize:
            if np.any(masked_columns):
                Dtemp = np.ma.zeros((M, colsize))  # note np.ma, not np.
                Dtemp[:, ~masked_columns] = D
                Dtemp[:, masked_columns] = MASKED
                D = Dtemp

        # Put back masked rows in D
        # -------------------------
        if axis == -2 or axis == 0 or axis == "both":  # and D.shape[1] == colsize:
            if np.any(masked_rows):
                Dtemp = np.ma.zeros((rowsize, N))
                Dtemp[~masked_rows] = D
                Dtemp[masked_rows] = MASKED
                D = Dtemp

        # if Dtemp is None and np.any(self._X_mask):
        #     raise IndexError("Can not restore mask. Please check the given index")

        # return the D array with restored masked data
        return D

    # ----------------------------------------------------------------------------------
    # Private validation and default getter methods
    # ----------------------------------------------------------------------------------
    @tr.default("name")
    def _name_default(self):
        # this ensure a name has been defined for the subclassed model estimators
        # or an error is returned
        raise NameError("The name of the object was not defined.")

    @tr.default("_X")
    def _X_default(self):
        raise NotFittedError

    @tr.validate("_X")
    def _X_validate(self, proposal):
        # validation fired when self._X is assigned
        X = proposal.value
        # we need a dataset with eventually  a copy of the original data (default being
        # to copy them)
        X = self._make_dataset(X)
        # as in fit methods we often use np.linalg library, we cannot handle directly
        # masked data (so we remove them here and they will be restored at the end of
        # the process during transform or inverse transform methods
        # store the original shape as it will be eventually modified
        self._shape = X.shape
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

    # ----------------------------------------------------------------------------------
    # Private methods that should be most of the time overloaded in subclass
    # ----------------------------------------------------------------------------------
    @tr.observe("_X")
    def _preprocess_as_X_changed(self, change):
        # to be optionally replaced by user defined function (with the same name)
        X = change.new
        # .... preprocessing as scaling, centering, ...
        # set a np.ndarray
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
            X.data = self._restore_masked_data(X.data, axis="both")
        return X

    def fit(self, X, Y=None, **kwargs):
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
            Returns the instance itself.
        """
        self._fitted = False  # reiniit this flag

        # fire the X and eventually Y validation and preprocessing.
        # X and Y are epected to be resp. NDDataset and NDDataset or list of NDDataset.
        self._X = self._make_dataset(X)
        if Y is not None:
            self._Y = self._make_dataset(Y)

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


class DecompositionAnalysisConfigurable(AnalysisConfigurable):
    """
    Abstract class to write analysis decomposition model such as PCA, ...

    Subclass this to get a minimal structure
    """

    # THis class is subclass Analysisonfigurable so we define only additional
    # attributes and methods necessary for decomposition model.

    # ----------------------------------------------------------------------------------
    # Runtime Parameters (in addition to those of AnalysisConfigurable)
    # ----------------------------------------------------------------------------------
    _Y = tr.Union(
        (
            tr.Tuple(tr.Instance(NDDataset), tr.Instance(NDDataset)),
            tr.Instance((NDDataset)),
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

    @_wrap_ndarray_output_to_nddataset(
        keepunits=False, keeptitle=False, typex="components"
    )
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
            Thhe number of compopnents is eventually the one specified or determined
            in the fit process.

        Returns
        -------
        NDDataset(n_observations, n_components)
            Projection of X in the first principal components, where `n_observations`
            is the number of observations and `n_components` is the number of the components.
        """
        if not self._fitted:
            raise NotFittedError()

        # fire the validation and preprocessing
        self._X = X if X is not None else self.X

        # get the processed ndarray data
        newX = self._X_preprocessed

        X_transform = self._transform(newX)

        # slice according to n_components
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
            provided, but a starsform of X provided in fit is performed first.
        **kwargs
            Additional keyword parameters. See Other Parameters

        Other Parameters
        ----------------
        n_components : int, optional
            The number of components to use for the inverse_transformation. If not given
            The number of compopnents is eventually the one specified or determined
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
        **kwargs :
            Additional optional keywords parameters as for the `transform` method.

        Returns
        -------
        NDDataset(n_observations, n_components)
        """
        self.fit(X, Y, **kwargs)
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

    @_wrap_ndarray_output_to_nddataset(
        keepunits=None, keeptitle=False, typey="components"
    )
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
    @_wrap_ndarray_output_to_nddataset(keepunits=None, keeptitle=True, typex="feature")
    def components(self):
        """
        Return a NDDataset with the components in feature space.

        See Also
        --------
        get_components: retrieve only the specified number of components
        """
        return self._get_components()

    # ----------------------------------------------------------------------------------
    # Plot methods
    # ----------------------------------------------------------------------------------
    def plotmerit(self, X, X_hat, **kwargs):
        """
        Plots the input dataset, reconstructed dataset and residuals.

        Parameters
        ----------
        **kwargs
            optional "colors" argument: tuple or array of 3 colors
            for :math:`X`, :math:`\hat X` and :math:`E`.

        Returns
        -------
        ax
            subplot.
        """
        if not self._fitted:
            raise NotFittedError(
                "The fit method must be used " "before using this method"
            )

        colX, colXhat, colRes = kwargs.pop("colors", ["blue", "green", "red"])

        res = X - X_hat
        ax = X.plot()
        ma = max(X.max(), X_hat.max())
        if X.x is not None and X.x.data is not None:
            ax.plot(X.x.data, X_hat.T.masked_data - ma, color=colXhat)
            ax.plot(X.x.data, res.T.masked_data - 1.2 * ma, color=colRes)
        else:
            ax.plot(X_hat.T.masked_data - ma, color=colXhat)
            ax.plot(res.T.masked_data - 1.2 * ma, color=colRes)
        ax.autoscale(enable=True, axis="y")
        ax.set_title(f"{self.name} plot of merit")
        ax.yaxis.set_visible(False)
        return ax
