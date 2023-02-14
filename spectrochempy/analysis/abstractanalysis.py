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

from spectrochempy.core import app, set_loglevel
from spectrochempy.core.common.meta import Meta
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.utils import MASKED, exceptions
from spectrochempy.utils.traits import MetaConfigurable


class AnalysisConfigurable(MetaConfigurable):
    """
    Abstract class to write analysis estimators.

    Subclass this to get a minimal structure
    """

    name = tr.Unicode()
    description = tr.Unicode()

    # ----------------------------------------------------------------------------------
    # Runtime Parameters
    # ----------------------------------------------------------------------------------
    _warm_start = tr.Bool(False, help="If True previous execution state " "is reused")
    _fitted = tr.Bool(False, help="False if the model was not yet fitted")
    _masked_rc = tr.Tuple(allow_none=True, help="List of masked rows ans columns")

    _X = tr.Instance(NDDataset, allow_none=True, help="Data to fit an estimate")
    _X_mask = tr.instance(
        NDDataset, allow_none=True, help="mask information of the " "input data"
    )
    _shape = tr.Tuple(help="original shape of the data, before any transformation")

    # ----------------------------------------------------------------------------------
    # Configuration parameters (depends on the estimator)
    # ----------------------------------------------------------------------------------

    copy = tr.Bool(default_value=True, help="If True passed X data are copied").tag(
        config=True
    )

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
            if k in defaults.keys():
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

    # ----------------------------------------------------------------------------------
    # Data
    # ----------------------------------------------------------------------------------
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
        X._mask = None

        # return the modified X dataset
        X.data = data
        return X

    def _restore_masked_data(self, D, axis=-1):
        # by default we restore columns, put axis=0 to restore rows instead
        # Note that it is very important to use here the ma version of zeros
        # array constructor or both if both axis should be restored
        if self._X_mask is None:
            # return it inchanged as wa had no mask originally
            return D

        rowsize, colsize = self._shape
        masked_rows, masked_columns = self._get_masked_rc(self._X_mask)
        M, N = D.shape

        # Put back masked columns in D
        # ----------------------------
        if (axis == -1 or axis == 1) and D.shape[0] == rowsize:
            if np.any(masked_columns):
                Dtemp = np.ma.zeros((M, colsize))  # note np.ma, not np.
                Dtemp[:, ~masked_columns] = D
                Dtemp[:, masked_columns] = MASKED
                D = Dtemp
        else:
            raise IndexError("Can not restore mask. Please check the given index")

        # Put back masked rows in D
        # -------------------------
        if (axis == -2 or axis == 0 or axis == "both") and D.shape(1) == colsize:
            if np.any(masked_rows):
                Dtemp = np.ma.zeros((rowsize, N))
                Dtemp[~masked_rows] = D
                Dtemp[masked_rows] = MASKED
                D = Dtemp
        else:
            raise IndexError("Can not restore mask. Please check the given index")

        # return the D array with restored masked data
        return D

    @property
    def X(self):
        # We use X property only to show this information to the end user. Internally
        # we must always use _X attribute to refer to the input data
        X = self._X.copy()
        if self._X_mask is not None:
            # restore masked row and column if necessary
            X = self._restore_masked_data(self, X, axis="both")
        return X

    @X.setter
    def X(self, value):
        self._X = value
        # this implies an automatic validation of the X value

    @tr.validate("_X")
    def _X_validate(self, proposal):
        X = proposal.value

        # we need a dataset with eventually  a copy of the original data (default being
        # to copy them)
        if not isinstance(X, NDDataset):
            X = NDDataset(X, copy=self.copy)
        elif self.copy:
            X = X.copy()

        # as in fit methods we often use np.linalg library, we cannot handle directly
        # masked data (so we remove them here and they will be restored at the end of
        # the process during transform or inverse transform methods

        # store the original shape as it will be eventually modified
        self._shape = X.shape

        # remove masked data and return modified dataset
        X = self._remove_masked_data(X)
        return X

    def _X_is_missing(self):
        if self._X is None:
            warnings.warn(
                "Sorry, but the X dataset must be defined "
                f"before you can use {self.name} methods."
            )
            return True

    # ....

    @tr.default("name")
    def _name_default(self):
        raise NameError("The name of the object was not defined.")

    # ----------------------------------------------------------------------------------
    # Public methods
    # ----------------------------------------------------------------------------------
    def fit(self, X, y=None):
        # to be overriden by user defined function (with the same name)
        self._fitted = False  # reiniit this flag
        raise NotImplementedError("fit method has not yet been implemented")

    def reconstruct(self):
        """
        Intended to be replaced in the subclasses
        """
        raise NotImplementedError(
            "reconstruct/inverse_transform has not yet been implemented"
        )

    def reduce(self):
        """
        Intended to be replaced in the subclasses
        """
        raise NotImplementedError("reduce/transform has not yet been implemented")

    def fit_reconstruct(self, X, y=None, **kwargs):
        """
        Fit the model with X and apply the dimensionality reduction on X.

        Parameters
        ----------
        X : NDDataset
            Input dataset of shape (n_observation, n_feature) to fit
        y : Optional target
            It's presence or not depends on the model.
        **kwargs : Additional optional keywords parameters

        Returns
        -------
        NDDataset
        """
        self.fit(X, y)
        Xhat = self.reconstruct(**kwargs)
        return Xhat

    def plotmerit(self, **kwargs):
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
            raise exceptions.NotFittedError(
                "The fit method must be used " "before using this method"
            )

        colX, colXhat, colRes = kwargs.pop("colors", ["blue", "green", "red"])

        X_hat = self.reconstruct(**kwargs)
        res = self.X - X_hat
        ax = self.X.plot()
        if self.X.x is not None:
            ax.plot(self.X.x.data, X_hat.T.data, color=colXhat)
            ax.plot(self.X.x.data, res.T.data, color=colRes)
        else:
            ax.plot(X_hat.T.data, color=colXhat)
            ax.plot(res.T.data, color=colRes)
        ax.autoscale(enable=True)
        ax.set_title(f"{self.name} merit plot")
        return ax

    # ----------------------------------------------------------------------------------
    # Utility functions
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

    @property
    @exceptions.deprecated(
        "Use log instead. This attribute will be removed in future version"
    )
    def logs(self):
        """
        Logs output.
        """
        return self.log
