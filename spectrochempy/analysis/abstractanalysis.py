# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
This module implements the MCRALS class.
"""

import logging
import warnings

import traitlets as tr

from spectrochempy.core import app, set_loglevel
from spectrochempy.core.common.meta import Meta
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.utils import exceptions
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
    _warm_start = tr.Bool(False)
    _fitted = tr.Bool(False)

    _X = tr.Instance(NDDataset, allow_none=True)  # Data to fit an estimate

    # ----------------------------------------------------------------------------------
    # Configuration parameters (depends on the estimator)
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
    @property
    def X(self):
        return self._X

    @X.setter
    def X(self, value):
        self._X = value
        # this implies an automatic validation of the X value

    @tr.validate("_X")
    def _X_validate(self, proposal):
        X = proposal.value
        if not isinstance(X, NDDataset):
            X = NDDataset(X)
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
    def fit(self, X, y):
        # to be overriden by user defined function (with the same name)
        self._fitted = False  # reiniit this flag
        raise NotImplementedError("fit method has not yet been implemented")

    def reconstruct(self):
        """
        to be overriden
        """
        raise NotImplementedError(
            "reconstruct/inverse_transform has not yet been implemented"
        )

    inverse_transform = reconstruct
    inverse_transform.__doc__ = "Alias of reconstruct (Scikit-learn terminology)"

    def reduce(self):
        """
        to be overriden
        """
        raise NotImplementedError("reduce/transform has not yet been implemented")

    transform = reconstruct
    transform.__doc__ = "Alias of reduce (Scikit-learn terminology)"

    def fit_reconstruct(self, X, y=None, **kwargs):
        """
        Fit the model with X and apply the dimensionality reduction on X.

        Parameters
        ----------
        X : NDDataset
            Input dataset of shape (n_observation, n_feature) to fit
        y : ignored

        Returns
        -------
        NDDataset
        """
        self.fit(X, y)
        Xhat = self.reconstruct(**kwargs)
        return Xhat

    fit_transform = fit_reconstruct
    fit_transform.__doc__ = "Alias of fit_transform (Scikit-learn terminology)"

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
