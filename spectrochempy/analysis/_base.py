# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
This module implements the base abstract classes to define estimators such as PCA, ...
"""

import inspect
import logging
import warnings
from copy import copy
from functools import partial, update_wrapper

import matplotlib.pyplot as plt
import numpy as np
import traitlets as tr
from sklearn import linear_model

from spectrochempy.application.metaconfigurable import MetaConfigurable
from spectrochempy.core import app, set_loglevel
from spectrochempy.core.dataset.baseobjects.ndarray import NDArray
from spectrochempy.core.dataset.coordset import CoordSet
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.extern.traittypes import Array
from spectrochempy.utils import exceptions
from spectrochempy.utils.constants import MASKED, NOMASK
from spectrochempy.utils.decorators import deprecated, preserve_signature
from spectrochempy.utils.docstrings import _docstring
from spectrochempy.utils.plots import NBlue, NGreen, NRed
from spectrochempy.utils.traits import NDDatasetType


# ======================================================================================
# Exceptions for analysis models
# ======================================================================================
class NotFittedError(exceptions.SpectroChemPyError):
    """
    Exception raised when an analysis estimator is not fitted
    but one use one of its method.

    Parameters
    ----------
    attr : method, optional
        The method from which the error was issued. In general it is determined
        automatically.
    """

    def __init__(self, attr=None):
        frame = inspect.currentframe().f_back
        caller = frame.f_code.co_name if attr is None else attr
        model = frame.f_locals["self"].name
        message = (
            f"To use `{caller}` ,  the method `fit` of model `{model}`"
            f" should be executed first"
        )
        super().__init__(message)


# ======================================================================================
# A decorator to transform np.ndarray output from models to NDDataset
# according to the X (default) and/or Y input
# ======================================================================================
class _set_output(object):
    def __init__(
        self,
        method,
        *args,
        meta_from="_X",  # the attribute or tuple of attributes from which meta data are taken
        units="keep",
        title="keep",
        typex=None,
        typey=None,
        typesingle=None,
    ):
        self.method = method
        update_wrapper(self, method)
        self.meta_from = meta_from
        self.units = units
        self.title = title
        self.typex = typex
        self.typey = typey
        self.typesingle = typesingle

    @preserve_signature
    def __get__(self, obj, objtype):
        """Support instance methods."""
        newfunc = partial(self.__call__, obj)
        update_wrapper(newfunc, self.method)
        return newfunc

    def __call__(self, obj, *args, **kwargs):

        from spectrochempy.core.dataset.coord import Coord
        from spectrochempy.core.dataset.nddataset import NDDataset

        # HACK to be able to used deprecated alias of the method, without error
        # because if not this modification obj appears two times
        if args and type(args[0]) == type(obj):
            args = args[1:]

        # get the method output - one or two arrays depending on the method and *args
        output = self.method(obj, *args, **kwargs)

        # restore eventually masked rows and columns
        axis = "both"
        if self.typex is not None and self.typex != "features":
            axis = 0
        elif self.typey is not None:
            axis = 1

        # if a single array was returned...
        if not isinstance(output, tuple):
            # ... make a tuple of 1 array:
            data_tuple = (output,)
            # ... and a tuple of 1 from_meta element:
            if not isinstance(self.meta_from, tuple):
                meta_from_tuple = (self.meta_from,)
            else:
                # ensure that the first one
                meta_from_tuple = (self.meta_from[0],)
        else:
            data_tuple = output
            meta_from_tuple = self.meta_from

        out = []
        for data, meta_from in zip(data_tuple, meta_from_tuple):
            X_transf = NDDataset(data)

            # Now set the NDDataset attributes from the original X

            # determine the input X dataset
            X = getattr(obj, meta_from)

            if self.units is not None:
                if self.units == "keep":
                    X_transf.units = X.units
                else:
                    X_transf.units = self.units
            X_transf.name = f"{X.name}_{obj.name}.{self.method.__name__}"
            X_transf.history = f"Created using method {obj.name}.{self.method.__name__}"
            if self.title is not None:
                if self.title == "keep":
                    X_transf.title = X.title
                else:
                    X_transf.title = self.title
            # make coordset
            M, N = X.shape
            if X_transf.shape == X.shape and self.typex is None and self.typey is None:
                X_transf.set_coordset(y=X.coord(0), x=X.coord(1))
            else:
                if self.typey == "components":
                    X_transf.set_coordset(
                        y=Coord(
                            None,
                            labels=["#%d" % (i) for i in range(X_transf.shape[0])],
                            title="components",
                        ),
                        x=X.coord(-1),
                    )
                if self.typex == "components":
                    X_transf.set_coordset(
                        y=X.coord(0),  # cannot use X.y in case of transposed X
                        x=Coord(
                            None,
                            labels=["#%d" % (i) for i in range(X_transf.shape[-1])],
                            title="components",
                        ),
                    )
                if self.typex == "features":
                    X_transf.set_coordset(
                        y=Coord(
                            None,
                            labels=["#%d" % (i) for i in range(X_transf.shape[-1])],
                            title="components",
                        ),
                        x=X.coord(1),
                    )
                if self.typesingle == "components":
                    # occurs when the data are 1D such as ev_ratio...
                    X_transf.set_coordset(
                        x=Coord(
                            None,
                            labels=["#%d" % (i) for i in range(X_transf.shape[-1])],
                            title="components",
                        ),
                    )
                if self.typesingle == "targets":
                    # occurs when the data are 1D such as PLSRegression intercept...
                    if X.coordset[0].labels is not None:
                        labels = X.coordset[0].labels
                    else:
                        labels = ["#%d" % (i + 1) for i in range(X.shape[-1])]
                    X_transf.set_coordset(
                        x=Coord(
                            None,
                            labels=labels,
                            title="targets",
                        ),
                    )

            # eventually restore masks
            X_transf = obj._restore_masked_data(X_transf, axis=axis)
            out.append(X_transf.squeeze())

        if len(out) == 1:
            return out[0]
        else:
            return tuple(out)


def _wrap_ndarray_output_to_nddataset(
    method=None,
    meta_from="_X",
    units="keep",
    title="keep",
    typex=None,
    typey=None,
    typesingle=None,
):
    # wrap _set_output to allow for deferred calling
    if method:
        # case of the decorator without argument
        out = _set_output(method)
    else:
        # and with argument
        def wrapper(method):
            return _set_output(
                method,
                meta_from=meta_from,
                units=units,
                title=title,
                typex=typex,
                typey=typey,
                typesingle=typesingle,
            )

        out = wrapper
    return out


# ======================================================================================
# Base class AnalysisConfigurable
# ======================================================================================
class AnalysisConfigurable(MetaConfigurable):

    __doc__ = _docstring.dedent(
        r"""
    Abstract class to write analysis model estimators.

    Analysis model class must subclass this to get a minimal structure

    Parameters
    ----------
    log_level : any of [``"INFO"``\ , ``"DEBUG"``\ , ``"WARNING"``\ , ``"ERROR"``\ ], optional, default: ``"WARNING"``
        The log level at startup.
    warm_start : `bool`\ , optional, default: `False`
        When fitting repeatedly on the same dataset, but for multiple
        parameter values (such as to find the value maximizing performance),
        it may be possible to reuse previous model learned from the previous parameter
        value, saving time.

        When `warm_start` is `True`\ , the existing fitted model attributes is used to
        initialize the new model in a subsequent call to `fit`\ .
    """
    )

    # Get doc sections for reuse in subclass
    _docstring.get_sections(__doc__, base="AnalysisConfigurable")

    # ----------------------------------------------------------------------------------
    # Runtime Parameters
    # ----------------------------------------------------------------------------------
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
    _output_type = tr.Enum(
        ["NDDataset", "ndarray"],
        default_value="NDDataset",
        help="Whether the output is a NDDataset or a ndarray",
    )

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
        """ """
        # An empty __doc__ is placed here, else Configurable.__doc__
        # will appear when there is no __init___.doc in subclass

        # Reset default configuration if not warm_start
        reset = not warm_start

        # Call the super class (MetaConfigurable) for initialisation
        super().__init__(parent=app, reset=reset)

        # Set log_level of the console report (accessible using the log property)
        set_loglevel(log_level)

        # Initial configuration
        # ---------------------
        # Reset all config parameters to default, if not warm_start
        defaults = self.parameters(default=True)
        configkw = {} if warm_start else defaults

        # Eventually take parameters from kwargs
        configkw.update(kwargs)

        # Now update all configuration parameters
        # if an item k is not in the config parameters, an error is raised.
        for k, v in configkw.items():
            if hasattr(self, k) and k in defaults:
                if getattr(self, k) != v:
                    setattr(self, k, v)
            else:
                raise KeyError(
                    f"'{k}' is not a valid configuration parameters. "
                    f"Use the method `parameters()` to check the current "
                    f"allowed parameters and their current value."
                )

        # If warm start we can use the previous fit as starting profiles.
        # so the flag _fitted is not set.
        if not warm_start:
            # We should not be able to use any methods requiring fit results
            # until the fit method has been executed
            self._fitted = False

    # ----------------------------------------------------------------------------------
    # Private methods
    # ----------------------------------------------------------------------------------
    def _make_dataset(self, d):
        # Transform an array-like object to NDDataset
        # or a list of array-like to a list of NDQataset
        if d is None:
            return
        if isinstance(d, (tuple, list)):
            d = [self._make_dataset(item) for item in d]
        elif not isinstance(d, NDDataset):
            d = NDDataset(d, copy=True)
        else:
            d = d.copy()
        return d

    def _get_masked_rc(self, mask):
        # Get the mask by row and columns.
        # -------------------------------
        # When a single element in the array is
        # masked, the whole row and columns for this element is masked as well as the
        # corresponding columns.
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

        if not hasattr(X, "mask") or not np.any(X._mask):
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
    @tr.observe("_X")
    def _preprocess_as_X_changed(self, change):
        # to be optionally replaced by user defined function (with the same name)
        X = change.new
        # .... preprocessing as scaling, centering, ... must return a ndarray with
        #  same shape a X.data

        # Set a X.data by default
        self._X_preprocessed = X.data

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
        X : `NDDataset` or :term:`array-like` of shape (:term:`n_observations`\ , :term:`n_features`\ )
            Training data.
        Y : any
            Depends on the model.

        Returns
        -------
        self
            The fitted instance itself.

        See Also
        --------
        fit_transform :  Fit the model with an input dataset ``X`` and apply the dimensionality reduction on ``X``\ .
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
    _docstring.get_sections(
        _docstring.dedent(fit.__doc__),
        base="analysis_fit",
        sections=["Parameters", "Returns", "See Also"],
    )
    # extract useful individual parameters doc
    _docstring.keep_params("analysis_fit.parameters", "X")

    @property
    def log(self):
        """
        Return ``log`` output.
        """
        # A string handler (#2) is defined for the Spectrochempy logger,
        # thus we will return it's content
        return app.log.handlers[2].stream.getvalue().rstrip()

    @property
    def X(self):
        """
        Return the X input dataset (eventually modified by the model).
        """
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
        else:
            return np.asarray(X)


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
    _docstring.get_sections(
        _docstring.dedent(__doc__),
        base="DecompositionAnalysis",
        sections=["See Also"],
    )

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

    def _transform(self, *args, **kwargs):  # pragma:  no cover
        # to be overriden in subclass such as PCA, MCRALS, ...
        raise NotImplementedError("transform has not yet been implemented")

    def _inverse_transform(self, *args, **kwargs):  # pragma:  no cover
        # to be overriden in subclass such as PCA, MCRALS, ...
        raise NotImplementedError("inverse_transform has not yet been implemented")

    def _get_components(self, n_components=None):  # pragma:  no cover
        # to be overriden in subclass such as PCA, MCRALS, ...
        raise NotImplementedError("get_components has not yet been implemented")

    # ----------------------------------------------------------------------------------
    # Public methods
    # ----------------------------------------------------------------------------------
    @_wrap_ndarray_output_to_nddataset(units=None, title=None, typex="components")
    @_docstring.dedent
    def transform(self, X=None, **kwargs):
        r"""
        Apply dimensionality reduction to `X`\ .

        Parameters
        ----------
        X : `NDDataset` or :term:`array-like` of shape (:term:`n_observations`\ , :term:`n_features`\ ), optional
            New data, where :term:`n_observations` is the number of observations
            and :term:`n_features` is the number of features.
            if not provided, the input dataset of the `fit` method will be used.
        %(kwargs)s

        Returns
        -------
        `NDDataset`
            Dataset with shape (:term:`n_observations`\ , :term:`n_components`\ ).

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

    # Get doc sections for reuse in subclass
    _docstring.get_sections(
        _docstring.dedent(transform.__doc__),
        base="analysis_transform",
        sections=["Parameters", "Other Parameters", "Returns"],
    )
    _docstring.keep_params("analysis_transform.parameters", "X")

    @_wrap_ndarray_output_to_nddataset
    @_docstring.dedent
    def inverse_transform(self, X_transform=None, **kwargs):
        r"""
        Transform data back to its original space.

        In other words, return an input `X_original` whose reduce/transform would
        be `X_transform`.

        Parameters
        ----------
        X_transform : array-like of shape (:term:`n_observations`\ , :term:`n_components`\ ), optional
            Reduced `X` data, where `n_observations` is the number of observations
            and `n_components` is the number of components. If `X_transform` is not
            provided, a transform of `X` provided in `fit` is performed first.
        %(kwargs)s

        Returns
        -------
        `NDDataset`
            Dataset with shape (:term:`n_observations`\ , :term:`n_features`\ ).

        Other Parameters
        ----------------
        %(analysis_transform.other_parameters)s

        See Also
        --------
        reconstruct : Alias of inverse_transform (Deprecated).
        """
        if not self._fitted:
            raise NotFittedError

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

    _docstring.get_sections(
        _docstring.dedent(inverse_transform.__doc__),
        base="analysis_inverse_transform",
        sections=["Parameters", "Returns"],
    )
    # _docstring.keep_params("analysis_inverse_transform.parameters", "X_transform")

    @_docstring.dedent
    def fit_transform(self, X, Y=None, **kwargs):
        r"""
        Fit the model with `X` and apply the dimensionality reduction on `X`\ .

        Parameters
        ----------
        %(analysis_fit.parameters.X)s
        Y : any
            Depends on the model.
        %(kwargs)s

        Returns
        -------
        %(analysis_transform.returns)s

        Other Parameters
        ----------------
        %(analysis_transform.other_parameters)s
        """
        try:
            self.fit(X, Y)
        except TypeError:
            # the current model does not use Y
            self.fit(X)
        X_transform = self.transform(X, **kwargs)
        return X_transform

    def reduce(self, X=None, **kwargs):
        # deprecated decorator do not preserve signature, so
        # i use a workaround
        return deprecated(replace="transform")(self.transform)(X, **kwargs)

    reduce.__doc__ = transform.__doc__ + "\nNotes\n-----\nDeprecated in version 0.6.\n"

    def reconstruct(self, X_transform=None, **kwargs):
        return deprecated(replace="inverse_transform")(self.inverse_transform)(
            X_transform, **kwargs
        )

    reconstruct.__doc__ = (
        inverse_transform.__doc__ + "\nNotes\n-----\nDeprecated in version 0.6.\n"
    )

    @_wrap_ndarray_output_to_nddataset(units=None, title=None, typey="components")
    def get_components(self, n_components=None):
        r"""
        Return the component's dataset: (selected :term:`n_components`\ , :term:`n_features`\ ).

        Parameters
        ----------
        n_components : `int`, optional, default: `None`
            The number of components to keep in the output dataset.
            If `None`, all calculated components are returned.

        Returns
        -------
        `~spectrochempy.core.dataset.nddataset.NDDataset`
            Dataset with shape (:term:`n_components`\ , :term:`n_features`\ )
        """
        if n_components is None or n_components > self._n_components:
            n_components = self._n_components

        # we call the specific _get_components method defined in subclasses
        components = self._get_components()[:n_components]

        return components

    @property
    @_wrap_ndarray_output_to_nddataset(units=None, title="keep", typey="components")
    def components(self):
        r"""
        `NDDataset` with components in feature space (:term:`n_components`\ , :term:`n_features`\ ).

        See Also
        --------
        get_components : Retrieve only the specified number of components.
        """
        return self._get_components()

    @property
    def n_components(self):
        """
        Number of components that were fitted.
        """
        if self._fitted:
            return self._n_components
        else:
            raise NotFittedError("n_components")

    # ----------------------------------------------------------------------------------
    # Plot methods
    # ----------------------------------------------------------------------------------
    @_docstring.dedent
    def plotmerit(self, X=None, X_hat=None, **kwargs):
        r"""
        Plot the input (:math:`X`\ ), reconstructed (:math:`\hat{X}`\ ) and residuals (:math:`E`\ ) datasets.

        :math:`X` and :math:`\hat{X}` can be passed as arguments. If not,
        the `X` attribute is used for :math:`X`\ and :math:`\hat{X}`\ is computed by
        the `inverse_transform` method

        Parameters
        ----------
        X : `NDDataset`\ , optional
            Original dataset. If is not provided (default), the `X`
            attribute is used and X_hat is computed using `inverse_transform`\ .
        X_hat : `NDDataset`\ , optional
            Inverse transformed dataset. if `X` is provided, `X_hat`
            must also be provided as compuyed externally.
        %(kwargs)s

        Returns
        -------
        `~matplotlib.axes.Axes`
            Matplotlib subplot axe.

        Other Parameters
        ----------------
        colors : `tuple` or `~numpy.ndarray` of 3 colors, optional
            Colors for `X` , `X_hat` and residuals ``E`` .
            in the case of 2D, The default colormap is used for `X` .
            By default, the three colors are :const:`NBlue` , :const:`NGreen`
            and :const:`NRed`  (which are colorblind friendly).
        offset : `float`, optional, default: `None`
            Specify the separation (in percent) between the
            :math:`X` , :math:`X_hat` and :math:`E`\ .
        nb_traces : `int` or ``'all'``\ , optional
            Number of lines to display. Default is ``'all'``\ .
        **others : Other keywords parameters
            Parameters passed to the internal `plot` method of the `X` dataset.
        """
        colX, colXhat, colRes = kwargs.pop("colors", [NBlue, NGreen, NRed])

        if X is None:
            X = self.X  # we need to use self.X here not self._X because the mask
            # are restored automatically
            if X_hat is None:
                # compute the inverse transform (this check that the model
                # is already fitted and handle eventual masking)
                X_hat = self.inverse_transform()
        elif X_hat is None:
            raise ValueError(
                "If X is provided, An externally computed X_hat dataset "
                "must be also provided."
            )

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
        offset = kwargs.pop("offset", None)
        if offset is None:
            offset = 0
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

    _docstring.get_sections(_docstring.dedent(plotmerit.__doc__), base="plotmerit")

    @property
    def Y(self):
        r"""
        The `Y` input.
        """
        # We use Y property only to show this information to the end-user. Internally
        # we use _Y attribute to refer to the input data
        if self._Y_is_missing:
            raise NotFittedError
        Y = self._Y
        return Y


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
    _docstring.get_sections(
        _docstring.dedent(__doc__),
        base="CrossDecompositionAnalysis",
        sections=["See Also"],
    )

    # ----------------------------------------------------------------------------------
    # Private methods that should be most of the time overloaded in subclass
    # ----------------------------------------------------------------------------------
    def _predict(self, *args, **kwargs):  # pragma:  no cover
        # to be overriden in subclass such as PLSRegression, ...
        raise NotImplementedError("predict has not yet been implemented")

    # ----------------------------------------------------------------------------------
    # Public methods
    # ----------------------------------------------------------------------------------

    @_wrap_ndarray_output_to_nddataset(meta_from="_Y", title=None)
    @_docstring.dedent
    def predict(self, X=None):
        """
        Predict targets of given observations.

        Parameters
        ----------
        X : `NDDataset` or :term:`array-like` of shape (:term:`n_observations`\ , :term:`n_features`\ ), optional
            New data, where :term:`n_observations` is the number of observations
            and :term:`n_features` is the number of features.
            if not provided, the input dataset of the `fit` method will be used.

        Returns
        -------
        `NDDataset`
            Datasets with shape (:term:`n_observations`\ ,) or ( :term:`n_observations`\ , :term:`n_targets`\ ).
        """
        if not self._fitted:
            raise NotFittedError()

        if X is None:
            X = self._X_preprocessed
        elif isinstance(X, NDDataset):
            X = X.data

        return self._predict(X)

    @_docstring.dedent
    def score(self, X=None, Y=None, sample_weight=None):
        r"""
        Return the coefficient of determination of the prediction.

        The coefficient of determination :math:`R^2` is defined as
        :math:`(1 - \frac{u}{v})` , where :math:`u` is the residual
        sum of squares ``((y_true - y_pred)** 2).sum()`` and :math:`v`
        is the total sum of squares ``((y_true - y_true.mean()) ** 2).sum()``\ .
        The best possible score is ``1.0`` and it can be negative (because the
        model can be arbitrarily worse). A constant model that always predicts
        the expected value of `Y`\ , disregarding the input features, would get
        a :math:`R^2` score of 0.0.

        Parameters
        ----------
        X : `NDDataset` or :term:`array-like` of shape (:term:`n_observations`\ , :term:`n_features`\ ), optional
            Test samples. If not given, the X attribute is used.
        Y : `NDDataset` or :term:`array-like` of shape (:term:`n_observations`\ , :term:`n_targets`\ ), optional
            True values for `X`.
        sample_weight : `NDDataset` or :term:`array-like` of shape (:term:`n_samples`\ ,), default: `None`
            Sample weights.

        Returns
        -------
        `float`
            :math:`R^2` of `predict`\ (X) w.r.t `Y`\ .
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
        units=None, title=None, meta_from=("_X", "_Y"), typex="components"
    )
    @_docstring.dedent
    def transform(self, X=None, Y=None, both=False, **kwargs):
        r"""
        Apply dimensionality reduction to `X`\ and `Y`\ .

        Parameters
        ----------
        X : `NDDataset` or :term:`array-like` of shape (:term:`n_observations`\ , :term:`n_features`\ ), optional
            New data, where :term:`n_observations` is the number of observations
            and :term:`n_features` is the number of features.
            if not provided, the input dataset of the `fit` method will be used.
        Y : `NDDataset` or :term:`array-like` of shape (:term:`n_observations`\ , :term:`n_targets`\ ), optional
            New data, where :term:`n_targets` is the number of variables to predict.
            if not provided, the input dataset of the `fit` method will be used.
        both : `bool`, default: `False`
            Whether to also apply the dimensionality reduction to Y when neither X nor Y are provided.
        %(kwargs)s

        Returns
        -------
        x_score, y_score: `NDDataset` or tuple of `NDDataset`
            Datasets with shape (:term:`n_observations`\ , :term:`n_components`\ ).

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
        else:
            return self._transform(newX, None)

    # Get doc sections for reuse in subclass
    _docstring.get_sections(
        _docstring.dedent(transform.__doc__),
        base="cross_decomposition_transform",
        sections=["Parameters", "Other Parameters", "Returns"],
    )
    _docstring.keep_params("cross_decomposition_transform.parameters", "X", "Y", "both")

    @_wrap_ndarray_output_to_nddataset(meta_from=("_X", "_Y"))
    @_docstring.dedent
    def inverse_transform(
        self, X_transform=None, Y_transform=None, both=False, **kwargs
    ):
        """
        Transform data back to its original space.

        In other words, return reconstructed `X` and `Y` whose reduce/transform would
        be `X_transform` and `Y_transform`.

        Parameters
        ----------
        X_transform : array-like of shape (:term:`n_observations`\ , :term:`n_components`\ ), optional
            Reduced `X` data, where `n_observations` is the number of observations
            and `n_components` is the number of components. If `X_transform` is not
            provided, a transform of `X` provided in `fit` is performed first.
        Y_transform : `NDDataset` or :term:`array-like` of shape (:term:`n_observations`\ , `n_components`\ ), optional
            New data, where :term:`n_targets` is the number of variables to predict. If `Y_transform` is not
            provided, a transform of `Y` provided in `fit` is performed first.
        %(kwargs)s

        Returns
        -------
        `NDDataset`
            Dataset with shape (:term:`n_observations`\ , :term:`n_features`\ ).

        Other Parameters
        ----------------
        %(analysis_transform.other_parameters)s

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
            X = self._inverse_transform(X_transform)
            return X
        else:
            X, Y = self._inverse_transform(X_transform, X_transform)
            return X, Y

    _docstring.get_sections(
        _docstring.dedent(inverse_transform.__doc__),
        base="analysis_inverse_transform",
        sections=["Parameters", "Returns"],
    )
    # _docstring.keep_params("analysis_inverse_transform.parameters", "X_transform")

    @_docstring.dedent
    def fit_transform(self, X, Y, both=False):
        """
        Fit the model with `X` and `Y` and apply the dimensionality reduction on `X` and optionally on `Y`\ .

        Parameters
        ----------
        %(analysis_fit.parameters.X)s
        Y : `NDDataset` or :term:`array-like` of shape (:term:`n_observations`\ , :term:`n_features`\ )
            Training data.
        both : `bool`\ , optional
            Whether to apply the dimensionality reduction on `X` and `Y` .

        Returns
        -------
        %(analysis_transform.returns)s
        """

        self.fit(X, Y)
        if both:
            return self.transform(X, Y)
        else:
            return self.transform(X)

    # ----------------------------------------------------------------------------------
    # Plot methods
    # ----------------------------------------------------------------------------------
    @_docstring.dedent
    def parityplot(
        self,
        Y=None,
        Y_hat=None,
        clear=True,
        **kwargs,
    ):
        r"""
        Plot the predicted (:math:`\hat{Y}`\ ) vs measured (:math:`Y`\ ) values.

        :math:`Y` and :math:`\hat{Y}` can be passed as arguments. If not,
        the `Y` attribute is used for :math:`Y`\ and :math:`\hat{Y}`\ is computed by
        the `inverse_transform` method.

        Parameters
        ----------
        Y : `NDDataset`\ , optional
            Measured values. If is not provided (default), the `Y`
            attribute is used and Y_hat is computed using `inverse_transform`\ .
        Y_hat : `NDDataset`\ , optional
            Predicted values. if `Y` is provided, `Y_hat` must also be provided as
            computed externally.
        clear : `bool`\ , optional
            Whether to plot on a new axes. Default is True.
        %(kwargs)s

        Returns
        -------
        `~matplotlib.axes.Axes`
            Matplotlib subplot axe.

        Other Parameters
        ----------------
        s : `float` or :term:`array-like`, shape (n, ), optional
            The marker size in points**2 (typographic points are 1/72 in.).
            Default is rcParams['lines.markersize'] ** 2.
        c : :term:`array-like` or `list` of colors or color, optional
            The marker colors. Possible values:

            - A scalar or sequence of n numbers to be mapped to colors using cmap
              and norm.
            - A 2D array in which the rows are RGB or RGBA.
            - A sequence of colors of length n.
            - A single color format string.
              see `~matplotlib.pyplot.scatter` for details.

        marker : `markerMarkerStyle`, default: rcParams["scatter.marker"] (default: 'o')
            The marker style. marker can be either an instance of the class or the text
            shorthand for a particular marker. See `~matplotlib.markers` for more
            information.
        cmap : `str` or `Colormap`, default: rcParams["image.cmap"] (default: 'viridis')
            The Colormap instance or registered colormap name used to map scalar data
            to colors.
            This parameter is ignored if c is RGB(A).
        norm : `str` or Normalize, optional
            The normalization method used to scale scalar data to the [0, 1] range
            before mapping
            to colors using cmap. By default, a linear scaling is used, mapping the
            lowest value to
            0 and the highest to 1.
            If given, this can be one of the following:

            - An instance of Normalize or one of its subclasses
              (see Colormap Normalization).
            - A scale name, i.e. one of "linear", "log", "symlog", "logit", etc.
              For a list of available scales, call
              matplotlib.scale.get_scale_names(). In that case, a suitable Normalize
              subclass is dynamically generated
              and instantiated.
              This parameter is ignored if c is RGB(A).

        vmin, vmax : `float`\ , optional
            When using scalar data and no explicit norm, vmin and vmax define the data
            range that the colormap covers.
            By default, the colormap covers the complete value range of the supplied
            data. It is an error to use
            vmin/vmax when a norm instance is given (but using a str norm name together
            with vmin/vmax is acceptable).
            This parameter is ignored if c is RGB(A).
        alpha : `float`\ , default: 0.5
            The alpha blending value, between 0 (transparent) and 1 (opaque).
        linewidths : `float` or array-like, default: rcParams["lines.linewidth"] (default: 1.5)
            The linewidth of the marker edges. Note: The default edgecolors is 'face'.
            You may want to change this as well.
        edgecolors : {'face', 'none', None} or color or sequence of color, default: rcParams["scatter.edgecolors"], (default: 'face')
            The edge color of the marker. Possible values:
            'face': The edge color will always be the same as the face color.
            'none': No patch boundary will be drawn.
            A color or sequence of colors.
            For non-filled markers, edgecolors is ignored. Instead, the color is
            determined like with 'face',
            i.e. from c, colors, or facecolors.
        plotnonfinite : `bool`\ , default: False
            Whether to plot points with nonfinite c (i.e. inf, -inf or nan).
            If True the points are drawn with the bad
            colormap color (see Colormap.set_bad).
        """

        s = kwargs.pop("s", None)
        c = kwargs.pop("c", None)
        marker = kwargs.pop("marker", None)
        cmap = kwargs.pop("cmap", None)
        norm = kwargs.pop("norm", None)
        vmin = kwargs.pop("vmin", None)
        vmax = kwargs.pop("vmax", None)
        alpha = kwargs.pop("alpha", 0.5)
        linewidths = kwargs.pop("linewidths", None)
        edgecolors = kwargs.pop("edgecolors", None)
        plotnonfinite = kwargs.pop("plotnonfinite", False)
        data = kwargs.pop("data", None)

        if Y is None:
            Y = self.Y
            if Y_hat is None:
                # compute the inverse transform (this check that the model
                # is already fitted and handle eventual masking)
                Y_hat = self.predict(self.X)
        elif Y_hat is None:
            raise ValueError(
                "If Y is provided, An externally computed Y_hat dataset "
                "must be also provided."
            )

        if Y._squeeze_ndim == 1:
            # normally this was done before, but if needed.
            Y = Y.squeeze()
            Y_hat = Y_hat.squeeze()

        plt.style.use(["default"])
        plt.rcParams.update({"font.size": 14})
        if clear:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            ax = plt.gca()
        if len(Y.shape) == 1:
            plt.scatter(
                Y.data,
                Y_hat.data,
                s=s,
                c=c,
                marker=marker,
                cmap=cmap,
                norm=norm,
                vmin=vmin,
                vmax=vmax,
                alpha=alpha,
                linewidths=linewidths,
                edgecolors=edgecolors,
                plotnonfinite=plotnonfinite,
                data=data,
                **kwargs,
            )
        else:
            for col in Y.shape[1]:
                plt.scatter(
                    Y.data[:, col],
                    Y_hat.data[:, col],
                    s=s,
                    c=c,
                    marker=marker,
                    cmap=cmap,
                    norm=norm,
                    vmin=vmin,
                    vmax=vmax,
                    alpha=alpha,
                    linewidths=linewidths,
                    edgecolors=edgecolors,
                    plotnonfinite=plotnonfinite,
                    data=data,
                    **kwargs,
                )
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        xymin = min(xmin, ymin)
        xymax = max(xmax, ymax)
        ax.set_xlim(xymin, xymax)
        ax.set_ylim(xymin, xymax)
        plt.plot([xymin, xymax], [xymin, xymax])
        plt.legend()
        plt.xlabel("measured values")
        plt.ylabel("predicted values")
        plt.tight_layout()

        return ax

    _docstring.get_sections(_docstring.dedent(parityplot.__doc__), base="parityplot")


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
        help=r"When set to `True` , forces the coefficients (\ `coef`\ ) "
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
        r"""
        Fit linear model.

        Parameters
        ----------
        X : `NDDataset` or :term:`array-like` of shape (:term:`n_observations`\ ,:term:`n_features`\ )
            Training data, where `n_observations` is the number of observations
            and `n_features` is the number of features.
        Y : :term:`array-like` of shape (:term:`n_observations`\ ,) or (:term:`n_observations`\ ,:term:`n_targets`\ )
            Target values. Will be cast to `X`\ 's dtype if necessary.
        sample_weight : :term:`array-like` of shape (:term:`n_observations`\ ,), default: `None`
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
        Return the `Y` input dataset.
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
        r"""
        Estimated coefficients for the linear regression problem.

        If multiple targets are passed during the fit (Y 2D), this is a 2D array of
        shape (:term:`n_targets`\ , :term:`n_features`\ ), while if only one target
        is passed, this is a 1D array of length :term:`n_features`\ .
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
        r"""
        Return a float or an array of shape (:term:`n_targets`\ ,).

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
        X : `NDDataset` or :term:`array-like` matrix, shape (:term:`n_observations`\ ,:term:`n_features`\ )
            Observations. If `X` is not set, the input `X` for `fit` is used.

        Returns
        -------
        `~spectrochempy.core.dataset.nddataset.NDDataset`
            Predicted values (object of type of the input) using a ahape (:term:`n_observations`\ ,).
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
        the expected value of `Y`\ , disregarding the input features, would get
        a :math:`R^2` score of 0.0.

        Parameters
        ----------
        X : `NDDataset` or :term:`array-like` of shape (:term:`n_observations`\ , :term:`n_features`\ )
            Test samples.

        Y : `NDDataset` or :term:`array-like` of shape (:term:`n_observations`\ ,)
            True values for `X`\ .

        sample_weight : :term:`array-like` of shape (:term:`n_observations`\ ,), default: `None`
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
