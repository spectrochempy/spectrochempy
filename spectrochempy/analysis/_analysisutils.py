# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
This module implement some common utilities for the various analysis model.
"""
import inspect
from functools import partial

import numpy as np

from spectrochempy.utils import exceptions


# specific Analysis method exceptions
class NotFittedError(exceptions.SpectroChemPyError):
    """
    Exception raised when an analysis estimator is not fitted before use.

    Parameters
    ----------
    """

    def __init__(self, attr=None):
        frame = inspect.currentframe().f_back
        caller = frame.f_code.co_name if attr is None else attr
        model = frame.f_locals["self"].name
        message = (
            f"To use `{caller}`,  the method `fit` of model `{model}`"
            f" should be executed first"
        )
        super().__init__(message)


def _svd_flip(U, VT, u_based_decision=True):
    """
    Sign correction to ensure deterministic output from SVD.
    Adjusts the columns of u and the rows of v such that the loadings in the
    columns in u that are largest in absolute value are always positive.

    Parameters
    ----------
    u_based_decision : boolean, (default=True)
        If True, use the columns of u as the basis for sign flipping.
        Otherwise, use the rows of v.

    Notes
    -----
    Copied and modified from scikit-learn.utils.extmath (BSD 3 Licence)
    """

    if u_based_decision:
        # columns of U, rows of VT
        max_abs_cols = np.argmax(np.abs(U), axis=0)
        signs = np.sign(U[max_abs_cols, range(U.shape[1])])
        U *= signs
        VT *= signs[:, np.newaxis]
    else:
        # rows of V, columns of U
        max_abs_rows = np.argmax(np.abs(VT), axis=1)
        signs = np.sign(VT[range(VT.shape[0]), max_abs_rows])
        U *= signs
        VT *= signs[:, np.newaxis]

    return U, VT


class _set_output(object):
    # A decorator to transform np.ndarray output from models to NDDataset
    # according to the X input

    def __init__(
        self,
        method,
        *args,
        keepunits=True,
        keeptitle=True,
        typex=None,
        typey=None,
        typesingle=None,
    ):
        self.method = method
        self.keepunits = keepunits
        self.keeptitle = keeptitle
        self.typex = typex
        self.typey = typey
        self.typesingle = typesingle

    def __repr__(self):
        """Return the method's docstring."""
        return self.method.__doc__

    def __get__(self, obj, objtype):
        """Support instance methods."""
        return partial(self.__call__, obj)

    def __call__(self, obj, *args, **kwargs):

        from spectrochempy.core.dataset.coord import Coord
        from spectrochempy.core.dataset.nddataset import NDDataset

        # HACK to be able to used deprecated alias of the method, without error
        # because if not this modification obj appears two times
        if args and type(args[0]) == type(obj):
            args = args[1:]

        # determine the input X dataset
        X = obj.X
        # get the sklearn data output
        data = self.method(obj, *args, **kwargs)

        # restore eventually masked rows and columns
        axis = "both"
        if self.typex is not None and self.typex != "features":
            axis = 0
        elif self.typey is not None:
            axis = 1

        data = obj._restore_masked_data(data, axis=axis)

        # make a new dataset with this data
        X_transf = NDDataset(data)
        # Now set the NDDataset attributes
        if self.keepunits:
            X_transf.units = X.units
        X_transf.name = f"{X.name}_{obj.name}.{self.method.__name__}"
        X_transf.history = f"Created using method {obj.name}.{self.method.__name__}"
        if self.keeptitle:
            X_transf.title = X.title
        # make coordset
        M, N = X.shape
        if X_transf.shape == X.shape:
            X_transf.set_coordset(y=X.y, x=X.x)
        else:
            if self.typey == "components":
                X_transf.set_coordset(
                    y=Coord(
                        None,
                        labels=["#%d" % (i + 1) for i in range(X_transf.shape[0])],
                        title="components",
                    ),
                    x=X.x,
                )
            if self.typex == "components":
                X_transf.set_coordset(
                    y=X.y,
                    x=Coord(
                        None,
                        labels=["#%d" % (i + 1) for i in range(X_transf.shape[-1])],
                        title="components",
                    ),
                )
            if self.typex == "features":
                X_transf.set_coordset(
                    y=Coord(
                        None,
                        labels=["#%d" % (i + 1) for i in range(X_transf.shape[-1])],
                        title="components",
                    ),
                    x=X.x,
                )
            if self.typesingle == "components":
                # occurs when the data are 1D such as ev_ratio...
                X_transf.set_coordset(
                    x=Coord(
                        None,
                        labels=["#%d" % (i + 1) for i in range(X_transf.shape[-1])],
                        title="components",
                    ),
                )
        return X_transf


# wrap _set_output to allow for deferred calling
def _wrap_ndarray_output_to_nddataset(
    method=None, keepunits=True, keeptitle=True, typex=None, typey=None, typesingle=None
):
    if method:
        # case of the decorator without argument
        return _set_output(method)
    else:
        # and with argument
        def wrapper(method):
            return _set_output(
                method,
                keepunits=keepunits,
                keeptitle=keeptitle,
                typex=typex,
                typey=typey,
                typesingle=typesingle,
            )

        return wrapper
