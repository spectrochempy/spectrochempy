# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
This module implements the base abstract classes to define estimators such as PCA, ...
"""
import numpy as np
import traitlets as tr

from spectrochempy.core import app
from spectrochempy.utils.baseconfigurable import BaseConfigurable
from spectrochempy.utils.docstrings import _docstring
from spectrochempy.utils.exceptions import NotYetAppliedError


# ======================================================================================
# Base class ProcessingConfigurable
# ======================================================================================
class ProcessingConfigurable(BaseConfigurable):
    __doc__ = _docstring.dedent(
        r"""
    Abstract class to write processing models.

    Processing model class must subclass this to get a minimal structure

    Parameters
    ----------
    %(BaseConfigurable.parameters.log_level)s
    """
    )

    # Get doc sections for reuse in subclass
    _docstring.get_sections(__doc__, base="ProcessingConfigurable")

    # ----------------------------------------------------------------------------------
    # Runtime Parameters
    # ----------------------------------------------------------------------------------
    _applied = tr.Bool(False, help="False if the model was not yet applied")
    _out = tr.Any(help="the output of the _apply method")

    # ----------------------------------------------------------------------------------
    # Configuration parameters (mostly defined in subclass
    # as they depend on the model estimator)
    # ----------------------------------------------------------------------------------

    # Write here traits like e.g.,
    #     A = Unicode("A", help='description").tag(config=True)

    # ----------------------------------------------------------------------------------
    # Initialization
    # ----------------------------------------------------------------------------------
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # ----------------------------------------------------------------------------------
    # Private validation and default getter methods
    # ----------------------------------------------------------------------------------
    @tr.default("_X")
    def _X_default(self):
        raise NotYetAppliedError

    @property
    def _X_is_missing(self):
        # check whether X has been already defined
        try:
            if self._X is None:
                return True
        except NotYetAppliedError:
            return True
        return False

    # ----------------------------------------------------------------------------------
    # Private methods that should be, most of the time, overloaded in subclass
    # ----------------------------------------------------------------------------------
    def _apply(self, X):  # pragma: no cover
        #  Intended to be replaced in the subclasses by user defined function
        #  (with the same name)
        raise NotImplementedError("fit method has not yet been implemented")

    # ----------------------------------------------------------------------------------
    # Public methods and property
    # ----------------------------------------------------------------------------------
    def apply(self, X):
        r"""
        Apply the model with ``X`` as input dataset.

        Parameters
        ----------
        X : `NDDataset` or :term:`array-like` of shape (:term:`n_observations`\ , :term:`n_features`\ )
            Training data.

        Returns
        -------
        self
            The instance itself.
        """
        self._applied = False  # reinit this flag

        # fire the X validation and preprocessing.
        # X is expected to be a NDDataset or list of NDDataset.
        self._X = X

        # _X_preprocessed has been computed when X was set.
        # At this stage they should be simple ndarrays
        newX = self._X_preprocessed

        # Call to the actual _apply method (overloaded in the subclass)
        # warning : _apply must take ndarray arguments not NDDataset arguments.
        # when method must return NDDataset from the calculated data,
        # we use the decorator _wrap_ndarray_output_to_nddataset.
        self._out = self._apply(newX)

        # if the process was successful, _applied is set to True so that other method
        # which needs apply will be possibly used.
        self._applied = True
        return self

    # we do not use this method as a decorator as in this case signature of subclasses
    _docstring.get_sections(
        _docstring.dedent(apply.__doc__),
        base="processing_apply",
        sections=["Parameters", "Returns"],
    )
    # extract useful individual parameters doc
    _docstring.keep_params("processing_apply.parameters", "X")

    @property
    def log(self):
        """
        Return ``log`` output.
        """
        # A string handler (#1) is defined for the Spectrochempy logger,
        # thus we will return it's content
        return app.log.handlers[1].stream.getvalue().rstrip()

    @property
    def X(self):
        """
        Return the X input dataset (eventually modified by the model).
        """
        if self._X_is_missing:
            raise NotYetAppliedError
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
