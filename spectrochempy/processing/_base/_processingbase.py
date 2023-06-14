# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
This module implements the base abstract classes to define estimators such as PCA, ...
"""
import traitlets as tr

from spectrochempy.application import app
from spectrochempy.utils.baseconfigurable import BaseConfigurable
from spectrochempy.utils.decorators import _wrap_ndarray_output_to_nddataset
from spectrochempy.utils.docstrings import _docstring
from spectrochempy.utils.exceptions import NotTransformedError


# ======================================================================================
# Base class ProcessingConfigurable
# ======================================================================================
class ProcessingConfigurable(BaseConfigurable):
    __doc__ = _docstring.dedent(
        r"""
    Abstract class to write processing models.

    Unlike the `AnalysisConfigurable` class,
    this class has no fit methods but a only a transform method.

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
    _transformed = tr.Bool(False, help="False if the model was not yet applied")
    _reversed = tr.Bool(default_value=False, help="Whether the last axis is reversed")
    _dim = tr.Integer(default_value=-1, help="axis along which to apply ")

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

    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)

    # ----------------------------------------------------------------------------------
    # Private validation and default getter methods
    # ----------------------------------------------------------------------------------
    @tr.default("_X")
    def _X_default(self):
        raise NotTransformedError

    @property
    def _X_is_missing(self):
        # check whether X has been already defined
        try:
            if self._X is None:
                return True
        except NotTransformedError:
            return True
        return False

    @tr.observe("_X", "_dim")
    def _X_or_dim_changed(self, change):
        X = None
        if change.name == "_X":
            X = change.new
        elif change.name == "_dim":
            X = self._X
            dim = change.new
            # make dim an integer
            self._dim, _ = X.get_axis(dim, negative_axis=True)

        # is a reversed x axis (if x exists)
        if X.coordset is not None:
            self._reversed = X.coord(self._dim).reversed

    # ----------------------------------------------------------------------------------
    # Private methods that should be, most of the time, overloaded in subclass
    # ----------------------------------------------------------------------------------
    def _transform(self, X):  # pragma: no cover
        #  Intended to be replaced in the subclasses by user defined function
        #  (with the same name)
        raise NotImplementedError("_transform method has not yet been implemented")

    # ----------------------------------------------------------------------------------
    # Public methods and property
    # ----------------------------------------------------------------------------------
    @_wrap_ndarray_output_to_nddataset
    @_docstring.dedent
    def transform(self, dataset, dim=-1):
        r"""
        Transform the input dataset X using the current model.

        Parameters
        ----------
        %(dataset)s
        %(dim)s

        Returns
        -------
        `NDDataset`
            The transformed dataset.
        """
        self._transformed = False  # reinit this flag

        # fire the X validation and preprocessing.
        # X is expected to be a NDDataset or list of NDDataset.
        self._X = dataset
        self._dim = dim

        # _X_preprocessed has been computed when X was set.
        # At this stage they should be simple ndarrays
        newX = self._X_preprocessed

        # Call to the actual _transform method (overloaded in the subclass)
        Xt = self._transform(newX)

        # if the process was successful, _transformed is set to True so that other
        # methods which need to be applied will be possibly used.
        self._transformed = True
        return Xt

    _docstring.get_sections(
        _docstring.dedent(transform.__doc__),
        base="processing_transform",
        sections=["Parameters", "Returns"],
    )

    @property
    def log(self):
        """
        Return ``log`` output.
        """
        # A string handler (#1) is defined for the Spectrochempy logger,
        # thus we will return it's content
        return app.log.handlers[1].stream.getvalue().rstrip()
