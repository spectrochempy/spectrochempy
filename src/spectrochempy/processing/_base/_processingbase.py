# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Module implementing the base abstract classes to define estimators such as PCA, ..."""

import logging

import numpy as np
import traitlets as tr

from spectrochempy.application.application import app
from spectrochempy.utils.baseconfigurable import BaseConfigurable
from spectrochempy.utils.decorators import _wrap_ndarray_output_to_nddataset
from spectrochempy.utils.exceptions import NotTransformedError

logger = logging.getLogger(__name__)


# ======================================================================================
# Base class ProcessingConfigurable
# ======================================================================================
class ProcessingConfigurable(BaseConfigurable):
    """
    Abstract class to write processing models.

    Unlike the `AnalysisConfigurable` class,
    this class has no fit methods but a only a transform method.

    Processing model class must subclass this to get a minimal structure

    Parameters
    ----------
    log_level : any of [``"INFO"``, ``"DEBUG"``, ``"WARNING"``, ``"ERROR"``], optional, default: ``"WARNING"``
        The log level at startup. It can be changed later on using the
        `set_log_level` method or by changing the ``log_level`` attribute.

    """

    # Get doc sections for reuse in subclass

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
    @_wrap_ndarray_output_to_nddataset(meta_from="_X", preserve_identity=True)
    def transform(self, dataset, dim=-1):
        r"""
        Transform the input dataset X using the current model.

        Parameters
        ----------
        dataset : `NDDataset`
            The dataset to be transformed.
        dim : `int` or `str`, optional, default=-1
            The axis along which to apply the transformation. If negative, count from
            the last axis. If a string, it should be the name of the coordinate.

        Returns
        -------
        `NDDataset`
            The transformed dataset.

        """
        self._transformed = False  # reinit this flag

        # fire the X validation and preprocessing.
        # X is expected to be a NDDataset or list of NDDataset.
        self._X = dataset

        # Resolve string / complex dim selectors via the dataset's standard
        # get_axis() mechanism before assigning to the integer traitlet.
        resolved, _ = dataset.get_axis(dim, negative_axis=True)
        self._dim = resolved

        # _X_preprocessed has been computed when X was set.
        # At this stage they should be simple ndarrays
        newX = self._X_preprocessed

        # Call to the actual _transform method (overloaded in the subclass)
        Xt = self._transform(newX)

        # if the process was successful, _transformed is set to True so that other
        # methods which need to be applied will be possibly used.
        self._transformed = True
        return Xt

    # ------------------------------------------------------------------
    # Coordinate spacing detection (used by savgol for auto-delta)
    # ------------------------------------------------------------------
    def _detect_uniform_spacing(self, dataset, dim):
        """
        Detect uniform spacing from the coordinate along *dim*.

        Returns ``(delta_signed, message)`` where *delta_signed* is a float
        when the coordinate is uniformly spaced, or ``None`` when detection
        fails (in which case *message* explains why).

        The returned delta carries the sign of the real storage order:
        an ascending coordinate yields a positive delta, a descending one
        yields a negative delta.

        Detection uses ``numpy.diff`` and checks that the range of
        spacings is within a relative tolerance of 1% of the mean
        spacing.  This accommodates the limited precision of coordinate
        storage while remaining far below physically meaningful spacing
        variations.  No median or implicit fallback is produced.

        Parameters
        ----------
        dataset : `NDDataset`
            The dataset whose coordinate is inspected.
        dim : int
            The resolved integer axis index.

        Returns
        -------
        delta_signed : float or None
        message : str or None
        """
        try:
            coord = dataset.coord(dim)
        except Exception:
            return None, "no coordinate available"

        if coord is None:
            return None, "coordinate is None"

        values = np.asarray(coord.data, dtype=float)

        if values.ndim != 1 or values.size < 2:
            return None, "coordinate has fewer than 2 points"

        if not np.all(np.isfinite(values)):
            return None, "coordinate contains non-finite values (NaN or Inf)"

        diffs = np.diff(values)

        if np.all(diffs == 0):
            return None, "coordinate is degenerate (all values identical)"

        mean_diff = np.mean(diffs)
        spread = np.ptp(diffs)
        if spread / np.abs(mean_diff) > 0.01:
            return None, "coordinate is not uniformly spaced"

        return float(diffs[0]), None

    @property
    def log(self):
        """Return ``log`` output."""
        # A string handler (#1) is defined for the Spectrochempy logger,
        # thus we will return it's content
        return app.log.handlers[1].stream.getvalue().rstrip()
