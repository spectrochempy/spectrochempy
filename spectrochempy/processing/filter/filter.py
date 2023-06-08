# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
import numpy as np
import scipy.signal
import traitlets as tr

import spectrochempy.utils.traits as mtr
from spectrochempy.application import error_
from spectrochempy.extern.whittaker_smooth import whittaker_smooth as ws
from spectrochempy.processing._base._processingbase import ProcessingConfigurable
from spectrochempy.utils.decorators import signature_has_configurable_traits
from spectrochempy.utils.docstrings import _docstring

__dataset_methods__ = ["savgol_filter", "sgs", "smooth", "whittaker"]
__configurables__ = ["Filter"]
__all__ = __dataset_methods__ + __configurables__

_common_see_also = """
See Also
--------
Filter : Define and apply filters/smoothers using various algorithms.
sgs : Savitzky-Golay filter.
savgol_filter : Alias of `sgs`
whittaker : Whittaker-Eilers filter.
smooth : Smooth the data using a moving average procedure.
han : Han window filter.
hamming : Hamming window filter.
bartlett : Bartlett window filter.
blackman : Blackman window filter.
"""
_docstring.get_sections(
    _docstring.dedent(_common_see_also),
    base="Filter",
    sections=["See Also"],
)
_docstring.delete_params("Filter.see_also", "Filter")
_docstring.delete_params("Filter.see_also", "sgs")
_docstring.delete_params("Filter.see_also", "savgol_filter")
_docstring.delete_params("Filter.see_also", "whittaker")
_docstring.delete_params("Filter.see_also", "smooth")
_docstring.delete_params("Filter.see_also", "hanning")
_docstring.delete_params("Filter.see_also", "hamming")
_docstring.delete_params("Filter.see_also", "bartlett")
_docstring.delete_params("Filter.see_also", "blackman")


# ======================================================================================
# Filter class processor
# ======================================================================================
@signature_has_configurable_traits
# Note: with this decorator
# Configurable traits are added to the signature as keywords
# if they are not yet present.
class Filter(ProcessingConfigurable):
    __doc__ = _docstring.dedent(
        """
    Filters/smoothers processor.

    The filters can be applied to 1D datasets consisting in a single row
    with :term:`n_features` or to a 2D dataset with shape (:term:`n_observations`\ ,
    :term:`n_features`\ ).

    Various filters/smoothers can be applied to the data. The currently available
    filters are:

    - Moving average (`avg`)
    - Convolution filters (`han`, `hamming`, `bartlett`, `blackman`)
    - Savitzky-Golay filter (`sgs`)
    - Whittaker-Eilers filter (`whittaker`)

    Parameters
    ----------
    %(ProcessingConfigurable.parameters)s

    See Also
    --------
    %(Filter.see_also.no_Filter)s
    """
    )
    method = tr.Enum(
        [
            "avg",
            "han",
            "hamming",
            "bartlett",
            "blackman",
            "sgs",
            "whittaker",
        ],
        default_value="sgs",
        help="The filter method to be applied. By default, "
        "the Savitzky-Golay (sgs) filter is applied.",
    ).tag(config=True)

    size = mtr.PositiveOddInteger(
        default_value=5,
        help="The size of the filter window." "size must be a positive odd integer.",
    ).tag(config=True)

    order = tr.Integer(
        default_value=2,
        help="The order of the polynomial used to fit the data"
        "in the case of the Savitzky-Golay (sgs) filter. "
        "`order` must be less than size.\n"
        "In the case of the Whittaker-Eilers filter, order is the "
        "difference order of the penalized least squares.",
    ).tag(config=True, min=0)

    deriv = tr.Integer(
        default_value=0,
        help="The order of the derivative to compute in the case of "
        "the Savitzky-Golay (sgs) filter. This must be a "
        "non-negative integer. The default is 0, which means to "
        "filter the data without differentiating.",
    ).tag(config=True, min=0)

    lambd = tr.Float(
        default_value=1.0,
        help="Smoothing/Regularization parameter. The larger `lambd`\ , the smoother "
        "the data.",
    ).tag(config=True)

    delta = tr.Float(
        default_value=1.0,
        help="The spacing of the samples to which the filter will be applied. "
        "This is only used if deriv > 0.",
    ).tag(config=True)

    mode = tr.Enum(
        ["mirror", "constant", "nearest", "wrap", "interp"],
        default_value="interp",
        help="""
The type of extension to use for the padded signal to which the filter is applied.

* When mode is ‘constant’, the padding value is given by `cval`.\n
* When the ‘interp’ mode is selected (the default), no extension is used.
  Instead, a polynomial of degree `order` is fit to the last `size` values
  of the edges, and this polynomial is used to evaluate the last window_length // 2
  output values. \n
* When mode is ‘nearest’, the last size values are repeated. \n
* When mode is ‘mirror’, the padding is created by reflecting the signal about the end
  of the signal. \n"
* When mode is ‘wrap’, the signal is wrapped around on itself to create the padding.

See `scipy.signal.savgol_filter` for more details on ‘mirror’, ‘constant’, ‘wrap’,
and ‘nearest’.
""",
    ).tag(config=True)

    cval = tr.Float(
        default_value=0.0,
        help="Value to fill past the edges of the input if `mode` is ‘constant’. ",
    ).tag(config=True)

    # ----------------------------------------------------------------------------------
    # Initialisation
    # ----------------------------------------------------------------------------------
    def __init__(
        self,
        log_level="WARNING",
        **kwargs,
    ):
        # call the super class for initialisation of the configuration parameters
        # to do before anything else!
        super().__init__(
            log_level=log_level,
            **kwargs,
        )

    # ----------------------------------------------------------------------------------
    # Private methods
    # ----------------------------------------------------------------------------------
    def _transform(self, X):
        kwargs = dict(  # param for avg and convolution filters
            axis=self._dim,
            mode="reflect" if self.mode == "interp" else self.mode,
            cval=self.cval,
        )

        # smooth with moving average
        # --------------------------
        if self.method == "avg":
            data = scipy.ndimage.uniform_filter1d(X, self.size, **kwargs)

        # Convolution filters
        # -------------------
        elif self.method in ["han", "hamming", "bartlett", "blackman"]:
            win = scipy.signal.get_window(self.method, self.size)
            win = win / np.sum(win)
            data = scipy.ndimage.convolve1d(X, win, **kwargs)

        # Savitzky-Golay filter
        # ---------------------
        elif self.method == "sgs":
            kwargs = dict(
                axis=self._dim,
                deriv=self.deriv,
                delta=self.delta,
                mode=self.mode,
                cval=self.cval,
            )
            data = scipy.signal.savgol_filter(X, self.size, self.order, **kwargs)

            # Change derived data sign if we have reversed coordinate axis
            if self._reversed and self.deriv:
                data = data * (-1) ** self.deriv

        # Whittaker-Eilers filter
        # -----------------------
        elif self.method == "whittaker":
            data = np.apply_along_axis(ws, -1, X, self.lambd, self.order)

        return data


# ======================================================================================
# API / NDDataset functions
# ======================================================================================
# Instead of using directly the Filter class, we provide here some functions
# which are eventually more user-friendly and which can be used directly on NDDataset or
# called from the API.

# --------------------------------------------------------------------------------------
_docstring.keep_params("Filter.parameters", "size")


@_docstring.dedent
def smooth(dataset, size=5, window="avg", **kwargs):
    """
    Smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized in the
    beginning and end part of the output data.

    Parameters
    ----------
    %(dataset)s
    %(Filter.parameters.size)s
    window : `str`, optional, default:'flat'
        The type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'.
        flat window will produce a moving average smoothing.
    **kwargs
        Optional keyword parameters (see Other Parameters).

    Returns
    -------
    NDDataset
        Smoothed data.

    Other Parameters
    ----------------
    %(dim)s

    See Also
    --------
    %(Filter.see_also.no_smooth)s                                                                                                                                     : Wittaker smoother.
    """
    return Filter(size=size, method=window, **kwargs).apply(dataset)


# --------------------------------------------------------------------------------------
def sgs(dataset, **kwargs):
    """
    Savitzky-Golay filter.

    Wrapper of scpy.signal.savgol(). If dataset has dimension greater than 1,
    dim determines the axis along which the filter is applied.

    Parameters
    ----------
    dataset : `NDDataset`
        The dataset to be filtered.
    **kwargs
        Optional keyword parameters (see Other Parameters).

    Returns
    -------
    NDDataset
        The filtered data with same shape as x. data units are removed when deriv > 1.

    Other Parameters
    ----------------
    dim : str or int, optional, default='x'
        Specify on which dimension to apply this method. If `dim` is specified as an
        integer it is equivalent to the usual `axis` numpy parameter.
    inplace : bool, optional, default=False
        True if we make the transform inplace.  If False, the function return a new
        object.

    Notes
    -----
    Even spacing of the axis coordinates is NOT checked. Be aware that Savitzky-Golay
    algorithm is based on indexes, not on coordinates.

    See Also
    ---------
    smooth : Smooth the data using a window with requested size.
    whittaker_smooth : Whittaker smoother.

    """

    new = dataset.copy() if not kwargs.pop("inplace", False) else dataset

    is_ndarray = False
    axis = kwargs.pop("dim", kwargs.pop("axis", -1))
    if hasattr(new, "get_axis"):
        axis, dim = new.get_axis(axis, negative_axis=True)
        data = new.data
    else:
        is_ndarray = True
        data = new

    data = scipy.signal.savgol_filter(
        data, window_length, polyorder, deriv, delta, axis, mode, cval
    )

    if not is_ndarray:
        if deriv != 0 and dataset.coord(dim).reversed:
            data = data * (-1) ** deriv
        new.data = data
    else:
        new = data

    if not is_ndarray:
        new.history = (
            f"savgol_filter applied (window_length={window_length}, "
            f"polyorder={polyorder}, deriv={deriv}, delta={delta}, mode={mode}, "
            f"cval={cval}"
        )
    return new


def savgol_filter(*args, **kwargs):
    """
    Savitzky-Golay filter.

    Alias of `sgs`.

    See Also
    --------
    sgs : Savitzky-Golay filter.
    whittaker : Whittaker-Eilers filter.
    smooth : Smooth the data using a window with requested size.
    """
    # for backward compatibility TODO: should we deprecate this?
    return sgs(*args, **kwargs)


def whittaker(dataset, lamb=0.2, d=2, **kwargs):
    """
    Smooth the data using the Whittaker smoothing algorithm.

    This implementation based on the work by :cite:t:`eilers2003` uses sparse matrices
    enabling high-speed processing of large input vectors.

    Copyright M. H. V. Werts, 2017 (see LICENSES/WITTAKER_SMOOTH_LICENSE.rst)

    Parameters
    ----------
    dataset : `NDDataset` or a ndarray-like object
        Input object.
    lamb : `float`, optional, default=0.2
        Regularization parameter for the smoothing algorithm (roughness penalty)
        The larger `lamb`\ , the smoother the data.
    order : `int`, optional, default=2
        Order of the smoothing.
    **kwargs
        Optional keyword parameters (see Other Parameters).

    Returns
    -------
    NDdataset
        Smoothed data.

    Other Parameters
    ----------------
    dim : str or int, optional, default='x'.
        Specify on which dimension to apply this method. If `dim` is specified as an
        integer it is equivalent to the usual `axis` numpy parameter.
    inplace : bool, optional, default=False.
        True if we make the transform inplace.  If False, the function return a new
        object

    See Also
    --------
    savgol_filter : Apply a Savitzky-Golay filter.
    smooth : Smooth the data using a window with requested size.
    """
    if not kwargs.pop("inplace", False):
        # default
        new = dataset.copy()
    else:
        new = dataset

    is_ndarray = False
    axis = kwargs.pop("dim", kwargs.pop("axis", -1))
    if hasattr(new, "get_axis"):
        axis, dim = new.get_axis(axis, negative_axis=True)
    else:
        is_ndarray = True

    swapped = False
    if axis != -1:
        new.swapdims(axis, -1, inplace=True)  # must be done in  place
        swapped = True

    data = new.data
    for i in range(data.shape[0]):
        y = data[i]
        data[i] = ws(y, lamb, d)

    if not is_ndarray:
        new.data = data
        new.history = (
            f"smoothing using Whittaker algorithm with lambda={lamb} and d={d}"
        )

        # restore original data order if it was swapped
        if swapped:
            new.swapdims(axis, -1, inplace=True)  # must be done inplace
    else:
        new = data

    return new
