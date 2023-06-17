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
from spectrochempy.extern.whittaker_smooth import whittaker_smooth as ws
from spectrochempy.processing._base._processingbase import ProcessingConfigurable
from spectrochempy.utils.decorators import deprecated, signature_has_configurable_traits
from spectrochempy.utils.docstrings import _docstring

__dataset_methods__ = [
    "savgol_filter",
    "savgol",
    "smooth",
    "whittaker",
]
__configurables__ = ["Filter"]
__all__ = __dataset_methods__ + __configurables__

_common_see_also = """
See Also
--------
Filter : Define and apply filters/smoothers using various algorithms.
smooth : Function to smooth data using various window filters.
savgol : Savitzky-Golay filter.
savgol_filter : Alias of `savgol`
whittaker : Whittaker-Eilers filter.
"""
_docstring.get_sections(
    _docstring.dedent(_common_see_also),
    base="Filter",
    sections=["See Also"],
)
_docstring.delete_params("Filter.see_also", "Filter")
_docstring.delete_params("Filter.see_also", "savgol")
_docstring.delete_params("Filter.see_also", "savgol_filter")
_docstring.delete_params("Filter.see_also", "whittaker")
_docstring.delete_params("Filter.see_also", "smooth")


# ======================================================================================
# Filter class processor
# ======================================================================================
@signature_has_configurable_traits
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
    - Savitzky-Golay filter (`savgol`)
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
            "median",
            "savgol",
            "whittaker",
        ],
        default_value="savgol",
        help="The filter method to be applied. By default, "
        "the Savitzky-Golay (savgol) filter is applied.",
    ).tag(config=True)

    size = mtr.PositiveOddInteger(
        default_value=5,
        help="The size of the filter window." "size must be a positive odd integer.",
    ).tag(config=True)

    order = tr.Integer(
        default_value=2,
        help="The order of the polynomial used to fit the data"
        "in the case of the Savitzky-Golay (savgol) filter. "
        "`order` must be less than size.\n"
        "In the case of the Whittaker-Eilers filter, order is the "
        "difference order of the penalized least squares.",
    ).tag(config=True, min=0)

    deriv = tr.Integer(
        default_value=0,
        help="The order of the derivative to compute in the case of "
        "the Savitzky-Golay (savgol) filter. This must be a "
        "non-negative integer. The default is 0, which means to "
        "filter the data without differentiating.",
    ).tag(config=True, min=0)

    lamb = tr.Float(
        default_value=1.0,
        help="Smoothing/Regularization parameter. The larger `lamb`\ , the smoother "
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

* When mode is ‘constant’, the padding value is given by `cval`.
* When the ‘interp’ mode is selected (the default), no extension is used.
  Instead, a polynomial of degree `order` is fit to the last `size` values
  of the edges, and this polynomial is used to evaluate the last window_length // 2
  output values.
* When mode is ‘nearest’, the last size values are repeated.
* When mode is ‘mirror’, the padding is created by reflecting the signal about the end
  of the signal.
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
            win = scipy.signal.get_window(self.method, self.size, fftbins=False)
            win = win / np.sum(win)
            data = scipy.ndimage.convolve1d(X, win, **kwargs)

        # Median filter
        # -------------
        elif self.method == "median":
            if "axis" in kwargs:
                axis = kwargs.pop("axis")
            if axis in (-2, 0):
                size = (self.size, 1)
            elif axis in (-1, 1):
                size = (1, self.size)
            data = scipy.ndimage.median_filter(X, size=size, **kwargs)

        # Savitzky-Golay filter
        # ---------------------
        elif self.method == "savgol":
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
            data = np.apply_along_axis(ws, -1, X, self.lamb, self.order)

        return data


_docstring.keep_params("Filter.parameters", "log_level")
_docstring.keep_params("Filter.parameters", "method")
_docstring.keep_params("Filter.parameters", "size")
_docstring.keep_params("Filter.parameters", "order")
_docstring.keep_params("Filter.parameters", "deriv")
_docstring.keep_params("Filter.parameters", "lamb")
_docstring.keep_params("Filter.parameters", "delta")
_docstring.keep_params("Filter.parameters", "mode")
_docstring.keep_params("Filter.parameters", "cval")


# TODO history
#     new.history = (
#         f"savgol_filter applied (window_length={window_length}, "
#         f"polyorder={polyorder}, deriv={deriv}, delta={delta}, mode={mode}, "
#         f"cval={cval}"
#     )


# ======================================================================================
# API / NDDataset functions
# ======================================================================================
# Instead of using directly the Filter class, we provide here some functions
# which are eventually more user-friendly and which can be used directly on NDDataset or
# called from the API.

# --------------------------------------------------------------------------------------


@_docstring.dedent
def smooth(dataset, size=5, window="avg", **kwargs):
    """
    Smooth the data using a window with requested size.

    This method is based on the convolution of a scaled kernel window with the signal.

    Parameters
    ----------
    %(dataset)s
    %(Filter.parameters.size)s
    window : `str`, optional, default:'flat'
        The type of window from 'flat' or 'avg', 'han' or 'hanning', 'hamming',
        'bartlett', 'blackman'.
        `avg` window will produce a moving average smoothing.
    %(kwargs)s

    Returns
    -------
    `NDDataset`
        Smoothed data.

    Other Parameters
    ----------------
    %(dim)s
    %(Filter.parameters.mode)s
    %(Filter.parameters.cval)s
    %(Filter.parameters.log_level)s

    See Also
    --------
    %(Filter.see_also.no_smooth)s
    """
    if window in ["flat", "avg", "han", "hanning", "hamming", "bartlett", "blackman"]:
        if window == "flat":
            window = "avg"
        if window == "hanning":
            window = "han"

        if kwargs.get("window_length", None) is not None:
            deprecated("window_length", replace="size", removed="0.8")
            size = kwargs.pop("window_length")

        return Filter(method=window, size=size, **kwargs).transform(dataset)
    else:
        raise ValueError(
            f"Window type '{window}' is not supported. "
            f"Supported types are 'flat' or 'avg', 'han' or 'hanning', 'hamming', "
            f"'bartlett', 'blackman'."
        )


# --------------------------------------------------------------------------------------
@_docstring.dedent
def savgol(dataset, size=5, order=2, **kwargs):
    """
    Savitzky-Golay filter.

    Wrapper of scpy.signal.savgol(). See the documentation of this function for more
    details.

    Parameters
    ----------
    %(dataset)s
    %(Filter.parameters.size)s
    order : `int`, optional, default=2
        The order of the polynomial used to fit the data. `order` must be less
        than size.
    %(kwargs)s

    Returns
    -------
    `NDDataset`
        Smoothed data.

    Other Parameters
    ----------------
    %(dim)s
    %(Filter.parameters.deriv)s
    %(Filter.parameters.delta)s
    %(Filter.parameters.mode)s
    %(Filter.parameters.cval)s
    %(Filter.parameters.log_level)s

    See Also
    --------
    %(Filter.see_also.no_savgol)s

    Notes
    -----
    Even spacing of the axis coordinates is NOT checked.
    Be aware that Savitzky-Golay algorithm is based on indexes, not on coordinates.
    """
    # TODO : check if coordinates are evenly spaced

    if kwargs.get("window_length", None) is not None:
        deprecated("window_length", replace="size", removed="0.8")
        size = kwargs.pop("window_length")

    if kwargs.get("polyorder", None) is not None:
        deprecated("polyorder", replace="order", removed="0.8")
        order = kwargs.pop("polyorder")

    return Filter(method="savgol", size=size, order=order, **kwargs).transform(dataset)


def savgol_filter(*args, **kwargs):
    """
    Savitzky-Golay filter.

    Alias of `savgol`.
    """
    # for backward compatibility TODO: should we deprecate this?
    return savgol(*args, **kwargs)


@_docstring.dedent
def whittaker(dataset, lamb=1.0, order=2, **kwargs):
    """
    Smooth the data using the Whittaker smoothing algorithm.

    This implementation based on the work by :cite:t:`eilers:2003` uses sparse matrices
    enabling high-speed processing of large input vectors.

    Copyright M. H. V. Werts, 2017 (see LICENSES/WITTAKER_SMOOTH_LICENSE.rst)

    Parameters
    ----------
    %(dataset)s
    %(Filter.parameters.lamb)s
    order : `int`, optional, default=2
        The difference order of the penalized least-squares.
    %(kwargs)s

    Returns
    -------
    `NDdataset`
        Smoothed data.

    Other Parameters
    ----------------
    %(dim)s
    %(Filter.parameters.log_level)s

    See Also
    --------
    %(Filter.see_also.no_whittaker)s
    """
    return Filter(method="whittaker", lamb=lamb, order=order, **kwargs).transform(
        dataset
    )
