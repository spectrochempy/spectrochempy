# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory.
# ======================================================================================================================

__all__ = ["smooth"]
__dataset_methods__ = __all__

import numpy as np
from spectrochempy.core import error_


def smooth(dataset, window_length=5, window="flat", **kwargs):
    """
    Smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal (with the window size) in both ends so that
    transient parts are minimized in the begining and end part of the output data.

    Parameters
    ----------
    dataset :  |NDDataset| or a ndarray-like object
        Input object.
    window_length :  int, optional, default=5
        The dimension of the smoothing window; must be an odd integer.
    window : str, optional, default='flat'
        The type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'.
        flat window will produce a moving average smoothing.
    **kwargs : dict
        See other parameters.

    Returns
    -------
    smoothed
        Same type as input dataset.

    Other Parameters
    ----------------
    dim : str or int, optional, default='x'.
        Specify on which dimension to apply this method. If `dim` is specified as an integer it is equivalent
        to the usual `axis` numpy parameter.
    inplace : bool, optional, default=False.
        True if we make the transform inplace.  If False, the function return a new object

    See Also
    --------
    savgol_filter : Apply a Savitzky-Golay filter.

    Examples
    --------

    >>> ds = scp.read("irdata/nh4y-activation.spg")
    >>> ds.smooth(window_length=11)
    NDDataset: [float64] a.u. (shape: (y:55, x:5549))
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

    swaped = False
    if axis != -1:
        new.swapdims(axis, -1, inplace=True)  # must be done in  place
        swaped = True

    if (window_length % 2) != 1:
        error_("Window length must be an odd integer.")

    if new.shape[-1] < window_length:
        error_("Input vector needs to be bigger than window size.")
        return new

    if window_length < 3:
        return new

    wind = {
        "flat": np.ones,
        "hanning": np.hanning,
        "hamming": np.hamming,
        "bartlett": np.bartlett,
        "blackman": np.blackman,
    }

    if not callable(window):
        if window not in wind.keys():
            error_(
                "Window must be a callable or a string among 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
            )
            return new
        window = wind[window]

    # extend on both side to limit side effects
    dat = np.r_[
        "-1",
        new.data[..., window_length - 1 : 0 : -1],
        new.data,
        new.data[..., -1:-window_length:-1],
    ]

    w = window(window_length)
    data = np.apply_along_axis(np.convolve, -1, dat, w / w.sum(), mode="valid")
    data = data[..., int(window_length / 2) : -int(window_length / 2)]

    if not is_ndarray:
        new.data = data
        new.history = (
            f"smoothing with a window:{window.__name__} of length {window_length}"
        )

        # restore original data order if it was swaped
        if swaped:
            new.swapdims(axis, -1, inplace=True)  # must be done inplace
    else:
        new = data

    return new


if __name__ == "__main__":
    pass
