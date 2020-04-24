# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2016 Christian Fernandez
# Laboratoire Catalyse et Spectrochimie, Caen, France.
#
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

__all__ = ['savgol_filter']

__dataset_methods__  = __all__

import scipy.signal

"""wrappers od scipy.signal filters"""

def savgol_filter(dataset, window_length, polyorder, deriv=0, delta=1.0, axis=-2, mode='interp', cval=0.0):
    """Apply a Savitzky-Golay filter to an array.

    This is a 1-d filter. If x has dimension greater than 1, axis determines the axis along which the filter is applied.

    Parameters
    ----------

    X : |NDDataset|
        The data to be filtered. If X.data is not a single or double precision floating point array, it will be converted
        to type numpy.float64 before filtering.

    window_length : int
        The length of the filter window (i.e. the number of coefficients). window_length must be a positive odd integer.

    polyorder : int
        The order of the polynomial used to fit the NDDataset. polyorder must be less than window_length.

    deriv : int, optional
        The order of the derivative to compute. This must be a nonnegative integer. The default is 0, which means to filter
        the data without differentiating.

    delta : float, optional
        The spacing of the samples to which the filter will be applied. This is only used if deriv > 0. Default is 1.0.

    axis : int or str, optional
        The axis of the array x along which the filter is to be applied. Default is -1.

    mode : str, optional
        Must be ‘mirror’, ‘constant’, ‘nearest’, ‘wrap’ or ‘interp’. This determines the type of extension to use for
        the padded signal to which the filter is applied. When mode is ‘constant’, the padding value is given by cval.
        See :py:scipy.signal.savgol_filter: for more details on ‘mirror’, ‘constant’, ‘wrap’, and ‘nearest’. When the ‘interp’ mode is
        selected (the default), no extension is used. Instead, a degree polyorder polynomial is fit to the last
        window_length values of the edges, and this polynomial is used to evaluate the last window_length // 2 output values.

    cval : scalar, optional
        Value to fill past the edges of the input if mode is ‘constant’. Default is 0.0.

    Returns
    -------

    NDDataset: same shape as x. data units are removed when deriv > 1
        The filtered data.
    """
    if type(axis) == str:
        if axis=='x':
            axis=0
        elif axis=='y':
            axis=1
        elif axis=='z':
            axis=2

    data = scipy.signal.savgol_filter(dataset.data, window_length, polyorder,
                                          deriv, delta, axis, mode, cval)

    out = dataset.copy()
    if deriv == 0:
        out.data = data * dataset.units
    else:
        out.data = data
    return out