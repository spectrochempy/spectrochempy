# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

__all__ = ['savgol_filter', 'detrend']

__dataset_methods__  = __all__

import scipy.signal

"""wrappers of scipy.signal filters, """

#Todo:
#find_peaks_cwt(vector, widths[, wavelet, ...]) 	Attempt to find the peaks in a 1-D array.
#argrelmin(data[, axis, order, mode]) 	Calculate the relative minima of data.
#argrelmax(data[, axis, order, mode]) 	Calculate the relative maxima of data.
#argrelextrema(data, comparator[, axis, ...]) 	Calculate the relative extrema of data.

def savgol_filter(dataset, window_length, polyorder, deriv=0, delta=1.0, dim='x', mode='interp', cval=0.0):
    """
    Apply a Savitzky-Golay filter to an array.

    Wrapper of scpy.signal.savgol(). If dataset has dimension greater than 1, dim determines the axis along which the filter is applied.
    
    Parameters
    ----------
    
    dataset : |NDDataset|
        The data to be filtered. If dataset.data is not a single or double precision floating point array, it will be converted
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
    dim : str. Optional, default='x'.
        Along which axis to perform the alignment.
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
    
    Notes
    -----
    Even spacing of the axis coordinates is NOT checked. Be aware that Savitzky-Golay algorithm
    is based on indexes, not on coordinates.
    
    Details on the `mode` options:
        'mirror':
            Repeats the values at the edges in reverse order.  The value
            closest to the edge is not included.
        'nearest':
            The extension contains the nearest input value.
        'constant':
            The extension contains the value given by the `cval` argument.
        'wrap':
            The extension contains the values from the other end of the array.
    
    For example, if the input is [1, 2, 3, 4, 5, 6, 7, 8], and
    `window_length` is 7, the following shows the extended data for
    the various `mode` options (assuming `cval` is 0)::
    
        mode       |   Ext   |         Input          |   Ext
        -----------+---------+------------------------+---------
        'mirror'   | 4  3  2 | 1  2  3  4  5  6  7  8 | 7  6  5
        'nearest'  | 1  1  1 | 1  2  3  4  5  6  7  8 | 8  8  8
        'constant' | 0  0  0 | 1  2  3  4  5  6  7  8 | 0  0  0
        'wrap'     | 6  7  8 | 1  2  3  4  5  6  7  8 | 1  2  3
    

    """

    if dim == 'x':
        axis = -1
    if dim == 'y':
        axis = -2
    if dim == 'z':
        axis = -3

    data = scipy.signal.savgol_filter(dataset.data, window_length, polyorder,
                                          deriv, delta, axis, mode, cval)

    out = dataset.copy()
    if deriv == 0:
        out.data = data * dataset.units
    else:
        out.data = data
        if dataset.coord([dim]).reversed:
            out.data = out.data * (-1)**deriv
    return out


def detrend(dataset, dim='x', type='linear', bp=0, overwrite_data=False):
    """
    Wrapper of scpy.signal.detrend(). Remove linear trend along dim from dataset.
    
    Parameters
    ----------
    dataset :  |NDDataset|
        The input data
    dim : str, optional, default='x'
        The dimension, along which to detrend the data. By default this is the 'x' dimension.
    type : str among ['linear', 'constant'}, optional, default='linear'
        The type of detrending. If ``type == 'linear'`` (default),
        the result of a linear least-squares fit to `data` is subtracted from `data`.
        If ``type == 'constant'``, only the mean of `data` is subtracted.
    bp : array_like of ints, optional
        A sequence of break points. If given, an individual linear fit is
        performed for each part of `data` between two break points.
        Break points are specified as indices into `data`.
    overwrite_data : bool, optional, Default=False
        If True, perform in place detrending and avoid a copy.
    
    Returns
    -------
    ret : NDDataset
        The detrended input data.
    """
    if dim == 'x':
        axis = -1
    if dim == 'y':
        axis = -2
    if dim == 'z':
        axis = -3

    data = scipy.signal.detrend(dataset.data, axis=axis, type=type, bp=bp)
    out = dataset.copy()
    out.data = data
    return out
