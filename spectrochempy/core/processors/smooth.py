# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

__all__ = ["smooth"]
__dataset_methods__  = __all__

import numpy as np
from .. import error_, warning_, print_, debug_

def smooth(dataset, **kwargs):
    """
    Smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal (with the window size) in both ends so that
    transient parts are minimized in the begining and end part of the output data.
    
    Parameters
    ----------
    dataset :  |NDDataset| or a ndarray-like object
        Input object
        
    Other Parameters
    ----------------
    dim : str or int, optional, default='x'.
        Specify on which dimension to apply this method. If `dim` is specified as an integer it is equivalent
        to the usual `axis` numpy parameter.
    inplace : bool, optional, default=False.
        True if we make the transform inplace.  If False, the function return a new object
    len:  int, optional, default=11
        the dimension of the smoothing window; should be an odd integer
    window : str, optional, default='hanning'
        the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'.
        flat window will produce a moving average smoothing.
        
    Returns
    -------
    out : same type as input dataset
        the smoothed object
    
    TODO: implement this for NDPanel too
    """
    
    if not kwargs.pop('inplace', False):
        # default
        new = dataset.copy()
    else:
        new = dataset

    is_ndarray = False
    axis = kwargs.pop('dim', kwargs.pop('axis', -1))
    if hasattr(new, 'get_axis'):
        axis, dim = new.get_axis(axis, negative_axis=True)
    else:
        is_ndarray = True
        
    swaped = False
    if axis != -1:
        new.swapaxes(axis, -1, inplace=True)  # must be done in  place
        swaped = True

    length = kwargs.pop('length', 11)
    length = int(length / 2) * 2 + 1

    if new.shape[-1] < length:
        error_("Input vector needs to be bigger than window size.")
        return new

    if length < 3:
        return new

    window = kwargs.pop('window', 'hanning')
    
    wind = {
        'flat': np.ones,
        'hanning': np.hanning,
        'hamming': np.hamming,
        'bartlett': np.bartlett,
        'blackman': np.blackman,
    }
    if not callable(window):
        if not window in wind.keys():
            error_("Window must be a callable or a string among 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
            return new
        window = wind[window]
    
    # extend on both side to limit side effects
    dat = np.r_['-1', new.data[...,length - 1:0:-1], new.data, new.data[..., -1:-length:-1]]

    w = window(length)
    data = np.apply_along_axis(np.convolve, -1, dat, w / w.sum(), mode='valid')
    data = data[..., int(length/ 2):-int(length / 2)]

    if not is_ndarray:
        new.data = data
        new.history = f'smoothing with a window:{window.__name__} of length {length}'
    
        # restore original data order if it was swaped
        if swaped:
            new.swapaxes(axis, -1, inplace=True)  # must be done inplace
    else:
        new = data
        
    return new

if __name__ == '__main__':
    pass