# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

__all__ = ['find_peaks']

__dataset_methods__  = __all__

import scipy.signal
from datetime import datetime

"""wrappers of scipy.signal peak finding functions"""

#Todo:
#find_peaks_cwt(vector, widths[, wavelet, ...]) 	Attempt to find the peaks in a 1-D array.
#argrelmin(data[, axis, order, mode]) 	Calculate the relative minima of data.
#argrelmax(data[, axis, order, mode]) 	Calculate the relative maxima of data.
#argrelextrema(data, comparator[, axis, ...]) 	Calculate the relative extrema of data.

def find_peaks(X, height=None, threshold=None, distance=None,
               prominence=None, width=None, wlen=None, rel_height=0.5,
               plateau_size=None):
    """
    Wrapper of scpy.signal.find_peaks(). Find peaks inside a 1D NDDataset based on peak properties.
    This function finds all local maxima by simple comparison of neighbouring values. Optionally, a subset of these
    peaks can be selected by specifying conditions for a peak's properties.
    
    Parameters
    ----------
    x : |NDDataset|
        A 1D NDDataset or a 2D NDdataset with `len(X.y) == 1`
    height : number or ndarray or sequence, optional
        Required height of peaks. Either a number, ``None``, an array matching
        `x` or a 2-element sequence of the former. The first element is
        always interpreted as the  minimal and the second, if supplied, as the
        maximal required height.
    threshold : number or ndarray or sequence, optional
        Required threshold of peaks, the vertical distance to its neighbouring
        samples. Either a number, ``None``, an array matching `x` or a
        2-element sequence of the former. The first element is always
        interpreted as the  minimal and the second, if supplied, as the maximal
        required threshold.
    distance : number, optional
        Required minimal horizontal distance (>= 1) in samples between
        neighbouring peaks. Smaller peaks are removed first until the condition
        is fulfilled for all remaining peaks.
    prominence : number or ndarray or sequence, optional
        Required prominence of peaks. Either a number, ``None``, an array
        matching `x` or a 2-element sequence of the former. The first
        element is always interpreted as the  minimal and the second, if
        supplied, as the maximal required prominence.
    width : number or ndarray or sequence, optional
        Required width of peaks in samples. Either a number, ``None``, an array
        matching `x` or a 2-element sequence of the former. The first
        element is always interpreted as the  minimal and the second, if
        supplied, as the maximal required width.
    wlen : int, optional
        Used for calculation of the peaks prominences, thus it is only used if
        one of the arguments `prominence` or `width` is given. See argument
        `wlen` in `peak_prominences` for a full description of its effects.
    rel_height : float, optional
        Used for calculation of the peaks width, thus it is only used if `width`
        is given. See argument  `rel_height` in `peak_widths` for a full
        description of its effects.
    plateau_size : number or ndarray or sequence, optional
        Required size of the flat top of peaks in samples. Either a number,
        ``None``, an array matching `x` or a 2-element sequence of the former.
        The first element is always interpreted as the minimal and the second,
        if supplied as the maximal required plateau size.
    
    Returns
    -------
    peaks : ndarray
        Indices of peaks in `x` that satisfy all given conditions.
    
    properties : dict
        A dictionary containing properties of the returned peaks which were
        calculated as intermediate results during evaluation of the specified
        conditions:
    
        * peak_heights
              If `height` is given, the height of each peak in `x`.
        * left_thresholds, right_thresholds
              If `threshold` is given, these keys contain a peaks vertical
              distance to its neighbouring samples.
        * prominences, right_bases, left_bases
              If `prominence` is given, these keys are accessible. See
              `peak_prominences` for a description of their content.
        * width_heights, left_ips, right_ips
              If `width` is given, these keys are accessible. See `peak_widths`
              for a description of their content.
        * plateau_sizes, left_edges', 'right_edges'
              If `plateau_size` is given, these keys are accessible and contain
              the indices of a peak's edges (edges are still part of the
              plateau) and the calculated plateau sizes.
              
        To calculate and return properties without excluding peaks, provide the
        open interval ``(None, None)`` as a value to the appropriate argument
        (excluding `distance`).
    
    Warns
    -----
    PeakPropertyWarning
        Raised if a peak's properties have unexpected values (see
        `peak_prominences` and `peak_widths`).
    
    Warnings
    --------
    This function may return unexpected results for data containing NaNs. To
    avoid this, NaNs should either be removed or replaced.
    
    See Also
    --------
    find_peaks_cwt:
        Find peaks using the wavelet transformation.
    peak_prominences:
        Directly calculate the prominence of peaks.
    peak_widths:
        Directly calculate the width of peaks.
    
    Notes
    -----
    In the context of this function, a peak or local maximum is defined as any
    sample whose two direct neighbours have a smaller amplitude. For flat peaks
    (more than one sample of equal amplitude wide) the index of the middle
    sample is returned (rounded down in case the number of samples is even).
    For noisy signals the peak locations can be off because the noise might
    change the position of local maxima. In those cases consider smoothing the
    signal before searching for peaks or use other peak finding and fitting
    methods (like `find_peaks_cwt`).
    Some additional comments on specifying conditions:
    
    * Almost all conditions (excluding `distance`) can be given as half-open or
      closed intervals, e.g ``1`` or ``(1, None)`` defines the half-open
      interval :math:`[1, \\infty]` while ``(None, 1)`` defines the interval
      :math:`[-\\infty, 1]`. The open interval ``(None, None)`` can be specified
      as well, which returns the matching properties without exclusion of peaks.
    * The border is always included in the interval used to select valid peaks.
    * For several conditions the interval borders can be specified with
      arrays matching `x` in shape which enables dynamic constrains based on
      the sample position.
    * The conditions are evaluated in the following order: `plateau_size`,
      `height`, `threshold`, `distance`, `prominence`, `width`. In most cases
      this order is the fastest one because faster operations are applied first
      to reduce the number of peaks that need to be evaluated later.
    * While indices in `peaks` are guaranteed to be at least `distance` samples
      apart, edges of flat peaks may be closer than the allowed `distance`.
    * Use `wlen` to reduce the time it takes to evaluate the conditions for
      `prominence` or `width` if `x` is large or has many local maxima
      (see `peak_prominences`).

    """

    if X.ndim > 2 or (X.ndim == 2 and len(X.y) > 1):
        raise ValueError("Works only for 1D NDDataset or a 2D NDdataset with `len(X.y) <= 1`")

    peaks, properties = scipy.signal.find_peaks(X.data.squeeze(), height=height, threshold=threshold,
                                                distance=distance, prominence=prominence, width=width, wlen=wlen,
                                                rel_height=rel_height, plateau_size=plateau_size)

    if X.ndim == 1:
        out = X[peaks]
    if X.ndim == 2:
        out = X[:, peaks]

    #Todo: check why slicing by indexes removes the x coord
    out.x = X.x[peaks]
    out.name = 'peaks of ' + X.name
    out.history[-1] = str(datetime.now()) + f': find_peaks(): {len(peaks)} peak(s) found'

    return out, properties

