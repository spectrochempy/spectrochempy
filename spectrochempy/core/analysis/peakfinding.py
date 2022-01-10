# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory.
# ======================================================================================================================
"""
Peak finding module.

Contains wrappers of scipy.signal peak finding functions.
"""

__all__ = ["find_peaks"]

__dataset_methods__ = ["find_peaks"]

from datetime import datetime, timezone

from scipy import signal
import numpy as np

from spectrochempy.core.dataset.coord import Coord

# Todo:
# find_peaks_cwt(vector, widths[, wavelet, ...]) 	Attempt to find the peaks in a 1-D array.
# argrelmin(data[, axis, order, mode]) 	Calculate the relative minima of data.
# argrelmax(data[, axis, order, mode]) 	Calculate the relative maxima of data.
# argrelextrema(data, comparator[, axis, ...]) 	Calculate the relative extrema of data.


def find_peaks(
    dataset,
    height=None,
    window_length=3,
    threshold=None,
    distance=None,
    prominence=None,
    width=None,
    wlen=None,
    rel_height=0.5,
    plateau_size=None,
    use_coord=True,
):
    """
    Wrapper and extension of scpy.signal.find_peaks().

    Find peaks inside a 1D NDDataset based on peak properties.
    This function finds all local maxima by simple comparison of neighbouring values. Optionally, a subset of these
    peaks can be selected by specifying conditions for a peak's properties.

    ..warning::

      This function may return unexpected results for data containing NaNs.
      To avoid this, NaNs should either be removed or replaced.

    Parameters
    ----------
    dataset : |NDDataset|
        A 1D NDDataset or a 2D NDdataset with `len(X.y) == 1`.
    height : number or ndarray or sequence, optional
        Required height of peaks. Either a number, ``None``, an array matching
        `x` or a 2-element sequence of the former. The first element is
        always interpreted as the  minimal and the second, if supplied, as the
        maximal required height.
    window_length : int, default: 5
        The length of the filter window used to interpolate the maximum. window_length must be a positive odd integer.
        If set to one, the actual maximum is returned.
    threshold : number or ndarray or sequence, optional
        Required threshold of peaks, the vertical distance to its neighbouring
        samples. Either a number, ``None``, an array matching `x` or a
        2-element sequence of the former. The first element is always
        interpreted as the  minimal and the second, if supplied, as the maximal
        required threshold.
    distance : number, optional
        Required minimal horizontal distance in samples between
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
        supplied, as the maximal required width. Floats are interpreted as width
        measured along the 'x' Coord; ints are interpreted as a number of points.
    wlen : int or float, optional
        Used for calculation of the peaks prominences, thus it is only used if
        one of the arguments `prominence` or `width` is given. Floats are interpreted
        as measured along the 'x' Coord; ints are interpreted as a number of points.
        See argument len` in `peak_prominences` of the scipy documentation for a full
        description of its effects.
    rel_height : float, optional,
        Used for calculation of the peaks width, thus it is only used if `width`
        is given. See argument  `rel_height` in `peak_widths` of the scipy documentation
        for a full description of its effects.
    plateau_size : number or ndarray or sequence, optional
        Required size of the flat top of peaks in samples. Either a number,
        ``None``, an array matching `x` or a 2-element sequence of the former.
        The first element is always interpreted as the minimal and the second,
        if supplied as the maximal required plateau size. Floats are interpreted
        as measured along the 'x' Coord; ints are interpreted as a number of points.
    use_coord : bool, optional
        Set whether the x Coord (when it exists) should be used instead of indices
        for the positions and width. If True, the units of the other parameters
        are interpreted according to the coordinates.

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
              `peak_prominences` in scipy documentation for a description of their content.
        * width_heights, left_ips, right_ips
              If `width` is given, these keys are accessible. See `peak_widths`
              in scipy documentation for a description of their content.
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

    See Also
    --------
    find_peaks_cwt:
        In scipy.signal: Find peaks using the wavelet transformation.
    peak_prominences:
        In scipy.signal: Directly calculate the prominence of peaks.
    peak_widths:
        In scipy.signal: Directly calculate the width of peaks.

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

    Examples
    --------

    >>> dataset = scp.NDDataset.read("irdata/nh4y-activation.spg")
    >>> X = dataset[0, 1800.0:1300.0]
    >>> peaks, properties = X.find_peaks(height=1.5, distance=50.0, width=0.0)
    >>> len(peaks.x)
    2
    >>> peaks.x.values
    <Quantity([    1644     1455], '1 / centimeter')>
    >>> properties["peak_heights"][0]
    2.266634464263916
    >>> properties["widths"][0]
    38.73096143103294
    """

    X = dataset.squeeze()

    if X.ndim > 1:
        raise ValueError(
            "Works only for 1D NDDataset or a 2D NDdataset with `len(X.y) <= 1`"
        )

    window_length = window_length if window_length % 2 == 0 else window_length - 1

    # if the following parameters are entered as floats, the coordinates are used.
    # Else, they will be treated as indices as in scipy.signal.find_peak()

    use_coord = use_coord and X.coordset is not None

    # units
    xunits = X.x.units if use_coord else 1
    dunits = X.units if use_coord else 1

    # assume linear x coordinates when use_coord is True!
    step = np.abs(X.x.increment) if use_coord else 1

    # transform coord (if exists) to index
    distance = int(round(distance / step)) if distance is not None else None
    width = int(round(width / step)) if width is not None else None
    wlen = int(round(wlen / step)) if wlen is not None else None
    plateau_size = int(round(plateau_size / step)) if plateau_size is not None else None

    # now the distance, width ... parameters are given in data points
    peaks, properties = signal.find_peaks(
        X.data,
        height=height,
        threshold=threshold,
        distance=distance,
        prominence=prominence,
        width=width,
        wlen=wlen,
        rel_height=rel_height,
        plateau_size=plateau_size,
    )

    out = X[peaks]

    if not use_coord:
        out.coordset = None

    if window_length > 1:
        # quadratic interpolation to find the maximum
        x_pos = []
        for i, peak in enumerate(peaks):

            start = peak - window_length // 2
            end = peak + window_length // 2 + 1
            sle = slice(start, end)

            Xp = X[sle]

            y = Xp.data
            x = Xp.x.data if use_coord else range(start, end)

            coef = np.polyfit(x, y, 2)

            x_at_max = -coef[1] / (2 * coef[0])
            y_at_max = np.poly1d(coef)(x_at_max)

            out[i] = y_at_max
            x_pos.append(x_at_max)

        out.x = Coord(x_pos)
        out.x.units = X.x.units if use_coord else None

    # transform back index to coord
    if use_coord:

        for key in ["peak_heights", "width_heights", "prominences"]:
            if key in properties:
                properties[key] = [height * dunits for height in properties[key]]

        for key in (
            "left_bases",
            "right_bases",
            "left_edges",
            "right_edges",
        ):  # values are initially of int type

            if key in properties:
                properties[key] = [
                    X.x.values[int(index)]
                    for index in properties[key].astype("float64")
                ]

        def _prop(ips):
            # interpolate coord
            floor = int(np.floor(ips))
            return X.x.values[floor] + (ips - floor) * (
                X.x.values[floor + 1] - X.x.values[floor]
            )

        for key in ("left_ips", "right_ips"):  # values are float type
            if key in properties:
                properties[key] = [_prop(ips) for ips in properties[key]]

        if "widths" in properties:
            properties["widths"] = [
                np.abs(width * step) * xunits for width in properties["widths"]
            ]

        if "plateau_sizes" in properties:
            properties["plateau_sizes"] = [
                np.abs(sizes * step) * xunits for sizes in properties["plateau_sizes"]
            ]

    out.name = "peaks of " + X.name
    out.history = (
        f"{str(datetime.now(timezone.utc))}: find_peaks(): {len(peaks)} peak(s) found"
    )

    return out, properties
