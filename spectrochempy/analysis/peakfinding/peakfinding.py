# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Peak finding module.

Contains wrappers of `scipy.signal` peak finding functions.
"""

__all__ = ["find_peaks"]

__dataset_methods__ = ["find_peaks"]

import numpy as np
import scipy

from spectrochempy.application import warning_
from spectrochempy.core.units import Quantity

# Todo:
# find_peaks_cwt(vector, widths[, wavelet, ...]) Attempt to find the peaks in a 1-D array.
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
    Wrapper and extension of `scpy.signal.find_peaks`\ .

    Find peaks inside a 1D `NDDataset` based on peak properties.
    This function finds all local maxima by simple comparison of neighbouring values.
    Optionally, a subset of these
    peaks can be selected by specifying conditions for a peak's properties.

    .. warning::

        This function may return unexpected results for data containing NaNs.
        To avoid this, NaNs should either be removed or replaced.

    Parameters
    ----------
    dataset : `NDDataset`
        A 1D NDDataset or a 2D NDdataset with `len(X.y) == 1` .
    height : `float` or :term:`array-like`\ , optional, default: `None`
        Required height of peaks. Either a number, `None` , an array matching
        `x` or a 2-element sequence of the former. The first element is
        always interpreted as the minimal and the second, if supplied, as the
        maximal required height.
    window_length : `int`, default: 5
        The length of the filter window used to interpolate the maximum. window_length
        must be a positive odd integer.
        If set to one, the actual maximum is returned.
    threshold : `float` or :term:`array-like`\ , optional
        Required threshold of peaks, the vertical distance to its neighbouring
        samples. Either a number, `None` , an array matching `x` or a
        2-element sequence of the former. The first element is always
        interpreted as the  minimal and the second, if supplied, as the maximal
        required threshold.
    distance : `float`\ , optional
        Required minimal horizontal distance in samples between
        neighbouring peaks. Smaller peaks are removed first until the condition
        is fulfilled for all remaining peaks.
    prominence : `float` or :term:`array-like`\ , optional
        Required prominence of peaks. Either a number, `None` , an array
        matching `x` or a 2-element sequence of the former. The first
        element is always interpreted as the  minimal and the second, if
        supplied, as the maximal required prominence.
    width : `float` or :term:`array-like`\ , optional
        Required width of peaks in samples. Either a number, `None` , an array
        matching `x` or a 2-element sequence of the former. The first
        element is always interpreted as the  minimal and the second, if
        supplied, as the maximal required width. Floats are interpreted as width
        measured along the 'x' Coord; ints are interpreted as a number of points.
    wlen : `int` or `float`, optional
        Used for calculation of the peaks prominences, thus it is only used if
        one of the arguments `prominence` or `width` is given. Floats are interpreted
        as measured along the 'x' Coord; ints are interpreted as a number of points.
        See argument len` in `peak_prominences` of the scipy documentation for a full
        description of its effects.
    rel_height : `float`, optional,
        Used for calculation of the peaks width, thus it is only used if `width`
        is given. See argument  `rel_height` in `peak_widths` of the scipy documentation
        for a full description of its effects.
    plateau_size : `float` or :term:`array-like`\ , optional
        Required size of the flat top of peaks in samples. Either a number,
        `None` , an array matching `x` or a 2-element sequence of the former.
        The first element is always interpreted as the minimal and the second,
        if supplied as the maximal required plateau size. Floats are interpreted
        as measured along the 'x' Coord; ints are interpreted as a number of points.
    use_coord : `bool`\ , optional
        Set whether the x Coord (when it exists) should be used instead of indices
        for the positions and width. If True, the units of the other parameters
        are interpreted according to the coordinates.

    Returns
    -------
    peaks : `~numpy.ndarray`
        Indices of peaks in `dataset` that satisfy all given conditions.

    properties : `dict`
        A dictionary containing properties of the returned peaks which were
        calculated as intermediate results during evaluation of the specified
        conditions:

        * ``peak_heights``
            If `height` is given, the height of each peak in `dataset`\  .
        * ``left_thresholds``\ , ``right_thresholds``
            If `threshold` is given, these keys contain a peaks vertical
            distance to its neighbouring samples.
        * ``prominences``\ , ``right_bases``\ , ``left_bases``
            If `prominence` is given, these keys are accessible. See
            `scipy.signal.peak_prominences` for a
            full description of their content.
        * ``width_heights``\ , ``left_ips``\ , ``right_ips``
            If `width` is given, these keys are accessible. See
            `scipy.signal.peak_widths` for a full description of their content.
        * plateau_sizes, left_edges', 'right_edges'
            If `plateau_size` is given, these keys are accessible and contain
            the indices of a peak's edges (edges are still part of the
            plateau) and the calculated plateau sizes.

        To calculate and return properties without excluding peaks, provide the
        open interval `(None, None)` as a value to the appropriate argument
        (excluding `distance`\ ).

    Warns
    -----
    PeakPropertyWarning
        Raised if a peak's properties have unexpected values (see
        `peak_prominences` and `peak_widths` ).

    See Also
    --------
    find_peaks_cwt:
        In `scipy.signal`: Find peaks using the wavelet transformation.
    peak_prominences:
        In `scipy.signal`: Directly calculate the prominence of peaks.
    peak_widths:
        In `scipy.signal`: Directly calculate the width of peaks.

    Notes
    -----
    In the context of this function, a peak or local maximum is defined as any
    sample whose two direct neighbours have a smaller amplitude. For flat peaks
    (more than one sample of equal amplitude wide) the index of the middle
    sample is returned (rounded down in case the number of samples is even).
    For noisy signals the peak locations can be off because the noise might
    change the position of local maxima. In those cases consider smoothing the
    signal before searching for peaks or use other peak finding and fitting
    methods (like `scipy.signal.find_peaks_cwt` ).

    Some additional comments on specifying conditions:

    * Almost all conditions (excluding `distance`\ ) can be given as half-open or
      closed intervals, e.g `1` or `(1, None)` defines the half-open
      interval :math:`[1, \\infty]` while `(None, 1)` defines the interval
      :math:`[-\\infty, 1]`\ . The open interval `(None, None)` can be specified
      as well, which returns the matching properties without exclusion of peaks.
    * The border is always included in the interval used to select valid peaks.
    * For several conditions the interval borders can be specified with
      arrays matching `dataset` in shape which enables dynamic constrains based on
      the sample position.
    * The conditions are evaluated in the following order: `plateau_size` ,
      `height` , `threshold` , `distance` , `prominence` , `width` . In most cases
      this order is the fastest one because faster operations are applied first
      to reduce the number of peaks that need to be evaluated later.
    * While indices in `peaks` are guaranteed to be at least `distance` samples
      apart, edges of flat peaks may be closer than the allowed `distance` .
    * Use `wlen` to reduce the time it takes to evaluate the conditions for
      `prominence` or `width` if `dataset` is large or has many local maxima
      (see `scipy.signal.peak_prominences`\ ).

    Examples
    --------

    >>> dataset = scp.read("irdata/nh4y-activation.spg")
    >>> X = dataset[0, 1800.0:1300.0]
    >>> peaks, properties = X.find_peaks(height=1.5, distance=50.0, width=0.0)
    >>> len(peaks.x)
    2
    >>> peaks.x.values
    <Quantity([    1644     1455], '1 / centimeter')>
    >>> properties["peak_heights"][0]
    <Quantity(2.26663446, 'absorbance')>
    >>> properties["widths"][0]
    <Quantity(38.729003, '1 / centimeter')>
    """

    # get the dataset
    X = dataset.squeeze()
    if X.ndim > 1:
        raise ValueError(
            "Works only for 1D NDDataset or a 2D NDdataset with `len(X.y) <= 1`"
        )
    # TODO: implement for 2D datasets (would be useful e.g., for NMR)
    # be sure that data are real (NMR case for instance)
    if X.is_complex or X.is_quaternion:
        X = X.real

    # Check if we can work with the coordinates
    use_coord = use_coord and X.coordset is not None

    # init variable in case we do not use coordinates
    lastcoord = None
    xunits = 1
    dunits = 1
    step = 1

    if use_coord:
        # We will use the last coordinates (but if the data were transposed or sliced,
        # the name can be something else than 'x')
        lastcoord = X.coordset[X.dims[-1]]

        # units
        xunits = lastcoord.units if lastcoord.units is not None else 1
        dunits = X.units if X.units is not None else 1

        # assume linear x coordinates
        # TODO: what if the coordinates are not linear?
        if not lastcoord.linear:
            warning_(
                "The x coordinates are not linear. " "The peak finding might be wrong."
            )
            spacing = np.mean(lastcoord.spacing)
        else:
            spacing = lastcoord.spacing
        if isinstance(spacing, Quantity):
            spacing = spacing.magnitude
        step = np.abs(spacing)

    # transform coord (if exists) to index
    # TODO: allow units for distance, width, wlen, plateau_size
    distance = int(round(distance / step)) if distance is not None else None
    width = int(round(width / step)) if width is not None else None
    wlen = int(round(wlen / step)) if wlen is not None else None
    plateau_size = int(round(plateau_size / step)) if plateau_size is not None else None

    # now the distance, width ... parameters are given in data points
    peaks, properties = scipy.signal.find_peaks(
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
        out.coordset = None  # remove the coordinates

    # quadratic interpolation to find the maximum
    window_length = window_length if window_length % 2 == 0 else window_length - 1
    x_pos = []
    if window_length > 1:
        for i, peak in enumerate(peaks):
            start = peak - window_length // 2
            end = peak + window_length // 2 + 1
            sle = slice(start, end)

            y = X.data[sle]
            x = lastcoord.data[sle] if use_coord else range(start, end)

            coef = np.polyfit(x, y, 2)

            x_at_max = -coef[1] / (2 * coef[0])
            y_at_max = np.poly1d(coef)(x_at_max)

            out[i] = y_at_max
            if not use_coord:
                x_pos.append(x_at_max)
            else:
                out.coordset(out.dims[-1])[i] = x_at_max
    if x_pos and not use_coord:
        from spectrochempy.core.dataset.coord import Coord

        out.coordset = Coord(x_pos)

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
                    lastcoord.values[int(index)]
                    for index in properties[key].astype("float64")
                ]

        def _prop(ips):
            # interpolate coord
            floor = int(np.floor(ips))
            return lastcoord.values[floor] + (ips - floor) * (
                lastcoord.values[floor + 1] - lastcoord.values[floor]
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
    out.history = f"find_peaks(): {len(peaks)} peak(s) found"

    return out, properties
