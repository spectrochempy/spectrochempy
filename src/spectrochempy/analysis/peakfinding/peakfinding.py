# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
Peak finding module for spectral analysis.

This module provides functionality to identify and analyze peaks in spectroscopic data.
It wraps and extends scipy.signal's peak finding functions with additional features
specific to spectroscopic analysis, including:
- Coordinate system awareness
- Unit handling
- Integration with NDDataset objects
- Peak interpolation for improved accuracy
"""

__all__ = ["find_peaks"]

__dataset_methods__ = ["find_peaks"]

import numpy as np
import scipy

from spectrochempy.application.application import error_
from spectrochempy.application.application import warning_
from spectrochempy.core.units import DimensionalityError
from spectrochempy.core.units import Quantity

# Todo:
# find_peaks_cwt(vector, widths[, wavelet, ...]) Attempt to find the peaks in a 1-D array.
# argrelmin(data[, axis, order, mode]) 	Calculate the relative minima of data.
# argrelmax(data[, axis, order, mode]) 	Calculate the relative maxima of data.
# argrelextrema(data, comparator[, axis, ...]) 	Calculate the relative extrema of data.


def _as_peakfinding_quantity(value):
    if isinstance(value, str):
        return Quantity(value)
    if isinstance(value, Quantity):
        return value
    return None


def _format_peakfinding_units_error(name, supplied_units, coord_units, dim):
    return (
        f"Cannot use peak-finding parameter `{name}` along dimension '{dim}': "
        f"incompatible coordinate units ({supplied_units} and {coord_units}). "
        "Convert the parameter to coordinate units before retrying."
    )


def _convert_peak_parameter_to_points(name, value, step, xunits, use_coord, dim):
    """
    Convert coordinate-aware peak-finding constraints to sample-point counts.

    ``scipy.signal.find_peaks()`` expects values such as ``distance`` and
    ``width`` in sample points. When SpectroChemPy is working in coordinate
    space, we accept physical-unit inputs and convert them to the nearest
    integer point count using the current coordinate spacing.
    """
    if value is None:
        return None

    quantity = _as_peakfinding_quantity(value)
    if quantity is not None:
        if not use_coord:
            raise ValueError(
                f"Parameter `{name}` with physical units requires coordinate-aware "
                "peak finding. Provide a plain numeric value or keep `use_coord=True`."
            )
        try:
            value = quantity.to(xunits).magnitude
        except DimensionalityError as exc:
            raise ValueError(
                _format_peakfinding_units_error(
                    name,
                    quantity.units,
                    xunits,
                    dim,
                )
            ) from exc

    if isinstance(value, tuple):
        return tuple(
            _convert_peak_parameter_to_points(name, item, step, xunits, use_coord, dim)
            if item is not None
            else None
            for item in value
        )

    if use_coord:
        points = int(round(value / step))
        if value > 0 and points <= 0:
            raise ValueError(
                f"Parameter `{name}`={value} along dimension '{dim}' is smaller "
                f"than the coordinate sampling interval ({step}). Increase the "
                "value or use a plain point count with `use_coord=False`."
            )
        return points
    return value


def _normalize_interpolation_window(window_length):
    """
    Normalize the quadratic-interpolation window.

    Quadratic interpolation needs at least three samples and is symmetric only
    for odd window lengths. Values smaller than 3 disable interpolation.
    """
    if window_length is None or window_length < 3:
        return 0
    if window_length % 2 == 0:
        return window_length - 1
    return window_length


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
    Find and analyze peaks in spectroscopic data with advanced filtering options.

    This function extends scipy.signal.find_peaks by adding spectroscopy-specific
    features like coordinate system awareness and unit handling. It performs peak
    detection through local maxima analysis and supports various filtering criteria
    to identify significant peaks.

    Parameters
    ----------
    dataset : NDDataset
        Input dataset containing spectral data. Must be 1D or 2D with len(X.y) == 1.
    height : float or array-like, optional
        Minimum and/or maximum peak height criteria. Can be specified as:
        - Single value for minimum height
        - Tuple (min, max) for height range
        - Array matching x for position-dependent criteria
    window_length : int, default: 3
        Window size for peak interpolation. Must be odd.
        Larger values provide smoother interpolation but may miss narrow peaks.
    threshold : float or array-like, optional
        Minimum height difference between peak and neighboring points.
        Useful for filtering out noise-related peaks.
    distance : float, optional
        Minimum separation between peaks. Peaks closer than this are filtered
        based on their prominence.
    prominence : float or array-like, optional
        Required prominence (height above surrounding baseline) of peaks.
    width : float or array-like, optional
        Required width of peaks. Interpreted as coordinate units if use_coord=True,
        otherwise as number of points.
    wlen : int or float, optional
        Window length for prominence calculation. Affects computation speed
        for large datasets.
    rel_height : float, default: 0.5
        Relative height for width calculation (0-1 range).
    plateau_size : float or array-like, optional
        Required size of peak plateau (flat top).
    use_coord : bool, default: True
        Whether to use coordinate system units instead of array indices.

    Returns
    -------
    peaks : NDDataset
        Dataset containing identified peaks with interpolated positions and heights.
    properties : dict
        Peak properties including heights, widths, prominences, and more.
        All values use appropriate units when use_coord=True.

    Notes
    -----
    - Peak positions are refined using quadratic interpolation when window_length > 1
    - The function handles units automatically when use_coord=True
    - For noisy data, consider preprocessing with smoothing functions

    Examples
    --------
    Basic peak finding with a synthetic spectrum:
    >>> x = np.linspace(0.0, 10.0, 501)
    >>> y = np.exp(-((x - 3.0) ** 2) / 0.08) + 0.8 * np.exp(-((x - 7.0) ** 2) / 0.12)
    >>> ds = scp.NDDataset(y, coordset=[scp.Coord(x, title="x", units="cm^-1")])
    >>> peaks, props = ds.find_peaks(height=0.5)
    >>> len(peaks)
    2

    Physical-unit spacing constraints are accepted when coordinates carry units:
    >>> peaks, props = ds.find_peaks(distance="1 cm^-1", width=0.2)
    >>> len(peaks)
    2

    """
    # get the dataset
    X = dataset.squeeze()
    if X.ndim > 1:
        raise ValueError(
            "Works only for 1D NDDataset or a 2D NDdataset with `len(X.y) <= 1`",
        )
    # TODO: implement for 2D datasets (would be useful e.g., for NMR)
    # be sure that data are real (NMR case for instance)
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

        if not lastcoord.linear:
            if any(
                parameter is not None
                for parameter in (distance, width, wlen, plateau_size)
            ):
                raise ValueError(
                    "Coordinate-aware `distance`, `width`, `wlen`, and "
                    "`plateau_size` require a linear coordinate axis. "
                    "Use plain point counts with `use_coord=False` for non-linear "
                    "coordinates."
                )
            warning_(
                "The x coordinates are not linear. Peak detection still runs, "
                "but quadratic refinement and reported widths remain approximate.",
            )
            spacing = np.mean(lastcoord.spacing)
        else:
            spacing = lastcoord.spacing
        if isinstance(spacing, Quantity):
            spacing = spacing.magnitude
        step = np.abs(spacing)

    # transform coordinate-aware constraints to sample-point counts
    dim = X.dims[-1]
    distance = _convert_peak_parameter_to_points(
        "distance", distance, step, xunits, use_coord, dim
    )
    width = _convert_peak_parameter_to_points(
        "width", width, step, xunits, use_coord, dim
    )
    wlen = _convert_peak_parameter_to_points("wlen", wlen, step, xunits, use_coord, dim)
    plateau_size = _convert_peak_parameter_to_points(
        "plateau_size", plateau_size, step, xunits, use_coord, dim
    )

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

    # Check if any peaks were found
    if len(peaks) == 0:
        error_("No peaks found")
        return None, None

    # Ensure properties is a dictionary even if empty
    properties = properties if properties else {}

    out = X[peaks]

    if not use_coord:
        out.coordset = None  # remove the coordinates

    # quadratic interpolation to refine the peak maximum
    window_length = _normalize_interpolation_window(window_length)
    x_pos = []
    if window_length > 1:
        half_window = window_length // 2
        for i, peak in enumerate(peaks):
            start = max(0, peak - half_window)
            end = min(X.shape[-1], peak + half_window + 1)
            sle = slice(start, end)

            y = X.data[sle]
            if y.size < 3:
                # Too close to the border for a quadratic fit: keep the discrete peak.
                continue

            x = lastcoord.data[sle] if use_coord else np.arange(start, end)

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
