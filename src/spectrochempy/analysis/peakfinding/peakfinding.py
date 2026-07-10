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

__all__ = ["PeakFindingResult", "PeakTable", "find_peaks"]

__dataset_methods__ = ["find_peaks"]

import csv
from pathlib import Path

import numpy as np
import scipy

from spectrochempy.application.application import error_
from spectrochempy.application.application import warning_
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.units import DimensionalityError
from spectrochempy.core.units import Quantity

# Todo:
# find_peaks_cwt(vector, widths[, wavelet, ...]) Attempt to find the peaks in a 1-D array.
# argrelmin(data[, axis, order, mode]) 	Calculate the relative minima of data.
# argrelmax(data[, axis, order, mode]) 	Calculate the relative maxima of data.
# argrelextrema(data, comparator[, axis, ...]) 	Calculate the relative extrema of data.


def _as_scalar(value):
    """Return a plain scalar when *value* is a one-item numpy-like object."""
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    return value


def _sequence_item(values, index):
    """Return item *index* from numpy arrays, quantities, or Python sequences."""
    try:
        return _as_scalar(values[index])
    except (TypeError, IndexError):
        return _as_scalar(values)


def _stringify_csv_value(value):
    """Serialize peak table values without adding optional dependencies."""
    value = _as_scalar(value)
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def _column_value(value, *, unit=None, as_float=False):
    """Normalize a table value for sorting or export-oriented column access."""
    value = _as_scalar(value)
    if unit is not None and hasattr(value, "to"):
        value = value.to(unit)
    if as_float:
        return float(getattr(value, "magnitude", value))
    return value


_BASE_PEAK_COLUMNS = ("index", "position", "height")
_PROPERTY_COLUMN_NAMES = {
    "peak_heights": "peak_height",
    "prominences": "prominence",
    "widths": "width",
    "width_heights": "width_height",
    "left_bases": "left_base",
    "right_bases": "right_base",
    "left_ips": "left_ip",
    "right_ips": "right_ip",
    "plateau_sizes": "plateau_size",
    "left_edges": "left_edge",
    "right_edges": "right_edge",
}


def _peak_rows(peaks, properties, *, raw_property_names=False):
    """Build row dictionaries from peak positions, heights, and properties."""
    if peaks is None:
        return []

    n_peaks = len(peaks)
    positions = _peak_positions(peaks)
    heights = np.ravel(peaks.data)
    if peaks.units is not None:
        heights = heights * peaks.units
    rows = []

    per_peak_properties = {}
    for key, values in properties.items():
        try:
            if len(values) == n_peaks:
                per_peak_properties[key] = values
        except TypeError:
            continue

    for index in range(n_peaks):
        row = {
            "index": index,
            "position": _sequence_item(positions, index),
            "height": _sequence_item(heights, index),
        }
        for key, values in per_peak_properties.items():
            column = key if raw_property_names else _PROPERTY_COLUMN_NAMES.get(key, key)
            row[column] = _sequence_item(values, index)
        rows.append(row)

    return rows


def _peak_positions(peaks):
    """Return peak positions from coordinates or point indexes."""
    if peaks.coordset is None:
        return np.arange(len(peaks))
    coord = peaks.coordset[peaks.dims[-1]]
    return coord.values


def _fieldnames(rows, *, columns=_BASE_PEAK_COLUMNS):
    """Return stable field names from base columns plus row-specific columns."""
    fieldnames = list(columns)
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames


class PeakTable:
    """
    Dependency-light tabular view of detected peaks.

    Parameters
    ----------
    peaks : NDDataset or None
        Dataset containing identified peak positions and heights.
    properties : dict or None
        Peak properties returned by :func:`scipy.signal.find_peaks`.

    Notes
    -----
    ``PeakTable`` uses user-facing singular column names such as
    ``peak_height``, ``prominence`` and ``width``. The raw SciPy property
    dictionary remains available on :class:`PeakFindingResult`.
    """

    __slots__ = ("peaks", "properties", "_rows")

    def __init__(self, peaks, properties=None, rows=None):
        self.peaks = peaks
        self.properties = dict(properties or {})
        self._rows = None if rows is None else [dict(row) for row in rows]

    def __len__(self):
        if self._rows is not None:
            return len(self._rows)
        return 0 if self.peaks is None else len(self.peaks)

    def __iter__(self):
        return iter(self.to_dict())

    def __repr__(self):
        return f"PeakTable(n_peaks={len(self)})"

    @property
    def columns(self):
        """Return stable base columns plus currently available optional columns."""
        return tuple(_fieldnames(self.to_dict()))

    def to_dict(self):
        """
        Return peak information as a list of dictionaries.

        Each dictionary contains ``index``, ``position``, ``height`` and any
        available per-peak properties. Values keep their native units when they
        are unit-bearing quantities.
        """
        if self._rows is not None:
            return [dict(row) for row in self._rows]
        return _peak_rows(self.peaks, self.properties)

    def head(self, n):
        """
        Return the first *n* rows as a new ``PeakTable``.

        Parameters
        ----------
        n : int
            Number of rows to keep.
        """
        return PeakTable(None, None, rows=self.to_dict()[:n])

    def column(self, name, *, unit=None, as_float=False):
        """
        Return a column as a list of values.

        Parameters
        ----------
        name : str
            Column name.
        unit : str or Quantity, optional
            Target unit for unit-aware values.
        as_float : bool, default=False
            If True, return plain floats.
        """
        if name not in self.columns:
            raise KeyError(f"Unknown peak-table column: {name!r}")
        return [
            _column_value(row[name], unit=unit, as_float=as_float)
            for row in self.to_dict()
        ]

    def sort_by(self, name, *, reverse=False, unit=None):
        """
        Return a new ``PeakTable`` sorted by the given column.

        Parameters
        ----------
        name : str
            Column name used as sorting key.
        reverse : bool, default=False
            If True, sort in descending order.
        unit : str or Quantity, optional
            Target unit for unit-aware values before comparison.
        """
        if name not in self.columns:
            raise KeyError(f"Unknown peak-table column: {name!r}")
        rows = self.to_dict()
        rows.sort(
            key=lambda row: _column_value(row[name], unit=unit, as_float=True),
            reverse=reverse,
        )
        return PeakTable(None, None, rows=rows)

    def top(self, n, *, by, reverse=True, unit=None):
        """
        Return the top *n* rows according to a given column.

        Parameters
        ----------
        n : int
            Number of rows to keep.
        by : str
            Column name used as ranking key.
        reverse : bool, default=True
            If True, keep the largest values first.
        unit : str or Quantity, optional
            Target unit for unit-aware values before comparison.
        """
        return self.sort_by(by, reverse=reverse, unit=unit).head(n)

    def to_csv(self, path, *, delimiter=","):
        """
        Write peak information to a CSV file.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination path.
        delimiter : str, default=','
            CSV delimiter.

        Returns
        -------
        pathlib.Path
            The written path.
        """
        path = Path(path)
        rows = self.to_dict()
        fieldnames = _fieldnames(rows)

        with path.open("w", newline="", encoding="utf-8") as fid:
            writer = csv.DictWriter(fid, fieldnames=fieldnames, delimiter=delimiter)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {key: _stringify_csv_value(row.get(key, "")) for key in fieldnames}
                )

        return path


class PeakFindingResult:
    """
    Structured result returned by :func:`find_peaks`.

    Parameters
    ----------
    peaks : NDDataset or None
        Dataset containing identified peak positions and heights.
    properties : dict or None
        Peak properties returned by :func:`scipy.signal.find_peaks`.

    Notes
    -----
    The object is intentionally dependency-light. It exposes Python-native
    table helpers and does not require pandas.
    """

    __slots__ = ("peaks", "properties")

    def __init__(self, peaks, properties=None):
        self.peaks = peaks
        self.properties = dict(properties or {})

    def __len__(self):
        return 0 if self.peaks is None else len(self.peaks)

    def __iter__(self):
        """Allow ``peaks, properties = result`` for easy migration."""
        yield self.peaks
        yield self.properties

    def __repr__(self):
        return f"PeakFindingResult(n_peaks={len(self)})"

    @property
    def table(self):
        """Return a dependency-light tabular view of the detected peaks."""
        return PeakTable(self.peaks, self.properties)

    def to_dict(self):
        """
        Return peak information as a list of dictionaries.

        Each dictionary contains ``index``, ``position``, ``height``, and any
        per-peak properties whose length matches the number of detected peaks.
        Values keep their native units when they are unit-bearing quantities.
        """
        return _peak_rows(self.peaks, self.properties, raw_property_names=True)

    def to_csv(self, path, *, delimiter=","):
        """
        Write peak information to a CSV file.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination path.
        delimiter : str, default=','
            CSV delimiter.

        Returns
        -------
        pathlib.Path
            The written path.
        """
        path = Path(path)
        rows = self.to_dict()
        fieldnames = _fieldnames(rows)

        with path.open("w", newline="", encoding="utf-8") as fid:
            writer = csv.DictWriter(fid, fieldnames=fieldnames, delimiter=delimiter)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {key: _stringify_csv_value(row.get(key, "")) for key in fieldnames}
                )

        return path


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
    as_result=False,
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
    as_result : bool, default: False
        If True, return a :class:`PeakFindingResult` object. If False, preserve
        the historical ``(peaks, properties)`` tuple return.

    Returns
    -------
    peaks : NDDataset or None
        Dataset containing identified peaks with interpolated positions and
        heights. Returned when ``as_result=False``.
    properties : dict or None
        Peak properties including heights, widths, prominences, and more.
        All values use appropriate units when use_coord=True. Returned when
        ``as_result=False``.
    result : PeakFindingResult
        Structured peak finding result. Returned when ``as_result=True``.

    Notes
    -----
    - Peak positions are refined using quadratic interpolation when window_length > 1
    - The function handles units automatically when use_coord=True
    - For noisy data, consider preprocessing with smoothing functions

    Examples
    --------
    Basic peak finding with a synthetic spectrum:

    >>> x = scp.Coord.linspace(0.0, 10.0, 501, title="x", units="cm^-1")
    >>> y = (scp.gaussian(x, ampl=1.0, pos=3.0, width=0.5, normalized=False)
    ...    + scp.gaussian(x, ampl=0.8, pos=7.0, width=0.6, normalized=False))
    >>> ds = scp.NDDataset(y, coordset=[x])
    >>> peaks, props = ds.find_peaks(height=0.5)
    >>> len(peaks)
    2

    Physical-unit spacing constraints are accepted when coordinates carry units:

    >>> x = scp.Coord.linspace(0.0, 10.0, 501, title="x", units="cm^-1")
    >>> y = (scp.gaussian(x, ampl=1.0, pos=3.0, width=0.5, normalized=False)
    ...    + scp.gaussian(x, ampl=0.8, pos=7.0, width=0.6, normalized=False))
    >>> ds = scp.NDDataset(y, coordset=[x])
    >>> peaks, props = ds.find_peaks(distance="1 cm^-1", width=0.2)
    >>> len(peaks)
    2

    Return a structured result when a tabular/export representation is useful:

    >>> x = scp.Coord.linspace(0.0, 10.0, 501, title="x", units="cm^-1")
    >>> y = (scp.gaussian(x, ampl=1.0, pos=3.0, width=0.5, normalized=False)
    ...    + scp.gaussian(x, ampl=0.8, pos=7.0, width=0.6, normalized=False))
    >>> ds = scp.NDDataset(y, coordset=[x])
    >>> result = ds.find_peaks(height=0.5, as_result=True)
    >>> rows = result.to_dict()
    >>> len(rows)
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
        if as_result:
            return PeakFindingResult(None, None)
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

    if as_result:
        return PeakFindingResult(out, properties)

    return out, properties
