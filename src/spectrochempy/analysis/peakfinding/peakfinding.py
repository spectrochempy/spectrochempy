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
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path

import numpy as np
import scipy
from scipy import ndimage

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


_BASE_PEAK_COLUMNS = (
    "index",
    "position",
    "height",
)
_SAMPLE_PEAK_COLUMNS = (
    "sample_index",
    "sample_position",
    "sample_height",
)
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


def _copy_values(values):
    """Return a detached copy when the object supports copying."""
    if values is None:
        return None
    if hasattr(values, "copy"):
        return values.copy()
    return list(values)


def _peak_rows(
    peaks,
    properties,
    *,
    sample_index=None,
    sample_position=None,
    sample_height=None,
    raw_property_names=False,
):
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
        if sample_index is not None:
            row["sample_index"] = _sequence_item(sample_index, index)
        if sample_position is not None:
            row["sample_position"] = _sequence_item(sample_position, index)
        if sample_height is not None:
            row["sample_height"] = _sequence_item(sample_height, index)
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


def _has_sample_fields(sample_index, sample_position, sample_height):
    """Return True when all discrete sample fields are available."""
    return all(
        value is not None for value in (sample_index, sample_position, sample_height)
    )


class PeakTable:
    """
    Dependency-light tabular view of detected peaks.

    Parameters
    ----------
    peaks : NDDataset, PeakTable or None
        Dataset containing identified peaks for 1D results, or a table-backed
        representation for 2D results.
    properties : dict or None
        Peak properties returned by :func:`scipy.signal.find_peaks`.

    columns : sequence of str, optional
        Explicit stable column schema for row-backed multidimensional results.
    history_entries : list of dict, optional
        Detached structured history entries for row-backed results.

    Notes
    -----
    ``PeakTable`` uses user-facing singular column names such as
    ``peak_height``, ``prominence`` and ``width``. The raw SciPy property
    dictionary remains available on :class:`PeakFindingResult`.
    """

    __slots__ = (
        "peaks",
        "properties",
        "sample_index",
        "sample_position",
        "sample_height",
        "_rows",
        "_columns",
        "_history_entries",
        "_history",
    )

    def __init__(
        self,
        peaks,
        properties=None,
        rows=None,
        *,
        sample_index=None,
        sample_position=None,
        sample_height=None,
        columns=None,
        history_entries=None,
        history=None,
    ):
        self.peaks = peaks
        self.properties = dict(properties or {})
        self.sample_index = _copy_values(sample_index)
        self.sample_position = _copy_values(sample_position)
        self.sample_height = _copy_values(sample_height)
        self._rows = None if rows is None else [dict(row) for row in rows]
        self._columns = None if columns is None else tuple(columns)
        self._history_entries = deepcopy(history_entries)
        self._history = tuple(history or ())

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
        if self._columns is not None:
            return tuple(_fieldnames(self.to_dict(), columns=self._columns))
        columns = list(_BASE_PEAK_COLUMNS)
        if _has_sample_fields(
            self.sample_index,
            self.sample_position,
            self.sample_height,
        ):
            columns.extend(_SAMPLE_PEAK_COLUMNS)
        return tuple(_fieldnames(self.to_dict(), columns=columns))

    @property
    def history_entries(self):
        """Return detached structured history entries for the result table."""
        if self._history_entries is not None:
            return deepcopy(self._history_entries)
        if self.peaks is not None:
            return self.peaks.history_entries
        return []

    @property
    def history(self):
        """Return the read-only rendered history for the result table."""
        if self._history:
            return self._history
        if self.peaks is not None:
            return self.peaks.history
        return ()

    def to_dict(self):
        """
        Return peak information as a list of dictionaries.

        Each dictionary contains ``index``, ``position``, ``height``, available
        discrete sample fields and any available per-peak properties. Values
        keep their native units when they are unit-bearing quantities.
        """
        if self._rows is not None:
            return [dict(row) for row in self._rows]
        return _peak_rows(
            self.peaks,
            self.properties,
            sample_index=self.sample_index,
            sample_position=self.sample_position,
            sample_height=self.sample_height,
        )

    def head(self, n):
        """
        Return the first *n* rows as a new ``PeakTable``.

        Parameters
        ----------
        n : int
            Number of rows to keep.
        """
        return PeakTable(
            None,
            None,
            rows=self.to_dict()[:n],
            columns=self._columns,
            history_entries=self.history_entries,
            history=self.history,
        )

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
        return PeakTable(
            None,
            None,
            rows=rows,
            columns=self._columns,
            history_entries=self.history_entries,
            history=self.history,
        )

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
        fieldnames = self.columns

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
    peaks : NDDataset, PeakTable or None
        Dataset containing identified peak positions and heights for 1D results;
        a :class:`PeakTable` for 2D results.
    properties : dict or None
        Peak properties returned by :func:`scipy.signal.find_peaks`.

    Notes
    -----
    The object is intentionally dependency-light. It exposes Python-native
    table helpers and does not require pandas.
    """

    __slots__ = (
        "peaks",
        "properties",
        "sample_index",
        "sample_position",
        "sample_height",
        "analysis_dims",
        "parameters",
    )

    def __init__(
        self,
        peaks,
        properties=None,
        *,
        sample_index=None,
        sample_position=None,
        sample_height=None,
        analysis_dims=None,
        parameters=None,
    ):
        self.peaks = peaks
        self.properties = dict(properties or {})
        self.sample_index = _copy_values(sample_index)
        self.sample_position = _copy_values(sample_position)
        self.sample_height = _copy_values(sample_height)
        self.analysis_dims = None if analysis_dims is None else tuple(analysis_dims)
        self.parameters = deepcopy(parameters or {})

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
        if isinstance(self.peaks, PeakTable):
            return self.peaks
        return PeakTable(
            self.peaks,
            self.properties,
            sample_index=self.sample_index,
            sample_position=self.sample_position,
            sample_height=self.sample_height,
        )

    def to_dict(self):
        """
        Return peak information as a list of dictionaries.

        Each dictionary contains ``index`` (the row identifier), refined
        ``position`` and ``height``, available discrete sample fields, plus any
        per-peak properties whose length matches the number of detected peaks.
        Values keep their native units when they are unit-bearing quantities.
        """
        if isinstance(self.peaks, PeakTable):
            return self.peaks.to_dict()
        return _peak_rows(
            self.peaks,
            self.properties,
            sample_index=self.sample_index,
            sample_position=self.sample_position,
            sample_height=self.sample_height,
            raw_property_names=True,
        )

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
        if isinstance(self.peaks, PeakTable):
            return self.peaks.to_csv(path, delimiter=delimiter)
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

    @property
    def history_entries(self):
        """Return detached history entries for the result."""
        return self.table.history_entries

    @property
    def history(self):
        """Return the rendered history for the result."""
        return self.table.history


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


def _resolve_peakfinding_dims(dataset, dims):
    """Resolve a homogeneous sequence of dimension names or indices."""
    if isinstance(dims, (str, bytes)) or not isinstance(dims, (tuple, list)):
        raise TypeError("dims must be a sequence of dimension names or indices")
    if len(dims) != dataset.ndim:
        raise ValueError(
            f"dims must contain exactly {dataset.ndim} dimension(s) for this dataset"
        )
    if not dims:
        raise ValueError("dims must not be empty")

    accepted_names = all(isinstance(dim, str) for dim in dims)
    accepted_indices = all(
        isinstance(dim, (int, np.integer)) and not isinstance(dim, (bool, np.bool_))
        for dim in dims
    )
    if not accepted_names and not accepted_indices:
        unsupported = [
            dim
            for dim in dims
            if not isinstance(dim, str)
            and not (
                isinstance(dim, (int, np.integer))
                and not isinstance(dim, (bool, np.bool_))
            )
        ]
        if unsupported:
            raise TypeError("dims elements must be dimension names or integer indices")
        raise ValueError("dims must use either names or indices, not a mixture")

    name_dims = accepted_names

    resolved = []
    for dim in dims:
        if name_dims:
            if dim not in dataset.dims:
                raise ValueError(f"Unknown dimension name: {dim!r}")
            axis = dataset.dims.index(dim)
        else:
            axis = int(dim)
            if axis < 0:
                axis += dataset.ndim
            if axis < 0 or axis >= dataset.ndim:
                raise ValueError(f"Dimension index {dim!r} is out of range")
        if axis in [item[0] for item in resolved]:
            raise ValueError("dims must contain distinct dimensions")
        resolved.append((axis, dataset.dims[axis]))
    return tuple(resolved)


def _peakfinding_working_data(dataset):
    """Return a detached float64 work array and its scientific validity mask."""
    source = np.asarray(dataset.data)
    if source.dtype.kind not in "iuf" or source.dtype.kind == "b":
        raise ValueError("2D peak finding requires real numeric data")
    if source.ndim != 2:
        raise ValueError("2D peak finding requires a two-dimensional dataset")

    try:
        values = np.array(source, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("2D peak finding requires real numeric data") from exc

    source_mask = dataset.mask
    if source_mask is None:
        mask = np.zeros(source.shape, dtype=bool)
    else:
        try:
            mask = np.broadcast_to(np.asarray(source_mask, dtype=bool), source.shape)
        except ValueError as exc:
            raise ValueError("dataset mask must match the data shape") from exc
    valid = np.array(~mask & np.isfinite(values), dtype=bool, copy=True)
    values[~valid] = -np.inf
    return values, valid


def _peakfinding_coordinates(dataset, resolved_dims, *, use_coord):
    """Resolve and validate coordinate arrays for the selected source axes."""
    coordinates = {}
    for axis, dim in resolved_dims:
        coord = dataset.coord(dim)
        if coord is None or not coord.has_data:
            if use_coord:
                raise ValueError(
                    f"Coordinate-aware 2D peak finding requires coordinates for {dim!r}"
                )
            coordinates[dim] = None
            continue

        raw = np.asarray(coord.data)
        if raw.ndim != 1 or raw.size != dataset.shape[axis]:
            raise ValueError(f"Coordinate for dimension {dim!r} has an invalid shape")

        if use_coord:
            if raw.dtype.kind not in "iuf" or raw.dtype.kind == "b":
                raise ValueError(
                    f"Physical coordinates for dimension {dim!r} must be numeric"
                )
            numeric = np.asarray(raw, dtype=np.float64)
            if not np.isfinite(numeric).all():
                raise ValueError(
                    f"Physical coordinates for dimension {dim!r} must be finite"
                )
            differences = np.diff(numeric)
            if not (np.all(differences > 0) or np.all(differences < 0)):
                raise ValueError(
                    f"Physical coordinates for dimension {dim!r} must be strictly monotonic"
                )
            coordinates[dim] = (numeric, coord.units)
        else:
            coordinates[dim] = (raw, coord.units)
    return coordinates


def _normalize_2d_distance(distance, resolved_dims, coordinates, *, use_coord):
    """Normalize 2D distance constraints while preserving requested values."""
    if distance is None:
        return {"mode": "none", "requested": None, "effective": None}, None

    dim_names = [dim for _, dim in resolved_dims]
    if isinstance(distance, Mapping):
        if set(distance) != set(dim_names) or any(
            not isinstance(key, str) for key in distance
        ):
            raise ValueError("distance mapping keys must exactly match dims names")
        requested_values = [distance[dim] for dim in dim_names]
    elif isinstance(distance, (tuple, list)) and len(distance) == 2:
        requested_values = list(distance)
    else:
        raise ValueError(
            "2D distance must be None, a two-value sequence, or a dims mapping"
        )

    def is_quantity(value):
        return isinstance(value, Quantity)

    modes = [is_quantity(value) for value in requested_values]
    if any(modes) and not all(modes):
        raise ValueError("2D distances cannot mix point values and Pint quantities")
    mode = "physical" if all(modes) else "points"
    if mode == "physical" and not use_coord:
        raise ValueError("Pint distances require use_coord=True")

    requested = {}
    effective = {}
    numeric_distances = []
    for dim, value in zip(dim_names, requested_values, strict=True):
        if mode == "physical":
            coords = coordinates.get(dim)
            if coords is None or coords[1] is None:
                raise ValueError(
                    f"Physical distance for dimension {dim!r} requires coordinate units"
                )
            coord_values, coord_units = coords
            try:
                magnitude = float(value.to(coord_units).magnitude)
            except (DimensionalityError, TypeError, ValueError) as exc:
                raise ValueError(
                    _format_peakfinding_units_error(
                        "distance", value.units, coord_units, dim
                    )
                ) from exc
            requested[dim] = {
                "value": float(value.magnitude),
                "units": str(value.units),
            }
            effective[dim] = {"value": magnitude, "units": str(coord_units)}
        else:
            if not isinstance(
                value, (int, float, np.integer, np.floating)
            ) or isinstance(value, (bool, np.bool_)):
                raise ValueError("Point distances must be real numeric scalars")
            magnitude = float(value)
            requested[dim] = {"value": magnitude, "units": "point"}
            effective[dim] = {"value": magnitude, "units": "point"}

        if not np.isfinite(magnitude) or magnitude < 0:
            raise ValueError("2D distances must be finite and non-negative")
        numeric_distances.append(magnitude)

    history = {"mode": mode, "requested": requested, "effective": effective}
    return history, tuple(numeric_distances)


def _normalize_2d_height(height, dataset):
    """Validate and convert a scalar 2D positive-height threshold."""
    if height is None:
        return None, {"mode": "positive"}
    if isinstance(height, Quantity):
        if dataset.units is None:
            raise ValueError("A unit-bearing height requires dataset units")
        try:
            value = float(height.to(dataset.units).magnitude)
        except (DimensionalityError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Height units {height.units} are incompatible with dataset units {dataset.units}"
            ) from exc
        units = str(dataset.units)
        requested = {"value": float(height.magnitude), "units": str(height.units)}
    else:
        if not isinstance(height, (int, float, np.integer, np.floating)) or isinstance(
            height, (bool, np.bool_)
        ):
            raise ValueError(
                "2D height must be a positive real scalar or Pint quantity"
            )
        value = float(height)
        units = str(dataset.units) if dataset.units is not None else "dimensionless"
        requested = {"value": value, "units": units}

    if not np.isfinite(value) or value <= 0:
        raise ValueError("2D height must be finite and strictly positive")
    return value, {"value": value, "units": units, "requested": requested}


def _peak_position(value, units, *, use_coord):
    if not use_coord:
        return value
    if units is None:
        return value
    return value * units


def _history_for_2d_peaks(dataset, parameters, count):
    """Build detached dataset-compatible structured and rendered history."""
    result_dataset = dataset.copy(deep=False)
    result_dataset._append_history_entry(
        operation="find_peaks_2d",
        parameters=parameters,
        message=f"Found {count} peak(s) with find_peaks (2D)",
    )
    return result_dataset.history_entries, tuple(result_dataset.history)


def _find_peaks_2d(
    dataset,
    *,
    height,
    window_length,
    threshold,
    distance,
    prominence,
    width,
    wlen,
    rel_height,
    plateau_size,
    use_coord,
    as_result,
    dims,
):
    """Find deterministic positive maxima in an explicitly selected 2D spectrum."""
    if dataset.ndim != 2:
        raise ValueError("Explicit 2D peak finding requires a 2D dataset")
    resolved_dims = _resolve_peakfinding_dims(dataset, dims)
    dim_names = tuple(dim for _, dim in resolved_dims)

    unsupported = {
        "threshold": threshold,
        "prominence": prominence,
        "width": width,
        "wlen": wlen,
        "plateau_size": plateau_size,
    }
    provided = [name for name, value in unsupported.items() if value is not None]
    if provided:
        raise NotImplementedError(
            "2D peak finding does not yet support: " + ", ".join(provided)
        )
    if rel_height != 0.5:
        raise NotImplementedError("2D peak finding does not support rel_height")
    if window_length != 3:
        raise NotImplementedError("2D peak finding does not interpolate peak positions")
    if not isinstance(use_coord, (bool, np.bool_)):
        raise ValueError("use_coord must be a boolean")

    values, valid = _peakfinding_working_data(dataset)
    coordinates = _peakfinding_coordinates(
        dataset, resolved_dims, use_coord=bool(use_coord)
    )
    height_value, height_history = _normalize_2d_height(height, dataset)
    distance_history, distances = _normalize_2d_distance(
        distance, resolved_dims, coordinates, use_coord=bool(use_coord)
    )

    axes = tuple(axis for axis, _ in resolved_dims)
    if axes != (0, 1):
        working_values = np.transpose(values, axes)
        working_valid = np.transpose(valid, axes)
    else:
        working_values = values
        working_valid = valid

    filtered = ndimage.maximum_filter(
        working_values,
        size=(3, 3),
        mode="constant",
        cval=-np.inf,
    )
    if height_value is None:
        candidate_mask = (
            working_valid & (working_values > 0) & (working_values == filtered)
        )
    else:
        candidate_mask = (
            working_valid
            & (working_values >= height_value)
            & (working_values == filtered)
        )

    labels, n_labels = ndimage.label(
        candidate_mask,
        structure=np.ones((3, 3), dtype=bool),
    )
    candidates = []
    for label_id in range(1, n_labels + 1):
        cells = np.argwhere(labels == label_id)
        if not cells.size:
            continue
        representative = min(tuple(int(v) for v in cell) for cell in cells)
        candidates.append((representative, float(working_values[representative])))

    if distances is not None:
        constrained_axes = [value > 0 for value in distances]

        def too_close(left, right):
            if not any(constrained_axes):
                return False
            for axis, distance_value in enumerate(distances):
                if distance_value <= 0:
                    continue
                if use_coord and distance_history["mode"] == "physical":
                    dim = dim_names[axis]
                    coord_values, _coord_units = coordinates[dim]
                    delta = abs(
                        float(coord_values[left[0][axis]])
                        - float(coord_values[right[0][axis]])
                    )
                else:
                    delta = abs(left[0][axis] - right[0][axis])
                if delta >= distance_value:
                    return False
            return True

        ranked = sorted(candidates, key=lambda item: (-item[1], item[0]))
        retained = []
        for candidate in ranked:
            if not any(too_close(candidate, prior) for prior in retained):
                retained.append(candidate)
        candidates = retained

    candidates.sort(key=lambda item: item[0])
    columns = ["index"]
    columns.extend(f"{dim}_index" for dim in dim_names)
    columns.extend(f"{dim}_position" for dim in dim_names)
    columns.append("height")

    rows = []
    for ordinal, (indices, magnitude) in enumerate(candidates):
        source_indices = {dim: int(indices[pos]) for pos, dim in enumerate(dim_names)}
        source_positions = {}
        for pos, dim in enumerate(dim_names):
            if use_coord:
                coordinate_values, coordinate_units = coordinates[dim]
                coordinate = _peak_position(
                    float(coordinate_values[indices[pos]]),
                    coordinate_units,
                    use_coord=True,
                )
            else:
                coordinate = int(indices[pos])
            source_positions[dim] = coordinate
        row = {"index": ordinal}
        row.update({f"{dim}_index": source_indices[dim] for dim in dim_names})
        row.update({f"{dim}_position": source_positions[dim] for dim in dim_names})
        row["height"] = (
            magnitude * dataset.units if dataset.units is not None else magnitude
        )
        rows.append(row)

    properties = {
        "peak_heights": [
            row["height"].magnitude
            if hasattr(row["height"], "magnitude")
            else row["height"]
            for row in rows
        ]
    }
    if not candidates:
        properties = {}

    effective_distance = distance_history
    parameters = {
        "dims": list(dim_names),
        "axes": list(axes),
        "algorithm": "maximum_filter",
        "height": height_history,
        "distance": effective_distance,
        "neighborhood": {"shape": [3, 3], "connectivity": 8},
        "plateau": {"connectivity": 8, "representative": "lexicographic_min"},
        "invalid_policy": "exclude_masked_nonfinite",
        "tie_break": "intensity_desc_then_lexicographic",
        "interpolation": "none",
        "n_peaks": len(candidates),
    }
    history_entries, history = _history_for_2d_peaks(
        dataset, parameters, len(candidates)
    )
    table = PeakTable(
        None,
        properties,
        rows=rows,
        columns=columns,
        history_entries=history_entries,
        history=history,
    )
    if as_result:
        peak_indices = np.asarray(
            [[row[f"{dim}_index"] for dim in dim_names] for row in rows],
            dtype=int,
        ).reshape((-1, 2))
        peak_positions = [
            tuple(row[f"{dim}_position"] for dim in dim_names) for row in rows
        ]
        peak_heights = [row["height"] for row in rows]
        return PeakFindingResult(
            table,
            properties,
            sample_index=peak_indices,
            sample_position=peak_positions,
            sample_height=peak_heights,
            analysis_dims=dim_names,
            parameters=parameters,
        )
    return table, properties


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
    *,
    dims=None,
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
        Input dataset containing spectral data. The historical path accepts
        1D data (including squeezed singleton data). A true 2D spectrum must
        explicitly provide both dimensions through ``dims``.
    height : float or array-like, optional
        In 1D, minimum and/or maximum peak height criteria. Can be specified as:
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
        Minimum separation between peaks. In 1D, this follows SciPy peak
        spacing behavior. In 2D, use ``dims`` and provide a pair of distances
        or a mapping from each selected dimension name to a point count or
        physical distance. 2D suppression uses the documented rectangular
        anisotropic metric.
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
    dims : sequence of str or int, optional
        Explicit dimensions to analyze. ``None`` preserves historical dispatch.
        One dimension selects the existing 1D engine; exactly two dimensions
        select deterministic positive-maximum detection for a true 2D spectrum.

    Returns
    -------
    peaks : NDDataset, PeakTable or None
        Dataset containing identified 1D peaks with interpolated positions, or
        a :class:`PeakTable` containing discrete 2D peaks. Returned when
        ``as_result=False``.
    properties : dict or None
        In 1D, SciPy peak properties. In 2D, ``peak_heights`` only. Values use
        appropriate units when applicable. Returned when ``as_result=False``.
    result : PeakFindingResult
        Structured peak finding result. For 1D, ``peaks`` is an NDDataset; for
        2D, ``peaks`` and ``table`` are the same :class:`PeakTable`. Returned
        when ``as_result=True``.
        Result rows include ``sample_index``, ``sample_position`` and
        ``sample_height`` for the discrete detector sample in addition to the
        refined ``position`` and ``height``.

    Notes
    -----
    - Peak positions are refined using quadratic interpolation when window_length > 1
    - The function handles units automatically when use_coord=True
    - For noisy data, consider preprocessing with smoothing functions
    - In :class:`PeakFindingResult`, ``index`` is the row identifier,
      ``sample_index`` is the local index returned by the detector for the
      1D signal actually analyzed, ``sample_position`` and ``sample_height``
      are the coordinate and signal value at that sample, and ``position`` and
      ``height`` are the existing refined values. No global index into a parent
      dataset is reported for sliced inputs.
    - The 2D path is selected only with ``dims=(dim0, dim1)`` and returns rows
      with native source indices and one position per selected dimension.
    - The 2D path supports positive maxima in real numeric data only. It does
      not calculate interpolated positions, widths, prominence, volumes, or
      integrals. A true 2D dataset without explicit ``dims`` retains the
      historical dimensionality error.

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

    Explicit 2D detection with physical, anisotropic coordinate distances:

    >>> y = scp.Coord.linspace(-1.0, 1.0, 41, title="y", units="ppm")
    >>> x = scp.Coord.linspace(0.0, 2.0, 81, title="x", units="ppm")
    >>> yy, xx = np.meshgrid(y.data, x.data, indexing="ij")
    >>> z = np.exp(-((yy + 0.4) ** 2 + (xx - 0.6) ** 2) / 0.02)
    >>> z += 0.8 * np.exp(-((yy - 0.45) ** 2 + (xx - 1.5) ** 2) / 0.03)
    >>> map2d = scp.NDDataset(z, coordset=[y, x], units="absorbance")
    >>> result2d = map2d.find_peaks(
    ...     dims=("y", "x"),
    ...     height=0.2,
    ...     distance={"y": 0.2 * scp.ur.ppm, "x": 0.25 * scp.ur.ppm},
    ...     as_result=True,
    ... )
    >>> result2d.table.columns
    ('index', 'y_index', 'x_index', 'y_position', 'x_position', 'height')

    """
    if dims is not None:
        if dataset.ndim == 1:
            _resolve_peakfinding_dims(dataset, dims)
        elif dataset.ndim == 2:
            return _find_peaks_2d(
                dataset,
                height=height,
                window_length=window_length,
                threshold=threshold,
                distance=distance,
                prominence=prominence,
                width=width,
                wlen=wlen,
                rel_height=rel_height,
                plateau_size=plateau_size,
                use_coord=use_coord,
                as_result=as_result,
                dims=dims,
            )
        else:
            raise ValueError(
                "Explicit peak finding supports only one or two dimensions"
            )

    # get the dataset; keep the original no-dims path unchanged.
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
    sample_index = peaks.astype(int, copy=True)
    if use_coord:
        sample_position = _copy_values(lastcoord.values[sample_index])
    else:
        sample_position = sample_index.copy()
    sample_height = _copy_values(X.data[sample_index])
    if X.units is not None:
        sample_height = sample_height * X.units

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
    out.history = f"Found {len(peaks)} peak(s) with find_peaks"

    if as_result:
        return PeakFindingResult(
            out,
            properties,
            sample_index=sample_index,
            sample_position=sample_position,
            sample_height=sample_height,
        )

    return out, properties
