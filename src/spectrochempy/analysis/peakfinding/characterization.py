"""Grouping and apparent widths of detected 2D peaks."""

from collections import Counter
from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from pint import Quantity as PintQuantity

from spectrochempy.analysis.curvefitting.simulation import _interval_unit
from spectrochempy.analysis.curvefitting.simulation import _parameter_value
from spectrochempy.analysis.peakfinding.peakfinding import PeakFindingResult
from spectrochempy.analysis.peakfinding.peakfinding import PeakTable
from spectrochempy.analysis.peakfinding.peakfinding import _peakfinding_coordinates
from spectrochempy.analysis.segmentation import _threshold_record
from spectrochempy.analysis.segmentation import segment_regions
from spectrochempy.core.dataset.nddataset import NDDataset

__all__ = ["PeakCharacterizationResult", "characterize_peaks_2d"]
__dataset_methods__ = ["characterize_peaks_2d"]


@dataclass(frozen=True)
class PeakCharacterizationResult:
    """Per-peak measurements, group membership and structured provenance."""

    rows: tuple
    groups: tuple
    _history: tuple

    @property
    def history_entries(self):
        """Return detached structured history entries."""
        return deepcopy(list(self._history))


def _fraction(value):
    if isinstance(value, (int, np.integer)):
        return Fraction(int(value))
    return Fraction(*value.as_integer_ratio())


def _logarithmic_unit(unit):
    return any(
        unit._REGISTRY._units[name].converter.is_logarithmic for name in unit._units
    )


def _radii(radii, coordinates):
    if not isinstance(radii, Mapping) or set(radii) != set(coordinates):
        raise ValueError("radii must contain exactly the two dimension names")

    limits, records = {}, {}
    for dim, (_, unit) in coordinates.items():
        supplied = radii[dim]
        name = f"radius for {dim!r}"
        if isinstance(supplied, PintQuantity) and (
            unit is None or supplied.dimensionality != unit.dimensionality
        ):
            raise ValueError(f"{name} must have the same dimensionality as the axis")
        supplied_unit = supplied.units if isinstance(supplied, PintQuantity) else unit
        if any(
            candidate is not None and _logarithmic_unit(candidate)
            for candidate in (unit, supplied_unit)
        ):
            raise ValueError(f"{name} requires non-logarithmic units")
        value = _parameter_value(supplied, unit, name, interval=True)
        if value <= 0:
            raise ValueError(f"{name} must be strictly positive")
        limits[dim] = _fraction(value)
        requested = (
            supplied.magnitude if isinstance(supplied, PintQuantity) else supplied
        )
        records[dim] = {
            "requested": _threshold_record(requested, supplied_unit),
            "effective": _threshold_record(
                value, _interval_unit(unit) if unit is not None else None
            ),
        }
    return limits, records


def _validate_peaks(dataset, peaks, coordinates, data, invalid):
    if (
        not isinstance(peaks, PeakFindingResult)
        or not isinstance(peaks.peaks, PeakTable)
        or peaks.analysis_dims is None
        or len(peaks.analysis_dims) != 2
        or set(peaks.analysis_dims) != set(dataset.dims)
        or not isinstance(peaks.parameters.get("use_coord"), bool)
    ):
        raise ValueError("peaks must be a 2D PeakFindingResult for these dimensions")

    source_heights = getattr(peaks, "_native_sample_heights", None)
    result = {}
    cells = set()
    for row in peaks.table.to_dict():
        index = row.get("index")
        cell = tuple(row.get(f"{dim}_index") for dim in dataset.dims)
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or index in result
        ):
            raise ValueError("Peak identifiers must be unique nonnegative integers")
        if any(
            not isinstance(i, int) or isinstance(i, bool) or not 0 <= i < size
            for i, size in zip(cell, dataset.shape, strict=True)
        ):
            raise ValueError(f"Peak {index} has an invalid sample index")
        if cell in cells or invalid[cell]:
            raise ValueError(f"Peak {index} is duplicated or invalid")

        height = _parameter_value(row.get("height"), dataset.units, "peak height")
        if height <= 0 or height != float(data[cell]):
            raise ValueError(f"Peak {index} height does not match the spectrum")
        if source_heights is not None:
            detected_cell = tuple(
                cell[dataset.dims.index(dim)] for dim in peaks.analysis_dims
            )
            source_height = source_heights.get(detected_cell)
            if source_height is None or _fraction(source_height) != _fraction(
                data[cell]
            ):
                raise ValueError(
                    f"Peak {index} native height does not match the spectrum"
                )
        if peaks.parameters["use_coord"]:
            for dim, i in zip(dataset.dims, cell, strict=True):
                grid, unit = coordinates[dim]
                position = _parameter_value(
                    row.get(f"{dim}_position"), unit, "peak position"
                )
                if _fraction(position) != _fraction(grid[i]):
                    raise ValueError(
                        f"Peak {index} position does not match dimension {dim!r}"
                    )

        result[index] = cell
        cells.add(cell)
    return result


def _groups(cells, region_ids, coordinates, limits):
    parents = {index: index for index in cells}
    regions = defaultdict(list)
    for index, cell in cells.items():
        position = tuple(
            _fraction(grid[i])
            for (grid, _), i in zip(coordinates.values(), cell, strict=True)
        )
        regions[region_ids[index]].append((position, index))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    radius_first, radius_second = limits.values()
    for points in regions.values():
        points.sort()
        for right, ((first, second), index) in enumerate(points):
            for left in range(right - 1, -1, -1):
                (other_first, other_second), other = points[left]
                if first - other_first > radius_first:
                    break
                if abs(second - other_second) <= radius_second:
                    a, b = root(index), root(other)
                    parents[max(a, b)] = min(a, b)

    members = defaultdict(list)
    for index in sorted(cells):
        members[root(index)].append(index)
    return tuple(
        {"group_id": group, "peak_indices": tuple(indices)}
        for group, indices in enumerate(members.values(), start=1)
    )


def _crossing(signal, invalid, grid, start, step, half):
    previous = _fraction(signal[start])
    stop = len(grid) if step > 0 else -1
    for i in range(start + step, stop, step):
        if invalid[i]:
            return None, "invalid_sample"
        value = _fraction(signal[i])
        if value > previous:
            return None, "rising_profile"
        if value <= half:
            near, far = _fraction(grid[i - step]), _fraction(grid[i])
            return near + (far - near) * (previous - half) / (previous - value), None
        previous = value
    return None, "boundary"


def _width(signal, invalid, grid, start, half):
    low, reason = _crossing(signal, invalid, grid, start, -1, half)
    if reason:
        return None, reason
    high, reason = _crossing(signal, invalid, grid, start, 1, half)
    if reason:
        return None, reason
    try:
        width = float(abs(high - low))
    except OverflowError:
        return None, "numeric_range"
    if width == 0:
        return None, "numeric_range"
    return width, "available"


def characterize_peaks_2d(dataset, peaks, *, threshold, radii, connectivity="face"):
    """
    Group detected peaks and measure apparent axial widths on a 2D spectrum.

    Parameters
    ----------
    dataset : NDDataset
        Real 2D spectrum with finite, strictly monotone numeric coordinates.
    peaks : PeakFindingResult
        Result from 2D peak detection on this spectrum with `as_result=True`.
        Use `distance=None` during detection to retain nearby maxima.
    threshold : float or Quantity
        Segmentation threshold, strictly below every supplied peak.
    radii : mapping
        Positive distances by dimension name. Numbers use each axis unit.
        Pint quantities must have the same dimensionality as the axis and
        are converted as intervals. Logarithmic units are unsupported.
    connectivity : {"face", "full"}, optional
        Connectivity for the thresholded regions.

    Returns
    -------
    PeakCharacterizationResult
        Peak rows keyed by the original peak index and groups numbered from one.
        Each row reports sampled height, widths and width status by dimension.

    Notes
    -----
    Links require a shared region and inclusive radii on both axes. Groups
    are the connected components of those links. For regions containing one
    supplied peak, widths use the first half-height crossing on each side
    of the sampled maximum, with a zero baseline. Shared regions and
    truncated or ambiguous cuts report an unavailable width. Widths outside
    the representable float64 range have status ``numeric_range``.

    Examples
    --------
    A ridge above the segmentation threshold connects two detected peaks.

    >>> import spectrochempy as scp
    >>> spectrum = scp.NDDataset(
    ...     [[0, 0, 0, 0, 0], [0, 10, 6, 8, 0], [0, 0, 0, 0, 0]],
    ...     coordset=[scp.Coord([0, 1, 2]), scp.Coord([0, 1, 2, 3, 4])],
    ... )
    >>> peaks = spectrum.find_peaks(dims=("y", "x"), height=5, as_result=True)
    >>> result = spectrum.characterize_peaks_2d(
    ...     peaks, threshold=4, radii={"y": 1, "x": 2}
    ... )
    >>> result.groups[0]["peak_indices"]
    (0, 1)
    >>> result.rows[0]["width_status"]["x"]
    'shared_region'
    """
    if not isinstance(dataset, NDDataset) or dataset.ndim != 2 or not dataset.size:
        raise ValueError("Peak characterization requires a nonempty 2D spectrum")

    coordinates = _peakfinding_coordinates(
        dataset, tuple(enumerate(dataset.dims)), use_coord=True
    )
    limits, records = _radii(radii, coordinates)
    labels, _ = segment_regions(
        dataset, threshold=threshold, dim=dataset.dims, connectivity=connectivity
    )
    data = np.asarray(dataset.data)
    invalid = np.asarray(labels.mask) | ~np.isfinite(data)
    cells = _validate_peaks(dataset, peaks, coordinates, data, invalid)
    region_ids = {index: int(labels.data[cell]) for index, cell in cells.items()}
    if 0 in region_ids.values():
        raise ValueError("The segmentation threshold must be below every supplied peak")

    groups = _groups(cells, region_ids, coordinates, limits)
    group_ids = {
        index: group["group_id"] for group in groups for index in group["peak_indices"]
    }
    counts = Counter(region_ids.values())
    rows = []
    for index in sorted(cells):
        cell = cells[index]
        height = data[cell].item()
        region = region_ids[index]
        widths, statuses = {}, {}
        for axis, (dim, (grid, unit)) in enumerate(coordinates.items()):
            if counts[region] > 1:
                widths[dim], statuses[dim] = None, "shared_region"
                continue
            cut = list(cell)
            cut[axis] = slice(None)
            cut = tuple(cut)
            width, statuses[dim] = _width(
                data[cut], invalid[cut], grid, cell[axis], _fraction(height) / 2
            )
            widths[dim] = (
                width * _interval_unit(unit)
                if width is not None and unit is not None
                else width
            )
        rows.append(
            {
                "peak_index": index,
                "group_id": group_ids[index],
                "region_label": region,
                "sample_height": height * dataset.units
                if dataset.units is not None
                else height,
                "widths": widths,
                "width_status": statuses,
            }
        )

    history = labels.history_entries
    labels._append_history_entry(
        operation="characterize_peaks_2d",
        parameters={
            "dims": list(dataset.dims),
            "peak_finding_dims": list(peaks.analysis_dims),
            "threshold": history[-1]["parameters"]["threshold"],
            "radii": records,
            "connectivity": connectivity,
            "n_peaks": len(rows),
            "n_groups": len(groups),
        },
        message="Characterized 2D peaks",
    )
    return PeakCharacterizationResult(
        tuple(rows), groups, tuple(labels.history_entries)
    )
