# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""Integration of physical boxes on sampled spectral grids."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field

import numpy as np
from pint import Quantity as PintQuantity
from pint.errors import PintError

from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.coordset import CoordSet
from spectrochempy.core.dataset.nddataset import _history_entry
from spectrochempy.core.units import Quantity
from spectrochempy.core.units._conversion import _import_quantity

__all__ = ["RegionIntegrationResult", "integrate_region"]
__dataset_methods__ = ["integrate_region"]

_METHOD = "tensor_trapezoid"


@dataclass(frozen=True)
class RegionIntegrationResult:
    """One physical integral with standard uncertainty and detached provenance."""

    value: float | None
    unit: object
    uncertainty: float | None
    valid: bool
    reason: str | None
    uncertainty_status: str
    method: str = _METHOD
    _history: tuple = field(default=(), repr=False, compare=False)

    @property
    def history_entries(self):
        """Return detached structured history entries."""
        return deepcopy(list(self._history))


def _check_linear_unit(unit, quantity_type=Quantity):
    """Reject units that do not represent differences by a constant scale."""
    if unit is not None and any(
        not quantity_type._REGISTRY._units[name].converter.is_multiplicative
        for name, _ in quantity_type(1, unit).to_tuple()[1]
    ):
        raise ValueError("Region integration requires multiplicative units")


def _magnitude(value, unit, name):
    """Resolve a real scalar and retain its requested magnitude and unit."""
    quantity = value if isinstance(value, PintQuantity) else None
    scalar = quantity.magnitude if quantity is not None else value
    if isinstance(scalar, (bool, np.bool_, np.timedelta64)) or not isinstance(
        scalar, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite real scalar")
    requested = {
        "value": _record_number(scalar),
        "unit": str(quantity.units if quantity is not None else unit)
        if quantity is not None or unit is not None
        else None,
    }
    try:
        if quantity is not None:
            if unit is None:
                raise ValueError(f"{name} requires a unit-bearing axis or intensity")
            _check_linear_unit(quantity.units, type(quantity))
            if not isinstance(quantity, Quantity):
                quantity = _import_quantity(quantity, unit)
            scalar = quantity.to(unit).magnitude
        if not np.isfinite(np.longdouble(scalar)):
            raise ValueError(f"{name} must be a finite real scalar")
    except (PintError, OverflowError) as exc:
        raise ValueError(f"Invalid {name}: {exc}") from exc
    return scalar, requested


def _record_number(value):
    if isinstance(value, (int, np.integer)):
        return int(value)
    return str(value) if isinstance(value, np.longdouble) else float(value)


def _relative(value, origin):
    if isinstance(value, (int, np.integer)):
        return np.longdouble(int(value) - origin)
    rounded_origin = np.longdouble(origin)
    return (np.longdouble(value) - rounded_origin) - (origin - int(rounded_origin))


def _resolve_bounds(bounds, axes, label):
    if not isinstance(bounds, Mapping) or set(bounds) != set(axes):
        raise ValueError(f"{label} must specify exactly the native dimensions")
    effective, history = {}, {}
    for dim, (grid, origin, coord) in axes.items():
        pair = bounds[dim]
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"{label}[{dim!r}] must be a pair")
        converted = [_magnitude(v, coord.units, label) for v in pair]
        low, high = (
            v if origin is None else _relative(v, origin) for v, _ in converted
        )
        if not min(grid[0], grid[-1]) <= low < high <= max(grid[0], grid[-1]):
            raise ValueError(f"{label}[{dim!r}] must be ordered and inside the grid")
        effective[dim] = (low, high)
        history[dim] = {
            "requested": [requested for _, requested in converted],
            "effective": [_record_number(value) for value, _ in converted],
            "unit": str(coord.units) if coord.units is not None else None,
        }
    return effective, history


def _axis_weights(grid, low, high):
    """Integrate the two linear basis functions on each clipped interval."""
    reverse = grid[0] > grid[-1]
    grid = grid[::-1] if reverse else grid
    left, right = grid[:-1], grid[1:]
    start, stop = np.maximum(left, low), np.minimum(right, high)
    intersects = start < stop
    left, right, start, stop = (
        array[intersects] for array in (left, right, start, stop)
    )
    span = right - left
    half = (stop - start) / 2
    weights = np.zeros(grid.size, dtype=np.longdouble)
    indices = np.flatnonzero(intersects)
    weights[indices] = half * ((right - start) / span + (right - stop) / span)
    weights[indices + 1] += half * ((start - left) / span + (stop - left) / span)
    supported = np.zeros(grid.size, dtype=bool)
    supported[indices] = True
    supported[indices + 1] = True
    if reverse:
        weights, supported = weights[::-1], supported[::-1]
    indices = np.flatnonzero(supported)
    return weights, indices[::-1] if reverse else indices


def _background_std(samples):
    """Preserve small variations around a large background level."""
    if samples.dtype.kind in "iu":
        shifted = (samples.astype(object) - int(samples[0])).astype(np.longdouble)
    else:
        samples = samples.astype(np.longdouble)
        shifted = samples - samples[0]
    scale_factor = 1
    if not np.isfinite(shifted).all():
        shifted = samples * 0.5 - samples[0] * 0.5
        scale_factor = 2
    scale = np.max(np.abs(shifted))
    return np.std(shifted / scale, ddof=1) * scale * scale_factor if scale else 0


def integrate_region(dataset, bounds, *, noise_std=None, noise_bounds=None):
    """
    Integrate the multilinear interpolant over one physical ND box.

    Parameters
    ----------
    dataset : NDDataset
        Nonempty real spectrum with finite, strictly monotone coordinates.
    bounds : mapping
        Ordered lower and upper physical bounds for every native dimension.
        Numbers use axis units. Compatible Pint quantities are converted.
    noise_std : float or Quantity, optional
        Standard deviation per sample, in the dataset intensity units.
    noise_bounds : mapping, optional
        Disjoint background box for estimating the sample standard deviation.
        Mutually exclusive with noise_std.

    Returns
    -------
    RegionIntegrationResult
        Signed volume and optional propagated standard uncertainty. An invalid
        signal support gives no value. Insufficient background gives no
        uncertainty while preserving a valid integral.

    Notes
    -----
    Interpolated endpoints can use points outside the box. Masked or non-finite
    points with nonzero weight invalidate the integral. Uncertainty assumes
    additive, uncorrelated noise with constant variance and uses the same
    weights as the integral.

    Examples
    --------
    >>> import spectrochempy as scp
    >>> spectrum = scp.NDDataset([0, 1, 3], coordset=[scp.Coord([0, 1, 3])])
    >>> spectrum.integrate_region({"x": (0.5, 2)}).value
    1.875
    """
    if noise_std is not None and noise_bounds is not None:
        raise ValueError("Specify at most one of noise_std and noise_bounds")

    data = np.asarray(dataset.data)
    if not dataset.ndim or not data.size or data.dtype.kind not in "iuf":
        raise ValueError("Region integration requires a nonempty real dataset")

    unit = dataset.units
    _check_linear_unit(unit)
    axes = {}
    for axis, dim in enumerate(dataset.dims):
        coord = getattr(dataset, dim, None)
        if isinstance(coord, CoordSet):
            coord = coord.default
        if not isinstance(coord, Coord) or not coord.has_data:
            raise ValueError(f"Axis {dim!r} requires coordinates")
        grid = np.asarray(coord._data)
        if (
            grid.shape != (data.shape[axis],)
            or grid.size < 2
            or grid.dtype.kind not in "iuf"
            or not np.isfinite(grid).all()
            or not (np.all(grid[1:] > grid[:-1]) or np.all(grid[1:] < grid[:-1]))
        ):
            raise ValueError(
                f"Axis {dim!r} requires finite, strictly monotone coordinates"
            )
        _check_linear_unit(coord.units)
        if coord.units is not None:
            unit = coord.units if unit is None else unit * coord.units
        origin = min(int(grid[0]), int(grid[-1])) if grid.dtype.kind in "iu" else None
        grid = (
            grid.astype(np.longdouble)
            if origin is None
            else (grid.astype(object) - origin).astype(np.longdouble)
        )
        axes[dim] = (grid, origin, coord)

    contraction_order = tuple(sorted(range(len(axes)), key=list(axes).__getitem__))
    limits, bounds_record = _resolve_bounds(bounds, axes, "bounds")
    sigma, noise_record = None, {}
    noise_status = "not_requested"
    if noise_std is not None:
        sigma, requested = _magnitude(noise_std, dataset.units, "noise_std")
        if sigma < 0:
            raise ValueError("noise_std must be nonnegative")
        noise_record = {"model": "uncorrelated", "requested": requested}
        noise_status = "available"

    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        quadrature = [
            _axis_weights(grid, *limits[dim]) for dim, (grid, _, _) in axes.items()
        ]
        positive = [indices for _, indices in quadrature]
        vectors = [weights[indices] for weights, indices in quadrature]
        numeric_support = all(
            vector.size and np.isfinite(vector).all() and np.all(vector > 0)
            for vector in vectors
        )
        support = np.ix_(*positive)
        mask = np.broadcast_to(dataset.mask, data.shape)
        signal = data[support]
        invalid = int(np.count_nonzero(mask[support] | ~np.isfinite(signal)))

        if noise_bounds is not None:
            background_limits, record = _resolve_bounds(
                noise_bounds, axes, "noise_bounds"
            )
            background = [
                np.flatnonzero(
                    (grid >= background_limits[dim][0])
                    & (grid <= background_limits[dim][1])
                )
                for dim, (grid, _, _) in axes.items()
            ]
            if all(
                np.intersect1d(a, b, assume_unique=True).size
                for a, b in zip(positive, background, strict=True)
            ):
                raise ValueError("noise_bounds overlaps the signal quadrature support")
            background = [
                indices[::-1] if grid[0] > grid[-1] else indices
                for indices, (grid, _, _) in zip(background, axes.values(), strict=True)
            ]
            selection = np.ix_(*background)
            samples = data[selection].transpose(contraction_order)
            valid_samples = ~mask[selection].transpose(contraction_order) & np.isfinite(
                samples
            )
            samples = samples[valid_samples]
            noise_record = {
                "model": "uncorrelated",
                "bounds": record,
                "valid_samples": int(samples.size),
                "ddof": 1,
            }
            noise_status = "insufficient_background"
            if samples.size >= 2:
                sigma = _background_std(samples)
                noise_status = "available"

        value, uncertainty, reason = None, None, None
        if invalid:
            reason = "incomplete_support"
        elif not numeric_support:
            reason = "numeric_range"
        else:
            integral = signal.transpose(contraction_order).astype(np.longdouble)
            for axis in contraction_order:
                vector = vectors[axis]
                integral = vector @ integral.reshape(vector.size, -1)
            value = float(integral.item())
            if not np.isfinite(value):
                value, reason = None, "numeric_range"
            elif sigma is not None:
                propagated = np.longdouble(sigma)
                for axis in contraction_order:
                    propagated *= np.hypot.reduce(vectors[axis])
                uncertainty = float(propagated)
                if not np.isfinite(uncertainty):
                    uncertainty, noise_status = None, "numeric_range"

    if reason is not None and noise_status != "not_requested":
        noise_status = reason
    if sigma is not None and np.isfinite(np.longdouble(sigma)):
        noise_record["sigma"] = _record_number(sigma)
        noise_record["unit"] = str(dataset.units) if dataset.units is not None else None
    entry = _history_entry(
        operation="integrate_region",
        parameters={
            "method": _METHOD,
            "dims": list(axes),
            "coordinates": {dim: coord.name for dim, (_, _, coord) in axes.items()},
            "bounds": bounds_record,
            "contributing_samples": int(signal.size),
            "invalid_samples": invalid,
            "unit": str(unit) if unit is not None else None,
            "noise": noise_record,
            "uncertainty_status": noise_status,
            "reason": reason,
        },
        message="Integrated a physical region"
        if reason is None
        else "Region integral unavailable",
    )
    return RegionIntegrationResult(
        value,
        unit,
        uncertainty,
        reason is None,
        reason,
        noise_status,
        _history=(*dataset.history_entries, entry),
    )
