"""Bounded fits of separable peaks on a native spectral grid."""

from collections.abc import Mapping
from collections.abc import Sequence

import numpy as np
from scipy.optimize import least_squares

from spectrochempy.analysis.curvefitting.simulation import _MODELS
from spectrochempy.analysis.curvefitting.simulation import _evaluate_separable_peak
from spectrochempy.analysis.curvefitting.simulation import _interval_unit
from spectrochempy.analysis.curvefitting.simulation import _parameter_value
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.coordset import CoordSet
from spectrochempy.core.dataset.nddataset import NDDataset


def _grid(dataset):
    if not isinstance(dataset, NDDataset) or dataset.ndim < 2 or not dataset.size:
        raise ValueError(
            "ND fitting requires a nonempty NDDataset with at least two axes"
        )
    if dataset.data.dtype.kind not in "iuf":
        raise ValueError("ND fitting requires real intensities")

    coordinates, units = {}, {}
    for dim, length in zip(dataset.dims, dataset.shape, strict=True):
        coord = getattr(dataset, dim, None)
        if isinstance(coord, CoordSet):
            coord = coord.default
        if not isinstance(coord, Coord) or not coord.has_data:
            raise ValueError(f"Axis {dim!r} requires coordinates")
        grid = np.asarray(coord._data)
        if (
            grid.shape != (length,)
            or length < 2
            or grid.dtype.kind not in "iuf"
            or not np.isfinite(grid).all()
            or not (np.all(grid[1:] > grid[:-1]) or np.all(grid[1:] < grid[:-1]))
        ):
            raise ValueError(
                f"Axis {dim!r} requires finite, strictly monotone coordinates"
            )
        coordinates[dim] = grid
        units[dim] = coord.units
    return coordinates, units


def _parameter(spec, name, unit, *, width=False):
    if not isinstance(spec, Mapping):
        spec = {"value": spec}
    if "value" not in spec or spec.keys() - {"value", "bounds", "fixed"}:
        raise ValueError(f"Invalid parameter specification for {name}")

    fixed = spec.get("fixed", False)
    if not isinstance(fixed, (bool, np.bool_)):
        raise ValueError(f"{name}: fixed must be boolean")
    bounds = spec.get("bounds", (None, None))
    if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
        raise ValueError(f"{name}: bounds must contain two endpoints")

    def magnitude(value):
        converted = float(_parameter_value(value, unit, name, interval=width))
        if not np.isfinite(converted):
            raise ValueError(f"{name} is outside the solver's floating-point range")
        return converted

    low = -np.inf if bounds[0] is None else magnitude(bounds[0])
    high = np.inf if bounds[1] is None else magnitude(bounds[1])
    value = magnitude(spec["value"])
    if low >= high or not low <= value <= high:
        raise ValueError(f"{name}: initial value must lie within ordered bounds")
    if width and (value <= 0 or (not fixed and low <= 0)):
        raise ValueError(
            f"{name}: width and any free width lower bound must be positive"
        )
    return value, low, high, fixed


def _prepare_peaks(specs, intensity_unit, units):
    if not isinstance(specs, Sequence) or isinstance(specs, (str, bytes)) or not specs:
        raise ValueError("peaks must be a nonempty ordered sequence")

    peaks, free, initial, lower, upper, names, parameter_units = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    def register(target, key, spec, name, unit):
        width = key == "width"
        value, low, high, fixed = _parameter(spec, name, unit, width=width)
        target[key] = value
        if not fixed:
            free.append((target, key))
            initial.append(value)
            lower.append(low)
            upper.append(high)
            names.append(name)
            parameter_units.append(
                _interval_unit(unit) if width and unit is not None else unit
            )

    for index, spec in enumerate(specs):
        if not isinstance(spec, Mapping) or set(spec) != {"height", "axes"}:
            raise ValueError("Each peak requires height and axes")
        axes = spec["axes"]
        if not isinstance(axes, Mapping) or set(axes) != set(units):
            raise ValueError("Each peak must specify every dataset dimension")
        peak = {"axes": {}}
        register(
            peak, "height", spec["height"], f"peaks[{index}].height", intensity_unit
        )
        for dim in sorted(units):
            axis = axes[dim]
            if not isinstance(axis, Mapping) or set(axis) != {
                "model",
                "center",
                "width",
            }:
                raise ValueError(f"Axis {dim!r} requires model, center and width")
            model = axis["model"]
            if not isinstance(model, str) or model not in _MODELS:
                raise ValueError(f"Unsupported model on axis {dim!r}: {model!r}")
            resolved = {"model": model}
            for key in ("center", "width"):
                register(
                    resolved,
                    key,
                    axis[key],
                    f"peaks[{index}].axes[{dim!r}].{key}",
                    units[dim],
                )
            peak["axes"][dim] = resolved
        peaks.append(peak)
    return peaks, free, initial, (lower, upper), tuple(names), tuple(parameter_units)


def fit_separable_peaks(dataset, specs, max_calls):
    """Fit and reconstruct on the input grid with a fixed observation support."""
    coordinates, units = _grid(dataset)
    peaks, free, initial, bounds, names, parameter_units = _prepare_peaks(
        specs, dataset.units, units
    )
    with np.errstate(over="ignore"):
        data = np.asarray(dataset.data, dtype=np.float64)
    unmasked = ~np.asarray(dataset.mask)
    if np.any(np.isfinite(dataset.data) & ~np.isfinite(data) & unmasked):
        raise ValueError(
            "Observed intensity is outside the solver's floating-point range"
        )
    valid = np.isfinite(data) & unmasked
    count = np.count_nonzero(valid)
    if count == 0 or count < len(free):
        raise ValueError("ND fitting requires observations for all free parameters")
    observed = data[valid]

    def model(values, *, keep_components=False):
        for (target, key), value in zip(free, values, strict=True):
            target[key] = float(value)
        total = np.zeros(dataset.shape)
        components = []
        with np.errstate(over="ignore", invalid="ignore"):
            for peak in peaks:
                component = _evaluate_separable_peak(dataset.shape, coordinates, peak)
                total += component
                if keep_components:
                    components.append(component)
        return total, components

    def residuals(values):
        fitted, _ = model(values)
        with np.errstate(over="ignore", invalid="ignore"):
            residual = observed - fitted[valid]
        if not np.isfinite(residual).all():
            raise ValueError("ND fitting produced non-finite residuals on the support")
        return residual

    solution = None
    if free:
        solution = least_squares(
            residuals,
            initial,
            bounds=bounds,
            method="trf",
            jac="3-point",
            x_scale="jac",
            max_nfev=max_calls,
        )
    values = np.asarray(solution.x if solution is not None else initial)
    residual = np.zeros(dataset.shape)
    residual[valid] = residuals(values)
    fitted, components = model(values, keep_components=True)

    def output(array, title):
        result = NDDataset(
            array,
            dims=list(dataset.dims),
            coordset=dataset.coordset.copy(),
            mask=~valid,
            units=dataset.units,
            title=title,
            history=dataset.history_entries,
        )
        result._append_history_entry(
            operation="optimize",
            parameters={
                "mode": "separable_peaks",
                "dims": list(dataset.dims),
                "method": "trf",
                "n_observations": int(count),
                "n_varying_parameters": len(free),
                "parameter_order": list(names),
                "peaks": peaks,
            },
            message=title,
        )
        return result

    return (
        output(data, dataset.title),
        output(fitted, "fitted peak intensity"),
        tuple(
            output(part, f"fitted peak {index}")
            for index, part in enumerate(components)
        ),
        output(residual, "fit residuals"),
        peaks,
        values,
        names,
        parameter_units,
        solution,
    )


def physical_covariance(solution, diagnostics):
    """Return local covariance only when unconstrained inference is meaningful."""
    if solution is None:
        return None, "no_free_parameters"
    if not solution.success:
        return None, "not_converged"
    if diagnostics["degrees_of_freedom"] <= 0:
        return None, "insufficient_degrees_of_freedom"
    if np.any(solution.active_mask):
        return None, "active_bounds"

    jacobian = solution.jac
    variance = diagnostics["reduced_chi_square"]
    if not np.isfinite(jacobian).all() or not np.isfinite(variance):
        return None, "numeric_range"
    column_scale = np.max(np.abs(jacobian), axis=0)
    if np.any(column_scale == 0):
        return None, "rank_deficient"
    try:
        _, singular, vectors = np.linalg.svd(
            jacobian / column_scale, full_matrices=False
        )
    except np.linalg.LinAlgError:
        return None, "invalid_jacobian"
    threshold = singular[0] * np.sqrt(np.finfo(jacobian.dtype).eps)
    if singular[-1] <= threshold:
        return None, "rank_deficient"
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        scaled = (np.sqrt(variance) / singular)[:, None] * (vectors / column_scale)
        covariance = scaled.T @ scaled
    if not np.isfinite(covariance).all():
        return None, "numeric_range"
    covariance.flags.writeable = False
    return covariance, None
