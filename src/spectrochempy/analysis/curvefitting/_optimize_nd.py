"""Bounded fits of separable peaks on a native spectral grid."""

from collections.abc import Mapping
from collections.abc import Sequence
from copy import deepcopy

import numpy as np
from scipy.optimize import least_squares

from spectrochempy.analysis.curvefitting.simulation import _MODELS
from spectrochempy.analysis.curvefitting.simulation import _evaluate_separable_peak
from spectrochempy.analysis.curvefitting.simulation import _history_magnitude
from spectrochempy.analysis.curvefitting.simulation import _interval_unit
from spectrochempy.analysis.curvefitting.simulation import _parameter_value
from spectrochempy.analysis.curvefitting.simulation import _scaled_offsets
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.coordset import CoordSet
from spectrochempy.core.dataset.nddataset import NDDataset

_BOUND_NEIGHBORHOOD = 1e-5


def snapshot_peaks(spec):
    """Copy the peak specification without requiring pickleable mappings."""
    if isinstance(spec, Mapping):
        return {key: snapshot_peaks(value) for key, value in spec.items()}
    if isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)):
        return tuple(snapshot_peaks(value) for value in spec)
    return spec


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
        return _parameter_value(value, unit, name, interval=width)

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

    peaks, free, limits, names, parameter_units = [], [], [], [], []

    def register(target, key, parameter, name, unit):
        value, low, high, fixed = parameter
        target[key] = value
        if not fixed:
            free.append((target, key))
            limits.append((low, high))
            names.append(name)
            parameter_units.append(
                _interval_unit(unit) if key == "width" and unit is not None else unit
            )

    for index, spec in enumerate(specs):
        if not isinstance(spec, Mapping) or set(spec) != {"height", "axes"}:
            raise ValueError("Each peak requires height and axes")
        axes = spec["axes"]
        if not isinstance(axes, Mapping) or set(axes) != set(units):
            raise ValueError("Each peak must specify every dataset dimension")
        peak = {"axes": {}}
        register(
            peak,
            "height",
            _parameter(spec["height"], "height", intensity_unit),
            f"peaks[{index}].height",
            intensity_unit,
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
                    _parameter(
                        axis[key], f"{dim} {key}", units[dim], width=key == "width"
                    ),
                    f"peaks[{index}].axes[{dim!r}].{key}",
                    units[dim],
                )
            peak["axes"][dim] = resolved
        peaks.append(peak)
    return peaks, free, limits, tuple(names), tuple(parameter_units)


def _relative_value(value, origin):
    if isinstance(value, int) and isinstance(origin, int):
        return value - origin
    return np.longdouble(value) - np.longdouble(origin)


def _physical_value(origin, offset):
    if isinstance(origin, int) and origin:
        whole = int(round(offset))
        center = origin + whole
        fraction = offset - whole
        if abs(fraction) <= abs(np.spacing(np.longdouble(center))) / 2:
            return center
        return np.longdouble(center) + fraction
    return np.longdouble(origin) + offset


def _solver_model(peaks, coordinates, center_origins):
    solver_peaks = deepcopy(peaks)
    targets = {}
    grids = []
    for peak, solver_peak in zip(peaks, solver_peaks, strict=True):
        grid = coordinates.copy()
        targets[id(peak)] = (solver_peak, 1)
        for dim, axis in peak["axes"].items():
            solver_axis = solver_peak["axes"][dim]
            factor = 1
            if id(axis) in center_origins:
                factor = np.longdouble(axis["width"])
                grid[dim] = _scaled_offsets(
                    grid[dim], center_origins[id(axis)], factor
                )[0]
                solver_axis["width"] = 1
            targets[id(axis)] = (solver_axis, factor)
        grids.append(grid)
    return solver_peaks, grids, targets


class _NonfiniteResidualError(ValueError):
    """A trial model cannot produce finite residuals."""


def _bounded_jacobian(vector, residuals, bounds):
    """Differentiate a reconstructed model using evaluable in-bounds steps."""
    try:
        baseline = residuals(vector)
    except _NonfiniteResidualError:
        return None
    jacobian = np.empty((baseline.size, vector.size))
    epsilon = np.sqrt(np.finfo(vector.dtype).eps)
    for index, value in enumerate(vector):
        step = epsilon * max(1, abs(value))
        lower, upper = bounds[:, index]
        with np.errstate(over="ignore"):
            left = min(upper, max(lower, value - step))
            right = min(upper, max(lower, value + step))
        candidates = (left, right) if value - left > right - value else (right, left)
        for candidate in candidates:
            if candidate == value or not np.isfinite(candidate):
                continue
            trial = vector.copy()
            trial[index] = candidate
            try:
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    column = (residuals(trial) - baseline) / (candidate - value)
            except _NonfiniteResidualError:
                continue
            if np.isfinite(column).all():
                jacobian[:, index] = column
                break
        else:
            return None
    return jacobian


def fit_separable_peaks(dataset, specs, max_calls):
    """Fit and reconstruct on the input grid with a fixed observation support."""
    coordinates, units = _grid(dataset)
    peaks, free, limits, names, parameter_units = _prepare_peaks(
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

    initial = [target[key] for target, key in free]
    origins = [
        value if key == "center" else 0
        for (_, key), value in zip(free, initial, strict=True)
    ]
    center_origins = {
        id(target): origin
        for (target, key), origin in zip(free, origins, strict=True)
        if key == "center"
    }
    intensity_scale = (
        max(np.max(np.abs(observed)), *(abs(peak["height"]) for peak in peaks)) or 1.0
    )
    scales = np.array(
        [
            (abs(target[key]) or intensity_scale)
            if key == "height"
            else target["width"]
            for target, key in free
        ],
        dtype=np.longdouble,
    )

    def model(peak_list, grids, *, keep_components=False):
        total = np.zeros(dataset.shape)
        components = []
        with np.errstate(over="ignore", invalid="ignore"):
            for peak, grid in zip(peak_list, grids, strict=True):
                component = _evaluate_separable_peak(dataset.shape, grid, peak)
                total += component
                if keep_components:
                    components.append(component)
        return total, components

    def residuals(fitted):
        with np.errstate(over="ignore", invalid="ignore"):
            residual = observed - fitted[valid]
        if not np.isfinite(residual).all():
            raise _NonfiniteResidualError(
                "ND fitting produced non-finite residuals on the support"
            )
        return residual

    solution = None
    values = []
    jacobian = None
    bounds = None
    if free:
        relative = np.array(
            [
                [_relative_value(bound, origin) for bound in pair]
                for pair, origin in zip(limits, origins, strict=True)
            ],
            dtype=np.longdouble,
        ).T
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            bounds = np.asarray(relative / scales, dtype=float)
            start = np.asarray(
                [
                    _relative_value(value, origin)
                    for value, origin in zip(initial, origins, strict=True)
                ]
                / scales,
                dtype=float,
            )
        if not np.isfinite(start).all() or np.any(
            np.isfinite(relative) & ~np.isfinite(bounds)
        ):
            raise ValueError("ND parameter bounds are outside the scaled solver range")

        solver_peaks, solver_grids, targets = _solver_model(
            peaks, coordinates, center_origins
        )
        solver_free = [(targets[id(target)], key) for target, key in free]

        def scaled_residuals(vector):
            for ((target, factor), key), value in zip(
                solver_free, scales * vector, strict=True
            ):
                target[key] = float(value) if key == "height" else value / factor
            fitted, _ = model(solver_peaks, solver_grids)
            with np.errstate(over="ignore", invalid="ignore"):
                scaled = residuals(fitted) / intensity_scale
            if not np.isfinite(scaled).all():
                raise _NonfiniteResidualError(
                    "ND residuals are outside the scaled solver range"
                )
            return scaled

        solution = least_squares(
            scaled_residuals,
            start,
            bounds=bounds,
            method="trf",
            jac="3-point",
            x_scale="jac",
            gtol=None,
            max_nfev=max_calls,
        )
        for (target, key), origin, offset in zip(
            free, origins, scales * solution.x, strict=True
        ):
            value = _physical_value(origin, offset)
            target[key] = float(value) if key == "height" else value
            values.append(target[key])
        public_vector = np.asarray(
            [
                _relative_value(value, origin) / scale
                for value, origin, scale in zip(values, origins, scales, strict=True)
            ],
            dtype=float,
        )
        epsilon = np.finfo(float).eps
        if not np.allclose(
            public_vector, solution.x, rtol=8 * epsilon, atol=8 * epsilon
        ):
            solution.jac = _bounded_jacobian(public_vector, scaled_residuals, bounds)
        solution.x = public_vector
        if solution.jac is not None:
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                jacobian = np.asarray(
                    solution.jac.astype(np.longdouble)
                    * (np.longdouble(intensity_scale) / scales),
                    dtype=float,
                )

    fitted, components = model(peaks, [coordinates] * len(peaks), keep_components=True)
    residual = np.zeros(dataset.shape)
    residual[valid] = residuals(fitted)
    if solution is not None:
        solution.fun = residual[valid] / intensity_scale
    diagnostics = _fit_diagnostics(observed, residual[valid], len(free))
    covariance, reason = _physical_covariance(
        solution, jacobian, scales, bounds, diagnostics
    )
    diagnostics["covariance_reason"] = reason
    history_peaks = [
        {
            "height": _history_magnitude(peak["height"]),
            "axes": {
                dim: {key: _history_magnitude(value) for key, value in axis.items()}
                for dim, axis in peak["axes"].items()
            },
        }
        for peak in peaks
    ]

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
                "peaks": history_peaks,
            },
            message=title,
        )
        return result

    return {
        "observed": output(data, dataset.title),
        "fitted": output(fitted, "fitted peak intensity"),
        "components": tuple(
            output(part, f"fitted peak {index}")
            for index, part in enumerate(components)
        ),
        "residuals": output(residual, "fit residuals"),
        "peaks": peaks,
        "values": np.asarray(values, dtype=np.longdouble),
        "names": names,
        "units": parameter_units,
        "jacobian": jacobian,
        "diagnostics": diagnostics,
        "covariance": covariance,
    }, solution


def _fit_diagnostics(observed, residuals, n_free):
    count = residuals.size
    dof = count - n_free
    amplitude = np.max(np.abs(residuals))
    scaled = residuals / (amplitude or 1.0)
    rms = np.sqrt(np.mean(np.square(scaled)))
    rmse = float(amplitude * rms)
    with np.errstate(over="ignore", under="ignore", divide="ignore"):
        rss = float(np.square(amplitude * np.linalg.norm(scaled)))
        reduced_chi_square = (
            float(np.square(rmse * np.sqrt(count / dof))) if dof else np.nan
        )
        log_mse = 2 * (np.log(amplitude) + np.log(rms)) if amplitude else -np.inf
        r_squared = np.nan
        if np.min(observed) != np.max(observed):
            observation_scale = np.max(np.abs(observed))
            relative_rmse = rmse / observation_scale
            observed_std = np.std(observed / observation_scale)
            if observed_std:
                r_squared = float(1 - np.square(relative_rmse / observed_std))
    adjusted = (
        1 - (1 - r_squared) * (count - 1) / dof
        if dof and count > 1 and np.isfinite(r_squared)
        else np.nan
    )
    return {
        "n_observations": count,
        "n_varying_parameters": n_free,
        "degrees_of_freedom": dof,
        "sse": rss,
        "rss": rss,
        "rss_status": (
            "numeric_range"
            if amplitude and (rss == 0 or not np.isfinite(rss))
            else "available"
        ),
        "rmse": rmse,
        "r_squared": r_squared,
        "reduced_chi_square": reduced_chi_square,
        "adjusted_r_squared": adjusted,
        "aic": float(count * log_mse + 2 * n_free),
        "bic": float(count * log_mse + n_free * np.log(count)),
    }


def _physical_covariance(solution, physical_jacobian, scales, bounds, diagnostics):
    """Return local covariance only when unconstrained inference is meaningful."""
    if solution is None:
        return None, "no_free_parameters"
    if not solution.success:
        return None, "not_converged"
    dof = diagnostics["degrees_of_freedom"]
    if dof <= 0:
        return None, "insufficient_degrees_of_freedom"
    if physical_jacobian is None:
        return None, "invalid_jacobian"
    if not np.isfinite(physical_jacobian).all():
        return None, "numeric_range"

    jacobian = solution.jac
    column_scale = np.max(np.abs(jacobian), axis=0)
    if np.any(column_scale == 0):
        return None, "rank_deficient"
    try:
        left, singular, right = np.linalg.svd(
            jacobian / column_scale, full_matrices=False
        )
    except np.linalg.LinAlgError:
        return None, "invalid_jacobian"
    rank_tolerance = np.sqrt(np.finfo(jacobian.dtype).eps)
    if singular[-1] <= singular[0] * rank_tolerance:
        return None, "rank_deficient"

    lower, upper = bounds
    near = _BOUND_NEIGHBORHOOD * np.maximum(1, np.abs(solution.x))
    near_lower = solution.x - lower <= near
    near_upper = upper - solution.x <= near
    if np.any(near_lower | near_upper):
        step = (right.T @ ((left.T @ solution.fun) / singular)) / column_scale
        unconstrained = solution.x - step
        precision = (
            16 * np.finfo(jacobian.dtype).eps * np.maximum(1, np.abs(solution.x))
        )
        if np.any(
            (near_lower & (unconstrained <= lower + precision))
            | (near_upper & (unconstrained >= upper - precision))
        ):
            return None, "active_bounds"

    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        sigma = np.hypot.reduce(solution.fun) / np.sqrt(dof)
        physical_scale = (np.longdouble(sigma) * scales) / column_scale
        scaled = np.asarray((right / singular[:, None]) * physical_scale, dtype=float)
        covariance = scaled.T @ scaled
    if not np.isfinite(covariance).all() or (
        sigma and np.any(np.diag(covariance) == 0)
    ):
        return None, "numeric_range"
    covariance.flags.writeable = False
    return covariance, None
