"""Bounded fits of separable peaks on a native spectral grid."""

from collections.abc import Mapping
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import replace

import numpy as np
from pint import Quantity as PintQuantity

from spectrochempy.analysis.curvefitting._fit_backend import (
    _least_squares as least_squares,
)
from spectrochempy.analysis.curvefitting._prepared_parameter import _ParameterLayout
from spectrochempy.analysis.curvefitting._prepared_parameter import _PreparedParameter
from spectrochempy.analysis.curvefitting._profile_models import extended_profiles
from spectrochempy.analysis.curvefitting._profile_models import profile_history
from spectrochempy.analysis.curvefitting.simulation import _MODELS
from spectrochempy.analysis.curvefitting.simulation import _evaluate_separable_peak
from spectrochempy.analysis.curvefitting.simulation import _interval_unit
from spectrochempy.analysis.curvefitting.simulation import _parameter_value
from spectrochempy.analysis.curvefitting.simulation import _scaled_offsets
from spectrochempy.core.dataset.coord import Coord
from spectrochempy.core.dataset.coordset import CoordSet
from spectrochempy.core.dataset.nddataset import NDDataset

_BOUND_NEIGHBORHOOD = 1e-5
_GAUSSIAN_EXPONENT = 4 * np.log(2)


def snapshot_peaks(spec):
    """Copy the peak specification without requiring pickleable mappings."""
    if isinstance(spec, PintQuantity):
        return deepcopy(spec)
    if isinstance(spec, Mapping):
        return {key: snapshot_peaks(value) for key, value in spec.items()}
    if isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)):
        return tuple(snapshot_peaks(value) for value in spec)
    return spec


def _grid(dataset, *, min_ndim=2):
    if (
        not isinstance(dataset, NDDataset)
        or dataset.ndim < min_ndim
        or not dataset.size
    ):
        raise ValueError(
            "ND fitting requires a nonempty NDDataset with at least "
            + ("two axes" if min_ndim == 2 else "one axis")
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
    """Return the historical ND parser tuple."""
    return _prepare_parameter(spec, name, unit, width=width).as_legacy_tuple()


def _prepare_parameter(spec, name, unit, *, width=False, identifier=None):
    """Normalize an ND declaration into a physical parameter record."""
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
    return _PreparedParameter(
        identifier if identifier is not None else (name,),
        name,
        value,
        (low, high),
        fixed,
        _interval_unit(unit) if width and unit is not None else unit,
    )


def _prepare_peaks(specs, intensity_unit, units):
    """Preserve the historical peaks/bindings/bounds/names/units protocol."""
    peaks, free, layout = _prepare_peak_layout(specs, intensity_unit, units)
    return peaks, free, list(layout.free_bounds), layout.free_names, layout.free_units


def _prepare_peak_layout(specs, intensity_unit, units):
    if not isinstance(specs, Sequence) or isinstance(specs, (str, bytes)) or not specs:
        raise ValueError("peaks must be a nonempty ordered sequence")

    peaks, free, parameters = [], [], []

    def register(target, key, parameter, name):
        target[key] = parameter.value
        parameters.append(replace(parameter, name=name))
        if not parameter.fixed:
            free.append((target, key))

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
            _prepare_parameter(
                spec["height"],
                "height",
                intensity_unit,
                identifier=("peaks", index, "height"),
            ),
            f"peaks[{index}].height",
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
                    _prepare_parameter(
                        axis[key],
                        f"{dim} {key}",
                        units[dim],
                        width=key == "width",
                        identifier=("peaks", index, "axes", dim, key),
                    ),
                    f"peaks[{index}].axes[{dim!r}].{key}",
                )
            peak["axes"][dim] = resolved
        peaks.append(peak)
    return peaks, free, _ParameterLayout(tuple(parameters))


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


def _axis_log_derivatives(values, axis):
    """Return the negative log profile and the log magnitude of its slopes."""
    offsets, displacement, overflow = _scaled_offsets(
        values, axis["center"], axis["width"]
    )
    log_distance = np.asarray(np.log(np.abs(offsets)), dtype=float)
    recover = ~np.isfinite(offsets) | (
        (np.abs(offsets) < np.finfo(offsets.dtype).tiny) & (displacement != 0)
    )
    if np.any(recover):
        log_distance[recover] = (
            np.log(np.abs(displacement[recover]))
            - np.log(np.longdouble(axis["width"]))
            + np.log(2) * overflow[recover]
        )
    if axis["model"] == "gaussian":
        power = np.asarray((_GAUSSIAN_EXPONENT * offsets) * offsets, dtype=float)
        power[recover] = np.exp(np.log(_GAUSSIAN_EXPONENT) + 2 * log_distance[recover])
        log_center = np.log(2 * _GAUSSIAN_EXPONENT) + log_distance
    else:
        power = np.logaddexp(0.0, np.log(4) + 2 * log_distance)
        log_center = np.log(8) + log_distance - power
    return power, log_center, log_center + log_distance, np.signbit(displacement)


def _separable_jacobian(peak_grids, columns, support, log_scales):
    """Differentiate residuals on the fixed support in the requested units."""
    count = support[0].size
    jacobian = np.empty((count, len(columns)), order="F")
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        for peak, grid in peak_grids:
            log_profile = np.zeros(count)
            axis_columns = []
            for (dim, values), indices in zip(grid.items(), support, strict=True):
                axis = peak["axes"][dim]
                power, center, width, negative = _axis_log_derivatives(values, axis)
                log_profile -= power[indices]
                axis_columns.append((axis, indices, center, width, negative))

            height_column = columns.get((id(peak), "height"))
            if height_column is not None:
                index, _ = height_column
                jacobian[:, index] = -np.exp(log_profile + log_scales[index])

            height = peak["height"]
            log_height = np.log(np.longdouble(abs(height)))
            for axis, indices, center, width, negative in axis_columns:
                log_weight = log_height - np.log(np.longdouble(axis["width"]))
                for key, slope in (("center", center), ("width", width)):
                    column = columns.get((id(axis), key))
                    if column is None:
                        continue
                    index, log_factor = column
                    if height == 0:
                        jacobian[:, index] = 0.0
                        continue
                    weight = float(log_weight + log_scales[index] - log_factor)
                    magnitude = np.exp(log_profile + weight + slope[indices])
                    sign = 1.0 if height < 0 else -1.0
                    if key == "center":
                        sign = np.where(negative[indices], -sign, sign)
                    jacobian[:, index] = sign * magnitude
    return jacobian


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


def fit_separable_peaks(dataset, specs, max_calls, shared_parameters=None):
    """Fit and reconstruct on the input grid with a fixed observation support."""
    if shared_parameters is not None or extended_profiles(specs):
        from spectrochempy.analysis.curvefitting._optimize_profiles import (  # noqa: PLC0415
            fit_profiles,
        )

        return fit_profiles(dataset, specs, max_calls, shared_parameters)
    coordinates, units = _grid(dataset)
    peaks, free, layout = _prepare_peak_layout(specs, dataset.units, units)
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

    initial = layout.free_values
    limits = layout.free_bounds
    names = layout.free_names
    parameter_units = layout.free_units
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
    layout = replace(layout, normalization_scales=tuple(scales))
    scales = np.asarray(layout.free_scales, dtype=np.longdouble)

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

        def update_solver_parameters(vector):
            for ((target, factor), key), value in zip(
                solver_free, scales * vector, strict=True
            ):
                target[key] = float(value) if key == "height" else value / factor

        def scaled_residuals(vector):
            update_solver_parameters(vector)
            fitted, _ = model(solver_peaks, solver_grids)
            with np.errstate(over="ignore", invalid="ignore"):
                scaled = residuals(fitted) / intensity_scale
            if not np.isfinite(scaled).all():
                raise _NonfiniteResidualError(
                    "ND residuals are outside the scaled solver range"
                )
            return scaled

        use_analytic_jacobian = dataset.ndim in (2, 3)
        if use_analytic_jacobian:
            support = np.nonzero(valid)
            log_scales = np.log(scales) - np.log(np.longdouble(intensity_scale))
            columns = {
                (id(target), key): (index, np.log(factor))
                for index, ((target, factor), key) in enumerate(solver_free)
            }
            active_targets = {target for target, _ in columns}
            active_peaks = [
                index
                for index, peak in enumerate(solver_peaks)
                if id(peak) in active_targets
                or any(id(axis) in active_targets for axis in peak["axes"].values())
            ]
            peak_grids = [(solver_peaks[i], solver_grids[i]) for i in active_peaks]

            def jacobian_at(vector):
                update_solver_parameters(vector)
                result = _separable_jacobian(peak_grids, columns, support, log_scales)
                if np.isfinite(result).all():
                    return result
                return _bounded_jacobian(vector, scaled_residuals, bounds)

            def solver_jacobian(vector):
                result = jacobian_at(vector)
                if result is None:
                    raise ValueError(
                        "ND fitting could not evaluate a finite Jacobian on the support"
                    )
                return result

            jac = solver_jacobian
        else:
            jac = "3-point"

            def jacobian_at(vector):
                return _bounded_jacobian(vector, scaled_residuals, bounds)

        solution = least_squares(
            scaled_residuals,
            start,
            bounds=bounds,
            method="trf",
            jac=jac,
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
            solution.jac = jacobian_at(public_vector)
        solution.x = public_vector
        if solution.jac is not None:
            if use_analytic_jacobian:
                physical_columns = {
                    (id(target), key): (index, 0.0)
                    for index, (target, key) in enumerate(free)
                }
                jacobian = _separable_jacobian(
                    [(peaks[i], coordinates) for i in active_peaks],
                    physical_columns,
                    support,
                    np.zeros_like(scales),
                )
            if jacobian is None or np.isnan(jacobian).any():
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
    return _fit_outputs(
        dataset,
        data,
        valid,
        peaks,
        fitted,
        components,
        residual,
        values,
        names,
        parameter_units,
        jacobian,
        diagnostics,
        covariance,
    ), solution


def _fit_outputs(
    dataset,
    data,
    valid,
    peaks,
    fitted,
    components,
    residual,
    values,
    names,
    parameter_units,
    jacobian,
    diagnostics,
    covariance,
    history_parameters=None,
):
    """Package fitted components and diagnostics without changing their grid."""
    count = np.count_nonzero(valid)
    history_peaks = profile_history(peaks)

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
                "n_varying_parameters": len(names),
                "parameter_order": list(names),
                "peaks": history_peaks,
                **(history_parameters or {}),
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
    }


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
