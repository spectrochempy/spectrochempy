"""Global fitting of arbitrary separable profiles and shared parameters."""

from collections.abc import Mapping
from collections.abc import Sequence
from copy import copy
from dataclasses import replace

import numpy as np

from spectrochempy.analysis.curvefitting._fit_backend import (
    _least_squares as least_squares,
)
from spectrochempy.analysis.curvefitting._optimize_nd import _fit_diagnostics
from spectrochempy.analysis.curvefitting._optimize_nd import _fit_outputs
from spectrochempy.analysis.curvefitting._optimize_nd import _grid
from spectrochempy.analysis.curvefitting._optimize_nd import _physical_covariance
from spectrochempy.analysis.curvefitting._optimize_nd import _prepare_parameter
from spectrochempy.analysis.curvefitting._prepared_parameter import _ParameterLayout
from spectrochempy.analysis.curvefitting._profile_models import axis_parameters
from spectrochempy.analysis.curvefitting._profile_models import evaluate_profiles
from spectrochempy.analysis.curvefitting._profile_models import profile_history
from spectrochempy.analysis.curvefitting._profile_models import public_profiles
from spectrochempy.analysis.curvefitting.simulation import _interval_unit
from spectrochempy.analysis.curvefitting.simulation import _parameter_value
from spectrochempy.analysis.curvefitting.simulation import _scaled_offsets
from spectrochempy.core.units import Quantity
from spectrochempy.core.units import ur


def _prepare_profiles(specs, shared, intensity_unit, units, *, include_fixed=False):
    """Prepare bindings, optionally retaining fixed groups for private qualification."""
    if not isinstance(specs, Sequence) or isinstance(specs, (str, bytes)) or not specs:
        raise ValueError("peaks must be a nonempty ordered sequence")
    if shared is None:
        shared = {}
    if not isinstance(shared, Mapping) or any(
        not isinstance(key, str) for key in shared
    ):
        raise ValueError("shared_parameters must be a mapping with string names")
    groups, shared_groups, peaks, target_units = [], {}, [], {}

    def register(
        target,
        key,
        raw,
        unit,
        name,
        positive=False,
        interval=False,
        infer_unit=False,
        identifier=None,
    ):
        spec = dict(raw) if isinstance(raw, Mapping) else {"value": raw}
        declared = spec.pop("unit", None)
        shared_name = spec.get("shared")
        if "shared" in spec:
            if (
                set(spec) != {"shared"}
                or not isinstance(shared_name, str)
                or shared_name not in shared
            ):
                raise ValueError(
                    f"{name}: expected a reference to a declared shared parameter"
                )
            if shared_name not in shared_groups:
                definition = shared[shared_name]
                definition = (
                    dict(definition)
                    if isinstance(definition, Mapping)
                    else {"value": definition}
                )
                shared_unit = definition.pop("unit", None)
                canonical = (
                    ur.Unit(shared_unit)
                    if shared_unit is not None
                    else getattr(definition.get("value"), "units", None)
                    or unit
                    or ur.dimensionless
                )
                if interval and canonical is not None:
                    canonical = _interval_unit(canonical)
                parameter = _prepare_parameter(
                    definition,
                    shared_name,
                    canonical,
                    width=positive,
                    identifier=("shared_parameters", shared_name),
                )
                group = {
                    "parameter": replace(
                        parameter, name=f"shared_parameters[{shared_name!r}]"
                    ),
                    "bindings": [],
                }
                groups.append(group)
                shared_groups[shared_name] = group
            group = shared_groups[shared_name]
            source_unit = group["parameter"].unit
            if (
                unit is None
                and declared is None
                and (infer_unit or source_unit.dimensionless)
            ):
                unit = source_unit
            if (source_unit is None) != (unit is None):
                raise ValueError(f"{name}: shared parameter units are incompatible")

            def convert(value):
                if source_unit is None:
                    return value
                return _parameter_value(
                    Quantity(value, source_unit), unit, name, interval=interval
                )

            initial, lower, upper, fixed = group["parameter"].as_legacy_tuple()
            if positive and (
                convert(initial) <= 0 or (not fixed and convert(lower) <= 0)
            ):
                raise ValueError(f"{name}: value and free lower bound must be positive")
            convert(initial)
        else:
            if declared is not None and ur.Unit(declared) != unit:
                raise ValueError(f"{name}: inconsistent declared units")
            parameter = _prepare_parameter(
                spec,
                name,
                unit,
                width=positive,
                identifier=identifier,
            )
            group = {
                "parameter": replace(
                    parameter,
                    unit=_interval_unit(unit)
                    if interval and unit is not None
                    else unit,
                ),
                "bindings": [],
            }
            groups.append(group)

            def convert(value):
                return value

        target_units[(id(target), key)] = unit
        group["relative_step"] = group.get("relative_step", False) or positive
        group["bindings"].append((target, key, convert))
        target[key] = convert(group["parameter"].value)

    for index, spec in enumerate(specs):
        if not isinstance(spec, Mapping) or set(spec) != {"height", "axes"}:
            raise ValueError("Each peak requires height and axes")
        if not isinstance(spec["axes"], Mapping) or set(spec["axes"]) != set(units):
            raise ValueError("Each peak must specify every dataset dimension")
        peak = {"axes": {}}
        register(
            peak,
            "height",
            spec["height"],
            intensity_unit,
            f"peaks[{index}].height",
            identifier=("peaks", index, "height"),
        )
        for dim in sorted(units):
            model, entries = axis_parameters(spec["axes"][dim], units[dim], dim)
            params = {}
            for key, raw, unit, positive, interval in entries:
                if model in ("voigt", "pseudovoigt") and key == "ratio":
                    raw = dict(raw) if isinstance(raw, Mapping) else {"value": raw}
                    if "shared" not in raw:
                        raw.setdefault("bounds", (0, 1))
                register(
                    params,
                    key,
                    raw,
                    unit,
                    f"peaks[{index}].axes[{dim!r}].parameters[{key!r}]",
                    positive,
                    interval,
                    infer_unit=callable(model),
                    identifier=("peaks", index, "axes", dim, "parameters", key),
                )
                if model in ("voigt", "pseudovoigt") and key == "ratio":
                    group = (
                        shared_groups[raw["shared"]] if "shared" in raw else groups[-1]
                    )
                    value, low, high, fixed = group["parameter"].as_legacy_tuple()
                    convert = group["bindings"][-1][2]
                    value = convert(value)
                    valid_bounds = fixed or (
                        np.isfinite(low)
                        and np.isfinite(high)
                        and 0 <= convert(low) < convert(high) <= 1
                    )
                    if not 0 <= value <= 1 or not valid_bounds:
                        raise ValueError("Voigt ratio requires bounds within [0, 1]")
            peak["axes"][dim] = {"model": model, "parameters": params}
        peaks.append(peak)
    if set(shared) != set(shared_groups):
        raise ValueError(
            "Every shared parameter must be referenced by a profile or height"
        )
    return (
        peaks,
        groups
        if include_fixed
        else [group for group in groups if not group["parameter"].fixed],
        target_units,
    )


def _prepare_profile_layout(specs, shared, intensity_unit, units):
    """Prepare the complete physical context without selecting solver coordinates."""
    peaks, groups, target_units = _prepare_profiles(
        specs, shared, intensity_unit, units, include_fixed=True
    )
    layout = _ParameterLayout(tuple(group["parameter"] for group in groups))
    return peaks, groups, target_units, layout


def fit_profiles(
    dataset, specs, max_calls, shared, *, min_ndim=2, residual_jacobian=None
):
    """Fit general profiles using the same observation and result contract as ND peaks."""
    coordinates, units = _grid(dataset, min_ndim=min_ndim)
    peaks, groups, target_units = _prepare_profiles(specs, shared, dataset.units, units)
    with np.errstate(over="ignore"):
        data = np.asarray(dataset.data, dtype=float)
    unmasked = ~np.asarray(dataset.mask)
    if np.any(np.isfinite(dataset.data) & ~np.isfinite(data) & unmasked):
        raise ValueError(
            "Observed intensity is outside the solver's floating-point range"
        )
    valid = np.isfinite(data) & unmasked
    count = np.count_nonzero(valid)
    if not count or count < len(groups):
        raise ValueError("ND fitting requires observations for all free parameters")
    observed = data[valid]
    intensity_scale = np.max(np.abs(observed)) or 1.0
    layout = _ParameterLayout(tuple(group["parameter"] for group in groups))
    initial = np.array(layout.free_values, dtype=np.longdouble)
    limits = np.array(layout.free_bounds, dtype=np.longdouble).T
    binding_groups = {
        (id(target), key): group
        for group in groups
        for target, key, _ in group["bindings"]
    }
    for peak in peaks:
        for axis in peak["axes"].values():
            parameters = axis["parameters"]
            group = binding_groups.get((id(parameters), "center"))
            if callable(axis["model"]) or group is None:
                continue
            width = parameters["width"]
            unit = target_units[(id(parameters), "width")]
            if unit is not None:
                width = _parameter_value(
                    Quantity(width, unit),
                    group["parameter"].unit,
                    "center scale",
                    interval=True,
                )
            group["center_scale"] = min(group.get("center_scale", width), width)
    origins = np.array(
        [
            value if "center_scale" in group else 0
            for group, value in zip(groups, initial, strict=True)
        ],
        dtype=np.longdouble,
    )
    scales = np.array(
        [
            group.get(
                "center_scale",
                abs(value)
                or (
                    group["parameter"].bounds[1] - group["parameter"].bounds[0]
                    if np.isfinite(group["parameter"].bounds).all()
                    else 1.0
                ),
            )
            for group, value in zip(groups, initial, strict=True)
        ],
        dtype=np.longdouble,
    )
    layout = replace(layout, normalization_scales=tuple(scales))
    scales = np.asarray(layout.free_scales, dtype=np.longdouble)
    bounds = np.asarray((limits - origins) / scales, dtype=float) if groups else None
    start = np.asarray((initial - origins) / scales, dtype=float)
    names = layout.free_names
    parameter_units = layout.free_units
    group_indices = {id(group): index for index, group in enumerate(groups)}
    solver_grids = []
    centered_axes = {}
    for peak in peaks:
        grid = coordinates.copy()
        for dim, axis in peak["axes"].items():
            parameters = axis["parameters"]
            group = binding_groups.get((id(parameters), "center"))
            if callable(axis["model"]) or group is None:
                continue
            index = group_indices[id(group)]
            center = parameters["center"]
            width = parameters["width"]
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                grid[dim] = _scaled_offsets(coordinates[dim], center, width)[0]
            centered_axes[id(parameters)] = (index, units[dim], width)
        solver_grids.append(grid)
    offsets = initial - origins
    centered_offsets = np.zeros_like(offsets)

    def update(vector):
        nonlocal offsets
        offsets = vector * scales
        vector = origins + offsets
        for group, value in zip(groups, vector, strict=True):
            if "center_scale" not in group:
                value = float(value)
            for target, key, convert in group["bindings"]:
                target[key] = convert(value)

    def model(*, keep_components=False, physical=False):
        total = np.zeros(dataset.shape)
        parts = []
        with np.errstate(over="ignore", invalid="ignore"):
            for peak, grid in zip(peaks, solver_grids, strict=True):
                if physical:
                    resolved = peak
                    grid = coordinates
                else:
                    axes = {}
                    for dim, axis in peak["axes"].items():
                        parameters = axis["parameters"]
                        binding = centered_axes.get(id(parameters))
                        if binding is None:
                            axes[dim] = axis
                            continue
                        index, unit, reference_width = binding
                        value = offsets[index] - centered_offsets[index]
                        if unit is not None:
                            value = _parameter_value(
                                Quantity(value, parameter_units[index]),
                                unit,
                                "center offset",
                                interval=True,
                            )
                        axes[dim] = {
                            "model": axis["model"],
                            "parameters": {
                                **parameters,
                                "center": value / reference_width,
                                "width": parameters["width"] / reference_width,
                            },
                        }
                    resolved = {"height": peak["height"], "axes": axes}
                part = evaluate_profiles(dataset.shape, grid, resolved)
                total += part
                if keep_components:
                    parts.append(part)
        return total, parts

    def residuals(vector):
        update(vector)
        fitted, _ = model()
        with np.errstate(over="ignore", invalid="ignore"):
            residual = (observed - fitted[valid]) / intensity_scale
        if not np.isfinite(residual).all():
            raise ValueError("ND fitting produced non-finite residuals on the support")
        return residual

    def analytic_at(vector, *, scaled):
        update(vector)
        matrix = np.asarray(residual_jacobian(peaks, names), dtype=np.longdouble)
        if matrix.shape != (*dataset.shape, len(groups)):
            raise ValueError("Profile Jacobian must match the grid and free parameters")
        matrix = matrix[valid]
        if scaled:
            matrix = matrix * scales / intensity_scale
        with np.errstate(over="ignore", invalid="ignore"):
            matrix = np.asarray(matrix, dtype=float)
        if not np.isfinite(matrix).all():
            if scaled:
                raise ValueError("ND fitting produced a non-finite analytic Jacobian")
            return None
        return matrix

    solution, jacobian = None, None
    represented_center = False
    if groups:
        solution = least_squares(
            residuals,
            start,
            bounds=bounds,
            method="trf",
            jac=(lambda vector: analytic_at(vector, scaled=True))
            if residual_jacobian is not None
            else "3-point",
            x_scale="jac",
            max_nfev=max_calls,
            gtol=None,
        )
        if centered_axes:
            represented = solution.x.copy()
            for index, group in enumerate(groups):
                if "center_scale" in group:
                    value = origins[index] + solution.x[index] * scales[index]
                    represented[index] = (value - origins[index]) / scales[index]
            precision = 8 * np.finfo(float).eps
            represented_center = not np.allclose(
                represented, solution.x, rtol=precision, atol=precision
            )
            if represented_center:
                solution.x = represented
        update(solution.x)
        if centered_axes:
            centered_offsets = offsets.copy()
            for peak, grid in zip(peaks, solver_grids, strict=True):
                for dim, axis in peak["axes"].items():
                    parameters = axis["parameters"]
                    binding = centered_axes.get(id(parameters))
                    if binding is not None:
                        grid[dim] = _scaled_offsets(
                            coordinates[dim], parameters["center"], binding[2]
                        )[0]
            represented_center = True

        def fitted_at(vector):
            update(vector)
            fitted, _ = model()
            return fitted[valid]

        jacobian = (
            analytic_at(solution.x, scaled=False)
            if residual_jacobian is not None
            else _physical_profile_jacobian(
                solution.x,
                fitted_at,
                bounds,
                scales,
                [
                    abs((origin + value * scale) / scale)
                    if group["relative_step"]
                    else max(1.0, abs(value))
                    for group, value, origin, scale in zip(
                        groups, solution.x, origins, scales, strict=True
                    )
                ],
            )
        )
        update(solution.x)
    fitted, components = model(keep_components=True, physical=True)
    residual = np.zeros(dataset.shape)
    residual[valid] = observed - fitted[valid]
    if not np.isfinite(residual[valid]).all():
        raise ValueError("ND fitting produced non-finite residuals on the support")
    if solution is not None:
        solution.fun = residual[valid] / intensity_scale
    diagnostics = _fit_diagnostics(observed, residual[valid], len(groups))
    covariance_solution = solution
    if (
        solution is not None
        and jacobian is not None
        and (residual_jacobian is None or represented_center)
    ):
        covariance_solution = copy(solution)
        mantissa, power = np.frexp(jacobian)
        scale, scale_power = np.frexp(scales)
        intensity, intensity_power = np.frexp(intensity_scale)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            covariance_solution.jac = np.asarray(
                np.ldexp(
                    mantissa * scale / intensity, power + scale_power - intensity_power
                ),
                dtype=float,
            )
    if represented_center:
        solution.jac = covariance_solution.jac if jacobian is not None else None
    covariance, reason = _physical_covariance(
        covariance_solution, jacobian, scales, bounds, diagnostics
    )
    diagnostics["covariance_reason"] = reason
    values = origins + solution.x * scales if solution is not None else []
    exported = public_profiles(peaks, target_units)
    return _fit_outputs(
        dataset,
        data,
        valid,
        exported,
        fitted,
        components,
        residual,
        values,
        names,
        parameter_units,
        jacobian,
        diagnostics,
        covariance,
        history_parameters={
            "profile_specification": profile_history(specs),
            "shared_parameters": profile_history(shared or {}),
        },
    ), solution


def _physical_profile_jacobian(vector, evaluate, bounds, scales, step_scales):
    """Differentiate model values before residual normalization loses small tails."""
    jacobian = None
    step_factor = np.cbrt(np.finfo(float).eps)
    for index, value in enumerate(vector):
        step = step_factor * (step_scales[index] or 1.0)
        left = max(bounds[0, index], value - step)
        right = min(bounds[1, index], value + step)
        if left == right:
            return None
        first, second = vector.copy(), vector.copy()
        first[index], second[index] = left, right
        a, b = evaluate(first), evaluate(second)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            return None
        if jacobian is None:
            jacobian = np.empty((a.size, vector.size))
        magnitude = np.maximum(np.abs(a), np.abs(b))
        reference = np.where(magnitude == 0, 1.0, magnitude)
        slope = (a / reference - b / reference) / (right - left)
        numerator, numerator_power = np.frexp(slope)
        factor, factor_power = np.frexp(magnitude)
        scale, scale_power = np.frexp(scales[index])
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            jacobian[:, index] = np.ldexp(
                numerator * factor / scale,
                numerator_power + factor_power - scale_power,
            )
    return jacobian
