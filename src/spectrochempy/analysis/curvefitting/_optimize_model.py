"""Bounded fitting of full-grid callable models using the shared ND result path."""

from collections.abc import Mapping
from copy import copy

import numpy as np

from spectrochempy.analysis.curvefitting._fit_backend import _least_squares
from spectrochempy.analysis.curvefitting._fit_backend import _nonnegative_least_squares
from spectrochempy.analysis.curvefitting._optimize_nd import _fit_diagnostics
from spectrochempy.analysis.curvefitting._optimize_nd import _fit_outputs
from spectrochempy.analysis.curvefitting._optimize_nd import _grid
from spectrochempy.analysis.curvefitting._optimize_nd import _physical_covariance
from spectrochempy.analysis.curvefitting._optimize_nd import _prepare_parameter
from spectrochempy.analysis.curvefitting.simulation import _parameter_value
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.units import ur


def _parameters(specs):
    if not isinstance(specs, Mapping):
        raise ValueError("parameters must be a mapping of named parameter definitions")
    records, scales = [], []
    for name, definition in specs.items():
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError("Model parameter names must be Python identifiers")
        spec = (
            dict(definition)
            if isinstance(definition, Mapping)
            else {"value": definition}
        )
        unit = spec.pop("unit", None)
        unit = ur.Unit(unit) if unit is not None else None
        scale = spec.pop("scale", None)
        record = _prepare_parameter(spec, name, unit, identifier=("model", name))
        if not np.isfinite(record.value):
            raise ValueError(f"{name}: initial value must be finite")
        records.append(record)
        if not record.fixed:
            scale = (
                max(abs(float(record.value)), 1.0)
                if scale is None
                else _parameter_value(scale, unit, name, interval=True)
            )
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError(f"{name}: scale must be finite and positive")
            scales.append(scale)
    return records, np.asarray(scales, dtype=float)


def fit_callable_model(dataset, function, specs, max_calls, linear_amplitudes, options):
    """Fit a full-grid callback with a fixed, finite observation support."""
    coordinates, _ = _grid(dataset)
    records, scales = _parameters(specs)
    free = [record for record in records if not record.fixed]
    options = dict(options)
    allowed = {"ftol", "xtol", "gtol", "diff_step", "x_scale"}
    if options.keys() - allowed:
        raise ValueError(
            f"Unsupported callable-model solver options: {options.keys() - allowed}"
        )
    options.setdefault("x_scale", scales if free else 1.0)
    data = np.asarray(dataset.data, dtype=float)
    if np.any(
        np.isfinite(dataset.data) & ~np.isfinite(data) & ~np.asarray(dataset.mask)
    ):
        raise ValueError(
            "Observed intensity is outside the solver's floating-point range"
        )
    valid = np.isfinite(data) & ~np.asarray(dataset.mask)
    if not valid.any():
        raise ValueError("Model fitting requires finite, unmasked observations")
    observed = data[valid]
    intensity_scale = float(np.max(np.abs(observed))) or 1.0
    template = dataset.copy()
    template.data = np.zeros(dataset.shape)
    template._mask = ~valid
    for dim in dataset.dims:
        coord = template.coord(dim)
        if hasattr(coord, "default"):
            coord = coord.default
        coord._rounding = False
    values = {record.name: float(record.value) for record in records}
    component_names = None

    def array(value):
        if isinstance(value, NDDataset):
            result_coordinates, _ = _grid(value)
            if value.dims != dataset.dims or value.shape != dataset.shape:
                raise ValueError(
                    "Model datasets must preserve the input dimensions and shape"
                )
            for dim in dataset.dims:
                source, target = value.coord(dim), template.coord(dim)
                if hasattr(source, "default"):
                    source = source.default
                if hasattr(target, "default"):
                    target = target.default
                source_values = result_coordinates[dim]
                if source.units != target.units:
                    if source.units is None or target.units is None:
                        raise ValueError(
                            "Model coordinate units must match the input grid"
                        )
                    source_values = (
                        ur.Quantity(source_values, source.units)
                        .to(target.units)
                        .magnitude
                    )
                if not np.array_equal(source_values, coordinates[dim]):
                    raise ValueError(
                        "Model datasets must preserve the input coordinates"
                    )
            if np.any(np.asarray(value.mask) & valid):
                raise ValueError("Model masks cannot hide fitted observations")
            if value.units != dataset.units:
                if value.units is None or dataset.units is None:
                    raise ValueError(
                        "Model intensity units must match the observations"
                    )
                value = value.to(dataset.units)
            value = value.data
        value = np.asarray(value)
        if value.shape != dataset.shape or value.dtype.kind not in "iuf":
            raise ValueError(
                "Model components must be real arrays on the full input grid"
            )
        with np.errstate(over="ignore"):
            result = np.array(value, dtype=float, copy=True)
        if not np.isfinite(result[valid]).all():
            raise ValueError("Model evaluations must be finite on the fitting support")
        return result

    def evaluate(vector):
        nonlocal component_names
        values.update(zip((record.name for record in free), vector, strict=True))
        result = function(template.copy(), **values)
        if isinstance(result, Mapping):
            if not result or any(
                not isinstance(name, str) or not name for name in result
            ):
                raise ValueError("Model components require nonempty string names")
            names = tuple(result)
            components = [array(value) for value in result.values()]
        else:
            if linear_amplitudes:
                raise ValueError(
                    "linear_amplitudes requires a mapping of named components"
                )
            names, components = ("model",), [array(result)]
        if component_names is not None and names != component_names:
            raise ValueError(
                "Model component names and order must remain fixed during fitting"
            )
        component_names = names
        amplitudes = np.ones(len(components))
        if linear_amplitudes:
            matrix = np.column_stack([part[valid] for part in components])
            maxima = np.max(np.abs(matrix), axis=0)
            supported = maxima > 0
            amplitudes[:] = 0
            if supported.any():
                normalized = matrix[:, supported] / maxima[supported]
                norms = np.linalg.norm(normalized, axis=0)
                coefficients = _nonnegative_least_squares(
                    normalized / norms, observed / intensity_scale
                )
                coefficient, coefficient_power = np.frexp(coefficients)
                intensity, intensity_power = np.frexp(intensity_scale)
                norm, norm_power = np.frexp(norms)
                maximum, maximum_power = np.frexp(maxima[supported])
                with np.errstate(over="ignore", under="ignore"):
                    amplitudes[supported] = np.ldexp(
                        coefficient * intensity / norm / maximum,
                        coefficient_power
                        + intensity_power
                        - norm_power
                        - maximum_power,
                    )
            with np.errstate(over="ignore", invalid="ignore"):
                components = [
                    weight * part
                    for weight, part in zip(amplitudes, components, strict=True)
                ]
        with np.errstate(over="ignore", invalid="ignore"):
            fitted = np.sum(components, axis=0)
        if not np.isfinite(fitted[valid]).all():
            raise ValueError("The summed model must be finite on the fitting support")
        return fitted, components, dict(zip(names, amplitudes.tolist(), strict=True))

    initial = np.asarray([record.value for record in free], dtype=float)
    limits = np.asarray([record.bounds for record in free], dtype=float).reshape(-1, 2)
    fitted, components, amplitudes = evaluate(initial)
    n_linear = len(components) if linear_amplitudes else 0
    count_parameters = len(free) + n_linear
    if observed.size < count_parameters:
        raise ValueError(
            "Model fitting requires observations for all estimated parameters"
        )

    def residuals(vector):
        model = evaluate(vector)[0]
        with np.errstate(over="ignore", invalid="ignore"):
            residual = observed / intensity_scale - model[valid] / intensity_scale
        if not np.isfinite(residual).all():
            raise ValueError("Model residuals exceed the solver's floating-point range")
        return residual

    solution, jacobian, covariance = None, None, None
    if free:
        solution = _least_squares(
            residuals,
            initial,
            bounds=(limits[:, 0], limits[:, 1]),
            method="trf",
            max_nfev=max_calls,
            **options,
        )
        fitted, components, amplitudes = evaluate(solution.x)
        jacobian = np.asarray(solution.jac * intensity_scale, dtype=float)
    residual = np.zeros(dataset.shape)
    with np.errstate(over="ignore", invalid="ignore"):
        residual[valid] = observed - fitted[valid]
    if not np.isfinite(residual[valid]).all():
        raise ValueError("Physical residuals exceed the floating-point range")
    diagnostics = _fit_diagnostics(observed, residual[valid], count_parameters)
    diagnostics.update(
        n_nonlinear_parameters=len(free),
        n_linear_parameters=n_linear,
        active_parameter_bounds=tuple(
            record.name
            for record, active in zip(free, solution.active_mask, strict=True)
            if active
        )
        if solution is not None
        else (),
    )
    if linear_amplitudes:
        reason = "profiled_amplitudes"
        diagnostics["active_amplitude_bounds"] = tuple(
            name for name, value in amplitudes.items() if value == 0
        )
    elif solution is not None:
        scaled = copy(solution)
        scaled.x = solution.x / scales
        scaled.jac = solution.jac * scales
        covariance, reason = _physical_covariance(
            scaled,
            jacobian,
            scales,
            (limits[:, 0] / scales, limits[:, 1] / scales),
            diagnostics,
        )
    else:
        reason = "no_free_parameters"
    diagnostics["covariance_reason"] = reason
    identity = f"{getattr(function, '__module__', '')}.{getattr(function, '__qualname__', type(function).__qualname__)}"
    fit = _fit_outputs(
        dataset,
        data,
        valid,
        [],
        fitted,
        components,
        residual,
        [values[record.name] for record in free],
        tuple(record.name for record in free),
        tuple(record.unit for record in free),
        jacobian,
        diagnostics,
        covariance,
        history_parameters={
            "mode": "callable_model",
            "model": identity,
            "linear_amplitudes": linear_amplitudes,
            "component_names": list(component_names),
            "n_varying_parameters": count_parameters,
            "model_parameters": {
                record.name: {
                    "value": float(values[record.name]),
                    "fixed": record.fixed,
                    "unit": str(record.unit) if record.unit is not None else None,
                    "bounds": [
                        float(bound) if np.isfinite(bound) else None
                        for bound in record.bounds
                    ],
                }
                for record in records
            },
        },
    )
    fit["fitted_parameters"] = {
        record.name: values[record.name]
        if record.unit is None
        else ur.Quantity(values[record.name], record.unit)
        for record in records
    }
    fit["amplitudes"] = amplitudes if linear_amplitudes else {}
    fit["component_names"] = component_names
    return fit, solution
