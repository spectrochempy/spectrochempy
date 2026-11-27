"""Bounded fitting of full-grid callable models using the shared ND result path."""

from collections.abc import Mapping
from copy import copy
from copy import deepcopy

import numpy as np

from spectrochempy.analysis.curvefitting._fit_backend import _least_squares
from spectrochempy.analysis.curvefitting._fit_backend import _nonnegative_least_squares
from spectrochempy.analysis.curvefitting._optimize_nd import _fit_diagnostics
from spectrochempy.analysis.curvefitting._optimize_nd import _fit_outputs
from spectrochempy.analysis.curvefitting._optimize_nd import _grid
from spectrochempy.analysis.curvefitting._optimize_nd import _physical_covariance
from spectrochempy.analysis.curvefitting._optimize_nd import _prepare_parameter
from spectrochempy.analysis.curvefitting._parameter_coordinates import (
    _ParameterCoordinates,
)
from spectrochempy.analysis.curvefitting._parameter_coordinates import _real_array
from spectrochempy.analysis.curvefitting._prepared_parameter import _ParameterLayout
from spectrochempy.analysis.curvefitting.simulation import _parameter_value
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.units import ur


def _parameter_layout(specs):
    """Normalize physical records and existing callback conditioning scales once."""
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
        if not record.fixed:
            scale = (
                max(abs(float(record.value)), 1.0)
                if scale is None
                else _parameter_value(scale, unit, name, interval=True)
            )
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError(f"{name}: scale must be finite and positive")
            scales.append(scale)
        records.append(record)
    return _ParameterLayout(tuple(records), tuple(scales))


def _parameters(specs):
    """Return the historical records/scales pair from the shared layout."""
    layout = _parameter_layout(specs)
    return list(layout.parameters), np.asarray(layout.free_scales, dtype=float)


def _bind_coordinates(layout, coordinates, linear_amplitudes, options):
    """Snapshot explicit coordinates against every current physical declaration."""
    if not isinstance(coordinates, _ParameterCoordinates):
        raise TypeError("parameter_coordinates must be a prepared coordinate space")
    if linear_amplitudes:
        raise ValueError("Explicit coordinates do not support projected amplitudes")
    for record in coordinates.layout.parameters:
        _real_array(record.value, (), "Physical parameter value")
        _real_array(record.bounds, (2,), "Physical parameter bounds", finite=False)
    if coordinates.layout.parameters != layout.parameters:
        raise ValueError("Coordinate declarations must match the current model layout")
    if "x_scale" in options:
        raise ValueError("Explicit coordinates own x_scale; use their normalization")
    for record in layout.parameters:
        _real_array(record.value, (), "Physical parameter value")
    for record, transform in zip(
        layout.free_parameters, coordinates.transforms, strict=True
    ):
        if transform == "log" and record.unit is not None:
            zero = ur.Quantity(0, record.unit).to_base_units().magnitude
            if zero != 0:
                raise ValueError(
                    "Log coordinates require multiplicative parameter units"
                )
    return deepcopy(coordinates)


def _coordinate_config(coordinates):
    """Retain initial declarations separately from fitted physical values."""
    return {
        record.name: {
            "coordinate_transform": transform,
            "normalization_scale": float(normalization),
            "log_reference": reference,
            "initial_value": float(record.value),
            "bounds": [float(b) if np.isfinite(b) else None for b in record.bounds],
            "unit": str(record.unit) if record.unit is not None else None,
        }
        for record, transform, normalization, reference in zip(
            coordinates.layout.free_parameters,
            coordinates.transforms,
            coordinates.layout.free_scales,
            coordinates.references,
            strict=True,
        )
    }


class _JacobianRangeError(ValueError):
    """A restored Jacobian cannot be represented in float64 magnitudes."""


def _rescale_jacobian(matrix, numerator, derivatives):
    """Restore residual/column scales without an overflowing intermediate."""
    matrix = np.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError("Solver Jacobian must be two dimensional")
    matrix = _real_array(matrix, (matrix.shape[0], len(derivatives)), "Solver Jacobian")
    values, powers = np.frexp(matrix)
    factors, factor_powers = np.frexp(numerator)
    divisor, divisor_powers = np.frexp(derivatives)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = np.ldexp(
            values * factors / divisor, powers + factor_powers - divisor_powers
        )
    if not np.isfinite(result).all() or np.any((matrix != 0) & (result == 0)):
        raise _JacobianRangeError("Restored physical Jacobian is outside float64 range")
    return result


def fit_callable_model(
    dataset,
    function,
    specs,
    max_calls,
    linear_amplitudes,
    options,
    *,
    parameter_coordinates=None,
):
    """
    Fit a full-grid callback with a fixed, finite observation support.

    Private explicit coordinates require matching physical declarations and no
    projected amplitudes. Their normalization is independent of the historical
    physical scales. The solver uses three-point differences and x_scale=1;
    tolerances and diff_step still apply in solver coordinates, with unchanged
    SciPy defaults unless supplied. No stopping-policy improvement is implied.
    The returned solution retains raw solver coordinates and scaled residuals;
    fit outputs, including the Jacobian and covariance, remain physical.
    """
    coordinates, _ = _grid(dataset)
    layout = _parameter_layout(specs)
    records, free = layout.parameters, layout.free_parameters
    scales = np.asarray(layout.free_scales, dtype=float)
    options = dict(options)
    allowed = {"ftol", "xtol", "gtol", "diff_step", "x_scale"}
    if options.keys() - allowed:
        raise ValueError(
            f"Unsupported callable-model solver options: {options.keys() - allowed}"
        )
    if parameter_coordinates is not None:
        parameter_coordinates = _bind_coordinates(
            layout, parameter_coordinates, linear_amplitudes, options
        )
        options["x_scale"] = 1.0
        coordinate_config = _coordinate_config(parameter_coordinates)
    else:
        options.setdefault("x_scale", scales if free else 1.0)
        coordinate_config = None
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
        values.update(zip(layout.free_names, vector, strict=True))
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

    initial = np.asarray(layout.free_values, dtype=float)
    limits = np.asarray(layout.free_bounds, dtype=float).reshape(-1, 2)
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
        if parameter_coordinates is not None:
            solver_initial = parameter_coordinates.initial
            solver_limits = np.asarray(parameter_coordinates.bounds)

            def objective(vector):
                return residuals(parameter_coordinates.to_physical(vector))

            coordinate_options = {"jac": "3-point"}
        else:
            solver_initial, solver_limits, objective = initial, limits, residuals
            coordinate_options = {}
        solution = _least_squares(
            objective,
            solver_initial,
            bounds=(solver_limits[:, 0], solver_limits[:, 1]),
            method="trf",
            max_nfev=max_calls,
            **options,
            **coordinate_options,
        )
        if parameter_coordinates is not None:
            physical_values = parameter_coordinates.to_physical(solution.x)
            derivatives = parameter_coordinates.derivatives(solution.x)
            jacobian = _rescale_jacobian(solution.jac, intensity_scale, derivatives)
        else:
            physical_values = solution.x
            jacobian = np.asarray(solution.jac * intensity_scale, dtype=float)
        fitted, components, amplitudes = evaluate(physical_values)
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
        scaled.x = physical_values / scales
        try:
            scaled.jac = (
                _rescale_jacobian(solution.jac, scales, derivatives)
                if parameter_coordinates is not None
                else solution.jac * scales
            )
        except _JacobianRangeError:
            covariance, reason = None, "numeric_range"
        else:
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
        layout.free_names,
        layout.free_units,
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
            **(
                {"parameter_coordinates": deepcopy(coordinate_config)}
                if coordinate_config is not None
                else {}
            ),
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
    if parameter_coordinates is not None:
        fit["config"] = {"parameter_coordinates": coordinate_config}
        diagnostics.update(
            jacobian_coordinates="physical", covariance_coordinates="physical"
        )
    return fit, solution
