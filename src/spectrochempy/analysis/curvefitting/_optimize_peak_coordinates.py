"""Explicit private coordinate wiring for the historical analytic peak route."""

from copy import copy

import numpy as np

from spectrochempy.analysis.curvefitting import _optimize_nd as backend
from spectrochempy.analysis.curvefitting._coordinate_context import (
    _bind_centered_coordinates,
)
from spectrochempy.analysis.curvefitting._coordinate_context import _coordinate_config
from spectrochempy.analysis.curvefitting._coordinate_context import (
    _scaled_physical_jacobian,
)
from spectrochempy.analysis.curvefitting._coordinate_context import (
    _validate_center_anchor_offsets,
)
from spectrochempy.analysis.curvefitting._profile_models import profile_history
from spectrochempy.analysis.curvefitting.simulation import _evaluate_separable_peak


def fit_peak_coordinates(dataset, specs, max_calls, parameter_coordinates):
    """
    Fit standard 2D/3D peaks with copied, preconverted coordinate declarations.

    Free centers require explicit linear origins retaining initial and finite
    bound offsets to eight float64 epsilons relative to the initial axis width.
    Other free parameters must
    remain uncentered. Log references are positive, preconverted magnitudes.
    The internal analytic Jacobian differentiates virtual centered motion.
    Nonfinite solver Jacobians raise rather than selecting an unqualified
    numerical fallback. Public estimators do not select this private route.

    Final derivatives and inference use the physically represented point,
    in the historical centered/scaled physical basis. Raw results retain
    solver coordinates. Existing TRF stopping rules and initial values are
    preserved; solver success does not establish physical stationarity.
    """
    specs = backend.snapshot_peaks(specs)
    coordinates, units = backend._grid(dataset)
    if dataset.ndim not in (2, 3):
        raise ValueError(
            "Historical peak coordinates require the qualified analytic 2D/3D route"
        )
    peaks, free, layout = backend._prepare_peak_layout(specs, dataset.units, units)
    space = _bind_centered_coordinates(
        layout, parameter_coordinates, kind="historical peak"
    )
    start = np.asarray(space.initial, dtype=float)
    for index, ((target, key), origin, transform) in enumerate(
        zip(free, space.origins, space.transforms, strict=True)
    ):
        if key == "center":
            if origin is None or transform != "lin":
                raise ValueError(
                    "Free historical centers require explicit linear origins"
                )
            _validate_center_anchor_offsets(
                space, index, target["width"], kind="Historical"
            )
        elif origin is not None:
            raise ValueError(
                "Only free historical centers support explicit origins here"
            )
    with np.errstate(over="ignore"):
        data = np.asarray(dataset.data, dtype=float)
    unmasked = ~np.asarray(dataset.mask)
    if np.any(np.isfinite(dataset.data) & ~np.isfinite(data) & unmasked):
        raise ValueError(
            "Observed intensity is outside the solver's floating-point range"
        )
    valid = np.isfinite(data) & unmasked
    if not np.count_nonzero(valid) or np.count_nonzero(valid) < len(free):
        raise ValueError("ND fitting requires observations for all free parameters")
    observed = data[valid]
    intensity = max(np.max(np.abs(observed)), *(abs(p["height"]) for p in peaks)) or 1.0
    initial = np.asarray(layout.free_values, dtype=np.longdouble)
    origins = np.asarray(
        [
            value if key == "center" else 0
            for (_, key), value in zip(free, initial, strict=True)
        ],
        dtype=np.longdouble,
    )
    scales = np.asarray(
        [
            (abs(target[key]) or intensity) if key == "height" else target["width"]
            for target, key in free
        ],
        dtype=np.longdouble,
    )
    limits = np.asarray(layout.free_bounds, dtype=np.longdouble).reshape(-1, 2).T
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        relative_limits = limits - origins
        inference_bounds = np.asarray(relative_limits / scales, dtype=float)

    def model(peak_list, grids, *, keep_components=False):
        total = np.zeros(dataset.shape)
        components = []
        with np.errstate(over="ignore", invalid="ignore"):
            for peak, grid in zip(peak_list, grids, strict=True):
                part = _evaluate_separable_peak(dataset.shape, grid, peak)
                total += part
                if keep_components:
                    components.append(part)
        return total, components

    solution, jacobian = None, None
    values = []
    numeric_range = False
    if free:
        center_origins = {
            id(target): target[key] for target, key in free if key == "center"
        }
        solver_peaks, grids, targets = backend._solver_model(
            peaks, coordinates, center_origins
        )
        bindings = [(targets[id(target)], key) for target, key in free]
        initial_offsets = space.to_offsets(start)
        support = np.nonzero(valid)
        columns = {
            (id(target), key): (index, np.log(factor))
            for index, ((target, factor), key) in enumerate(bindings)
        }

        def update(vector):
            offsets = space.to_offsets(vector)
            for i, (((target, factor), key), value) in enumerate(
                zip(bindings, offsets, strict=True)
            ):
                if key == "center":
                    value -= initial_offsets[i]
                target[key] = float(value) if key == "height" else value / factor

        def residuals(vector):
            update(vector)
            fitted, _ = model(solver_peaks, grids)
            with np.errstate(over="ignore", invalid="ignore"):
                residual = (observed - fitted[valid]) / intensity
            if not np.isfinite(residual).all():
                raise ValueError(
                    "ND fitting produced non-finite residuals on the support"
                )
            return residual

        def solver_jacobian(vector):
            update(vector)
            matrix = backend._separable_jacobian(
                list(zip(solver_peaks, grids, strict=True)),
                columns,
                support,
                np.log(space.derivatives(vector)) - np.log(np.longdouble(intensity)),
            )
            if not np.isfinite(matrix).all():
                raise ValueError(
                    "Historical coordinate fitting requires a finite analytic Jacobian; numerical fallback is not qualified"
                )
            return matrix

        solution = backend.least_squares(
            residuals,
            start,
            bounds=np.asarray(space.bounds, dtype=float).T,
            method="trf",
            jac=solver_jacobian,
            x_scale=1.0,
            gtol=None,
            max_nfev=max_calls,
        )
        encoded = space.to_solver(space.to_physical(solution.x))
        represented = solution.x.copy()
        for i, (_, key) in enumerate(free):
            if key == "center":
                represented[i] = encoded[i]
        precision = 8 * np.finfo(float).eps
        if not np.allclose(represented, solution.x, rtol=precision, atol=precision):
            solution.x = represented
        values = space.to_physical(solution.x)
        for (target, key), value in zip(free, values, strict=True):
            target[key] = float(value) if key == "height" else value
        jacobian = backend._separable_jacobian(
            [(p, coordinates) for p in peaks],
            {(id(target), key): (i, 0.0) for i, (target, key) in enumerate(free)},
            support,
            np.zeros(len(free)),
        )

    fitted, components = model(peaks, [coordinates] * len(peaks), keep_components=True)
    residual = np.zeros(dataset.shape)
    residual[valid] = observed - fitted[valid]
    if not np.isfinite(residual[valid]).all():
        raise ValueError("ND fitting produced non-finite residuals on the support")
    diagnostics = backend._fit_diagnostics(observed, residual[valid], len(free))
    inference = solution
    if solution is not None:
        solution.fun = residual[valid] / intensity
        inference = copy(solution)
        displacement = values - origins
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            inference.x = np.asarray(displacement / scales, dtype=float)
        inference.jac = _scaled_physical_jacobian(jacobian, scales, intensity)
        numeric_range = (
            np.any(~np.isfinite(inference.x))
            or np.any((displacement != 0) & (inference.x == 0))
            or np.any(np.isfinite(relative_limits) & ~np.isfinite(inference_bounds))
            or np.any((relative_limits != 0) & (inference_bounds == 0))
            or not np.isfinite(jacobian).all()
            or not np.isfinite(inference.jac).all()
            or np.any((jacobian != 0) & (inference.jac == 0))
        )
        solution.jac = _scaled_physical_jacobian(
            jacobian, space.derivatives(solution.x), intensity
        )
    if numeric_range:
        covariance, reason = None, "numeric_range"
    else:
        covariance, reason = backend._physical_covariance(
            inference, jacobian, scales, inference_bounds, diagnostics
        )
    diagnostics.update(
        covariance_reason=reason,
        jacobian_coordinates="physical",
        covariance_coordinates="physical",
    )
    config = _coordinate_config(space)
    fit = backend._fit_outputs(
        dataset,
        data,
        valid,
        peaks,
        fitted,
        components,
        residual,
        values,
        layout.free_names,
        layout.free_units,
        jacobian,
        diagnostics,
        covariance,
        history_parameters={
            "peak_specification": profile_history(specs),
            "parameter_coordinates": profile_history(config),
        },
    )
    fit["config"] = {"parameter_coordinates": config}
    return fit, solution
