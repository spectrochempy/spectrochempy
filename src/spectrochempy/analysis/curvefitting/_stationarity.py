"""Numerical qualification of first-order stationarity in physical coordinates."""

import numpy as np


def _bound_neighborhood(values, lower, upper):
    """Identify the same conservative physical neighborhoods as 1D covariance."""
    both = np.isfinite(lower) & np.isfinite(upper)
    scale = np.maximum(1.0, np.abs(values))
    scale = np.maximum(scale, np.where(np.isfinite(lower), np.abs(lower), 0.0))
    scale = np.maximum(scale, np.where(np.isfinite(upper), np.abs(upper), 0.0))
    scale[both] = upper[both] - lower[both]
    neighborhood = np.sqrt(np.finfo(float).eps) * scale
    return values - lower <= neighborhood, upper - values <= neighborhood


def _physical_column(residual, values, index, lower, upper, current, tolerance):
    """Qualify a feasible three-point residual derivative at two step sizes."""
    value = values[index]
    if np.isfinite(lower) and np.isfinite(upper):
        scale = upper - lower
    else:
        scale = max(1.0, abs(value))
    step = np.cbrt(np.finfo(float).eps) * scale
    if value - step >= lower and value + step <= upper:
        offsets = (-step, step)
    elif upper - value >= value - lower:
        step = min(step, (upper - value) / 2)
        offsets = (step, 2 * step)
    else:
        step = min(step, (value - lower) / 2)
        offsets = (-step, -2 * step)
    cache = {}

    def estimate(factor):
        probes, deltas = [], []
        for offset in offsets:
            trial = values.copy()
            trial[index] = value + offset * factor
            delta = trial[index] - value
            if delta == 0 or not lower <= trial[index] <= upper:
                raise ValueError("No representable feasible diagnostic step")
            if trial[index] not in cache:
                probe = np.asarray(residual(trial), dtype=float).reshape(-1)
                if probe.shape != current.shape or not np.isfinite(probe).all():
                    raise ValueError("Invalid diagnostic residual")
                cache[trial[index]] = probe
            probes.append(cache[trial[index]])
            deltas.append(delta)
        first, second = deltas
        if first == second:
            raise ValueError("Diagnostic points coincide")
        weight_first = second / (second - first)
        weight_second = first / (second - first)
        derivative = (probes[0] - current) / first * weight_first - (
            probes[1] - current
        ) / second * weight_second
        roundoff = np.finfo(float).eps * (
            (np.abs(probes[0]) + np.abs(current)) / abs(first) * abs(weight_first)
            + (np.abs(probes[1]) + np.abs(current)) / abs(second) * abs(weight_second)
        )
        return derivative, roundoff

    coarse, _ = estimate(1.0)
    fine, roundoff = estimate(0.5)
    scale = np.max(np.abs(fine))
    if scale == 0:
        raise ValueError("Unresolved physical derivative")
    norm = np.linalg.norm(fine / scale)
    relative_error = (
        np.linalg.norm((coarse - fine) / scale) + np.linalg.norm(roundoff / scale)
    ) / norm
    if not np.isfinite(relative_error) or relative_error >= 1:
        raise ValueError("Unresolved physical derivative")
    cosine_error = 2 * relative_error / (1 - relative_error)
    if cosine_error > tolerance / 4:
        raise ValueError("Unresolved physical derivative at stationarity tolerance")
    return fine, cosine_error


def _physical_stationarity(
    residual,
    values,
    bounds,
    current,
    jacobian,
    derivatives,
    *,
    reference=None,
    tolerance=1e-6,
):
    """
    Qualify a dimensionless projected residual gradient, without changing a fit.

    Regular columns use the retained Jacobian and signed transform derivative.
    Near bounds, independent feasible physical probes recover information lost
    in solver coordinates. The maximum projected residual/column cosine is a
    local numerical diagnostic, not a global minimum or identifiability test.
    """
    diagnostics = {
        "physical_stationarity_status": "unavailable",
        "physical_stationarity_reason": "missing_jacobian",
        "physical_stationarity_tolerance": tolerance,
        "physical_projected_gradient": None,
        "stationarity_evaluations": 0,
    }
    if jacobian is None:
        return diagnostics
    evaluations = 0

    def evaluate(trial):
        nonlocal evaluations
        evaluations += 1
        return residual(trial)

    try:
        values = np.asarray(values, dtype=float)
        lower, upper = np.asarray(bounds, dtype=float).T
        current = np.asarray(current, dtype=float).reshape(-1)
        derivatives = np.asarray(derivatives, dtype=float)
        jacobian = np.asarray(jacobian, dtype=float)
        if (
            values.size == 0
            or current.size == 0
            or jacobian.shape != (current.size, values.size)
            or derivatives.shape != values.shape
        ):
            raise ValueError("Invalid diagnostic shape")
        if not (
            np.isfinite(values).all()
            and np.isfinite(current).all()
            and np.isfinite(jacobian).all()
            and np.isfinite(derivatives).all()
        ):
            raise ValueError("Nonfinite diagnostic inputs")
        if np.isnan(lower).any() or np.isnan(upper).any() or np.any(lower >= upper):
            raise ValueError("Invalid physical bounds")
        if np.any(values < lower) or np.any(values > upper):
            raise ValueError("Physical values outside bounds")
        near_lower, near_upper = _bound_neighborhood(values, lower, upper)
        on_lower, on_upper = values == lower, values == upper
        residual_scale = np.max(np.abs(current))
        reference_scale = 0.0
        if reference is not None:
            reference_values = np.ma.masked_invalid(
                np.ma.asarray(reference, dtype=float)
            ).compressed()
            if reference_values.size:
                reference_scale = np.max(np.abs(reference_values))
        at_roundoff = residual_scale <= 32 * np.finfo(float).eps * reference_scale
        if residual_scale == 0 or at_roundoff:
            projected = 0.0
        else:
            normalized_residual = current / residual_scale
            normalized_residual /= np.linalg.norm(normalized_residual)
            projected = 0.0
            score_error = 0.0
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                for index in range(values.size):
                    if (
                        near_lower[index]
                        or near_upper[index]
                        or derivatives[index] == 0
                    ):
                        column, column_error = _physical_column(
                            evaluate,
                            values,
                            index,
                            lower[index],
                            upper[index],
                            current,
                            tolerance,
                        )
                        score_error = max(score_error, column_error)
                    else:
                        column = jacobian[:, index] / derivatives[index]
                    if not np.isfinite(column).all():
                        raise ValueError("Nonfinite physical Jacobian")
                    scale = np.max(np.abs(column))
                    if scale == 0:
                        continue
                    column = column / scale
                    cosine = float(column @ normalized_residual) / np.linalg.norm(
                        column
                    )
                    if (on_lower[index] and cosine >= 0) or (
                        on_upper[index] and cosine <= 0
                    ):
                        continue
                    projected = max(projected, abs(cosine))
            if projected - score_error <= tolerance < projected + score_error:
                raise ValueError("Stationarity threshold unresolved by physical probes")
        diagnostics.update(
            physical_stationarity_status=(
                "satisfied" if projected <= tolerance else "not_satisfied"
            ),
            physical_stationarity_reason="residual_at_roundoff"
            if at_roundoff
            else None,
            physical_projected_gradient=projected,
        )
    except Exception as error:
        diagnostics["physical_stationarity_reason"] = f"{type(error).__name__}: {error}"
    diagnostics["stationarity_evaluations"] = evaluations
    return diagnostics
