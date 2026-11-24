"""Axis profiles shared by separable simulation and global fitting."""

from collections.abc import Mapping
from collections.abc import Sequence

import numpy as np
from pint import Quantity as PintQuantity
from scipy.special import voigt_profile

from spectrochempy.core.units import Quantity
from spectrochempy.core.units import ur

_BUILTINS = (
    "gaussian",
    "lorentzian",
    "voigt",
    "pseudovoigt",
    "saturation_recovery",
    "exponential_decay",
)


def extended_profiles(peaks):
    """Identify inputs requiring the general profile evaluator."""
    if not isinstance(peaks, Sequence) or isinstance(peaks, (str, bytes)):
        return False
    return any(
        isinstance(peak, Mapping)
        and isinstance(peak.get("axes"), Mapping)
        and any(
            isinstance(axis, Mapping)
            and (
                "parameters" in axis
                or axis.get("model") not in ("gaussian", "lorentzian")
            )
            for axis in peak["axes"].values()
        )
        for peak in peaks
    )


def axis_parameters(axis, unit, dim):
    """Describe profile arguments and their physical units."""
    if not isinstance(axis, Mapping) or "model" not in axis:
        raise ValueError(f"Axis {dim!r} requires a model")
    model = axis["model"]
    custom = callable(model)
    if not custom and (not isinstance(model, str) or model not in _BUILTINS):
        raise ValueError(f"Unsupported model on axis {dim!r}: {model!r}")
    if "parameters" in axis:
        if set(axis) != {"model", "parameters"}:
            raise ValueError(
                f"Axis {dim!r}: parameters cannot be mixed with flat arguments"
            )
        parameters = axis["parameters"]
    else:
        parameters = {key: value for key, value in axis.items() if key != "model"}
    if not isinstance(parameters, Mapping):
        raise ValueError(f"Axis {dim!r} parameters must be a mapping")
    if custom:
        expected = set(parameters)
    elif model in ("saturation_recovery", "exponential_decay"):
        expected = {"tau"}
    else:
        expected = {"center", "width"}
        if model in ("voigt", "pseudovoigt"):
            expected.add("ratio")
    if set(parameters) != expected:
        raise ValueError(f"Axis {dim!r}: expected parameters {sorted(expected)}")
    entries = []
    for key, spec in parameters.items():
        if not isinstance(key, str) or not key.isidentifier():
            raise ValueError("Profile parameter names must be Python identifiers")
        interval = not custom and key in ("width", "tau")
        positive = interval
        target_unit = (
            (ur.dimensionless if key == "ratio" else unit) if not custom else None
        )
        if custom:
            declared = spec.get("unit") if isinstance(spec, Mapping) else None
            value = spec.get("value") if isinstance(spec, Mapping) else spec
            reference = isinstance(spec, Mapping) and "shared" in spec
            target_unit = declared or (
                value.units
                if isinstance(value, PintQuantity)
                else None
                if reference
                else ur.dimensionless
            )
        elif isinstance(spec, Mapping) and "unit" in spec:
            raise ValueError("Built-in profile parameters use the coordinate units")
        target_unit = ur.Unit(target_unit) if target_unit is not None else None
        entries.append((key, spec, target_unit, positive, interval))
    return model, entries


def profile_values(values, axis):
    """Evaluate an axis without imposing a normalization on user callbacks."""
    model, parameters = axis["model"], axis["parameters"]
    if callable(model):
        result = model(np.array(values, copy=True), **parameters)
    elif model in ("gaussian", "lorentzian"):
        from spectrochempy.analysis.curvefitting.simulation import (  # noqa: PLC0415
            _evaluate_separable_peak,
        )

        result = _evaluate_separable_peak(
            (values.size,),
            {"x": values},
            {"height": 1.0, "axes": {"x": {"model": model, **parameters}}},
        )
    elif model in ("voigt", "pseudovoigt"):
        if not 0 <= parameters["ratio"] <= 1:
            raise ValueError("Voigt ratio must lie between zero and one")
        width, ratio = parameters["width"], parameters["ratio"]
        offsets = np.asarray(
            values.astype(np.longdouble) - parameters["center"], dtype=float
        )
        if model == "voigt":
            sigma = ratio * width / np.sqrt(8 * np.log(2))
            gamma = (1 - ratio) * width / 2
            result = voigt_profile(offsets, sigma, gamma) / voigt_profile(
                0, sigma, gamma
            )
        else:
            result = ratio * np.exp(-4 * np.log(2) * (offsets / width) ** 2)
            result += (1 - ratio) / (1 + 4 * (offsets / width) ** 2)
    else:
        tau = parameters["tau"]
        if tau <= 0:
            raise ValueError("Recovery and decay tau must be positive")
        ratio = np.asarray(values, dtype=np.longdouble) / np.longdouble(tau)
        result = -np.expm1(-ratio) if model == "saturation_recovery" else np.exp(-ratio)
    if hasattr(result, "units") and result.units is not None:
        if not result.units.dimensionless:
            raise ValueError("Axis profiles must return dimensionless values")
        result = result.to(ur.dimensionless)
    if hasattr(result, "mask") and np.any(result.mask):
        raise ValueError("Axis profiles must not return masked values")
    result = getattr(result, "magnitude", getattr(result, "data", result))
    result = np.asarray(result)
    if result.shape != values.shape or result.dtype.kind not in "iuf":
        raise ValueError(
            "Axis profiles must return a real vector matching the coordinate"
        )
    return np.asarray(result, dtype=float)


def evaluate_profiles(shape, coordinates, peak):
    """Form a separable component on the native coordinate grid."""
    from spectrochempy.analysis.curvefitting.simulation import (  # noqa: PLC0415
        _profile_factors,
    )
    from spectrochempy.analysis.curvefitting.simulation import (  # noqa: PLC0415
        _scaled_offsets,
    )

    height, height_exponent = np.frexp(float(peak["height"]))
    mantissa = np.full(shape, height, dtype=float)
    exponent = np.full(shape, height_exponent, dtype=np.int64)
    for index, (dim, values) in enumerate(coordinates.items()):
        axis = peak["axes"][dim]
        vector = profile_values(values, axis)
        factor, power = np.frexp(vector)
        model, parameters = axis["model"], axis["parameters"]
        if model in ("gaussian", "lorentzian"):
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                offsets, displacement, overflow = _scaled_offsets(
                    values, parameters["center"], parameters["width"]
                )
            with np.errstate(
                over="ignore", under="ignore", invalid="ignore", divide="ignore"
            ):
                factor, power = _profile_factors(
                    model, vector, offsets, displacement, overflow, parameters["width"]
                )
            if model == "gaussian":
                tiny = vector < np.finfo(float).tiny
                with np.errstate(over="ignore", invalid="ignore"):
                    logarithm = np.maximum(
                        -4 * offsets[tiny] ** 2, -1024 * (len(shape) + 2)
                    )
                power[tiny] = np.floor(logarithm).astype(int)
                factor[tiny] = np.exp2(logarithm - power[tiny])
        broadcast = [1] * len(shape)
        broadcast[index] = values.size
        mantissa, shift = np.frexp(mantissa * factor.reshape(broadcast))
        exponent += power.reshape(broadcast) + shift
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        return np.ldexp(mantissa, exponent)


def resolved_simulation_peaks(peaks, units, intensity_unit):
    """Resolve numerical profile parameters for simulation."""
    from spectrochempy.analysis.curvefitting.simulation import (  # noqa: PLC0415
        _parameter_value,
    )

    result = []
    for peak in peaks:
        if not isinstance(peak, Mapping) or set(peak) != {"height", "axes"}:
            raise ValueError("Each peak requires height and axes")
        if not isinstance(peak["axes"], Mapping) or set(peak["axes"]) != set(units):
            raise ValueError("Each peak must specify every template dimension")
        resolved = {
            "height": _parameter_value(peak["height"], intensity_unit, "height"),
            "axes": {},
        }
        for dim, unit in units.items():
            model, entries = axis_parameters(peak["axes"][dim], unit, dim)
            parameters = {}
            for key, value, target_unit, positive, interval in entries:
                if isinstance(value, Mapping):
                    if set(value) - {"value", "unit"} or "value" not in value:
                        raise ValueError(
                            "Simulation requires resolved parameter values"
                        )
                    value = value["value"]
                value = _parameter_value(value, target_unit, key, interval=interval)
                if positive and value <= 0:
                    raise ValueError(f"{dim} {key} must be strictly positive")
                parameters[key] = value
            resolved["axes"][dim] = {"model": model, "parameters": parameters}
        result.append(resolved)
    return result


def public_profiles(peaks, units_by_target):
    """Keep callable parameter units when exporting an evaluable model."""
    result = []
    for peak in peaks:
        axes = {}
        for dim, axis in peak["axes"].items():
            parameters = {}
            for key, value in axis["parameters"].items():
                unit = units_by_target[(id(axis["parameters"]), key)]
                parameters[key] = (
                    Quantity(value, unit)
                    if callable(axis["model"]) and unit is not None
                    else value
                )
            axes[dim] = {"model": axis["model"], "parameters": parameters}
        result.append({"height": peak["height"], "axes": axes})
    return result


def profile_history(value):
    """Record callable identities and unit-bearing values as plain metadata."""
    if callable(value):
        return f"{getattr(value, '__module__', type(value).__module__)}.{getattr(value, '__qualname__', type(value).__qualname__)}"
    if isinstance(value, PintQuantity):
        return {"value": profile_history(value.magnitude), "unit": str(value.units)}
    if isinstance(value, Mapping):
        return {key: profile_history(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [profile_history(item) for item in value]
    if isinstance(value, np.longdouble):
        return str(value)
    return value
