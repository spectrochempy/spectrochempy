"""Named one-dimensional evolution laws using the native profile fitter."""

from collections.abc import Mapping

import numpy as np

from spectrochempy.analysis.curvefitting._optimize_nd import _grid
from spectrochempy.analysis.curvefitting._optimize_profiles import fit_profiles
from spectrochempy.analysis.curvefitting._profile_models import profile_history
from spectrochempy.analysis.curvefitting.simulation import _interval_unit
from spectrochempy.analysis.curvefitting.simulation import _parameter_value
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.units import Quantity

CURVE_MODELS = ("saturation_recovery", "exponential_decay")


def fit_curve(dataset, model, parameters, max_calls):
    if not isinstance(dataset, NDDataset) or dataset.ndim != 1:
        raise ValueError("Named recovery and decay models require a 1D NDDataset")
    coordinates, units = _grid(dataset, min_ndim=1)
    dim = dataset.dims[0]
    if np.any(coordinates[dim] < 0):
        raise ValueError("Recovery and decay coordinates must be nonnegative")
    if not isinstance(parameters, Mapping) or set(parameters) != {"amplitude", "tau"}:
        raise ValueError("Named curve parameters must contain amplitude and tau")
    tau = parameters["tau"]
    tau = dict(tau) if isinstance(tau, Mapping) else {"value": tau}
    if "bounds" not in tau:
        initial = _parameter_value(tau.get("value"), units[dim], "tau", interval=True)
        lower = max(np.nextafter(0.0, 1.0), abs(initial) * np.finfo(float).tiny)
        tau["bounds"] = (lower, None)
    peaks = [
        {
            "height": parameters["amplitude"],
            "axes": {dim: {"model": model, "parameters": {"tau": tau}}},
        }
    ]

    def derivatives(resolved, names):
        amplitude = np.longdouble(resolved[0]["height"])
        time_constant = np.longdouble(resolved[0]["axes"][dim]["parameters"]["tau"])
        ratio = np.asarray(coordinates[dim], dtype=np.longdouble) / time_constant
        decay = np.exp(-ratio)
        recovery = model == "saturation_recovery"
        profile = -np.expm1(-ratio) if recovery else decay
        tau_column = amplitude * (ratio * decay) / time_constant
        if not recovery:
            tau_column = -tau_column
        return np.column_stack(
            [-profile if name.endswith(".height") else tau_column for name in names]
        )

    fit, solution = fit_profiles(
        dataset,
        peaks,
        max_calls,
        None,
        min_ndim=1,
        residual_jacobian=derivatives,
    )
    fitted = fit["peaks"][0]
    values = {
        "amplitude": (fitted["height"], dataset.units),
        "tau": (
            fitted["axes"][dim]["parameters"]["tau"],
            _interval_unit(units[dim]) if units[dim] is not None else None,
        ),
    }
    fit["fitted_parameters"] = {
        key: value if unit is None else Quantity(value, unit)
        for key, (value, unit) in values.items()
    }
    fit["names"] = tuple(
        "amplitude" if name.endswith(".height") else "tau" for name in fit["names"]
    )
    fit["amplitudes"] = {}
    fit["component_names"] = (model,)
    for output in (
        fit["observed"],
        fit["fitted"],
        fit["residuals"],
        *fit["components"],
    ):
        output._append_history_entry(
            operation="optimize",
            parameters={
                "mode": "named_curve",
                "model": model,
                "dimension": dim,
                "parameter_order": list(fit["names"]),
                "fitted_parameters": profile_history(fit["fitted_parameters"]),
            },
            message=f"Fit {model} on the native 1D coordinate",
        )
    return fit, solution
