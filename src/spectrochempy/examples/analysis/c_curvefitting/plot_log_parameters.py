# ruff: noqa: D205, D400, D415, T201
"""
Positive parameters in logarithmic coordinates
==============================================

Choose an explicit positive reference for a rate fitted in log coordinates.
The rate and its bounds remain magnitudes in inverse seconds, since the
coordinate is expressed in seconds. The callable receives NumPy coordinate
magnitudes; the observations, masks and plotting use SpectroChemPy datasets.
This is a curve-fitting example, not a kinetic-mechanism engine.
"""

import numpy as np

import spectrochempy as scp


def decay(x, amplitude=2.0, rate=0.1, offset=0.0):
    return amplitude * np.exp(-rate * x) + offset


time = scp.linspace(
    0.0,
    5.0,
    81,
    coordset=[scp.Coord.linspace(0.0, 5.0, 81, units="s", rounding=False)],
)
observed = 2.0 * scp.exp(-0.7 * time) + 0.1 + 0.002 * scp.sin(3.0 * time)
observed.title = "Signal"
observed[1.5:1.7] = scp.MASKED
model = scp.Model(decay)
params = model.make_params()
params["amplitude"].set(min=0.1, max=5)
params["rate"].set(min=0.001, max=10, scale="log", log_reference=1.0)
params["offset"].set(min=-0.5, max=0.5)
fit = scp.Optimize(
    model=model, params=params, bounds_mode="native", method="least_squares"
).fit(observed)
print(fit.result.fitted_parameters)
print(fit.result.parameter_names)
print(fit.result.stderr)
print(fit.result.parameters["parameter_coordinates"]["rate"])
print(fit.report())
print("Initial rate:", params["rate"].value)
print("Retained reference:", fit.initial_params["rate"].log_reference)
_ = fit.plot()
