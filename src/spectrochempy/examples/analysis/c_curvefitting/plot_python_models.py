# ruff: noqa: D205, D400, D415, T201
"""
Compose Python models with an explicitly shared width
=====================================================

Fit two Gaussian components and an additive linear background. Gaussian
amplitudes follow the existing sample-normalized profile, not peak heights.
The DSL remains available for the same canonical fitting infrastructure.
"""

import numpy as np

import spectrochempy as scp

x = scp.Coord.linspace(1560, 1680, 481, units="cm^-1")
observed = scp.gaussian(x, ampl=3, pos=1600, width=10)
observed += scp.gaussian(x, ampl=2, pos=1630, width=10)
observed += scp.NDDataset(0.00003 * x.data - 0.04, coordset=[x])
observed.title = "intensity"

model = (
    scp.Gaussian(prefix="p1_") + scp.Gaussian(prefix="p2_") + scp.Linear(prefix="bg_")
)
params = model.make_params(p1_center=1600, p2_center=1630)
params["p1_width"].set(value=12, min=1, max=30)
params["p2_width"].set(expr="p1_width")
params["p1_center"].set(vary=False)
params["p2_center"].set(vary=False)
params["p1_amplitude"].set(min=0, max=10)
params["p2_amplitude"].set(min=0, max=10)

opt = scp.Optimize(model=model, params=params, method="least_squares").fit(observed)
print(opt.report())
print("Initial width:", opt.initial_params["p1_width"].value)
print(
    "Fitted widths:",
    opt.result.fitted_parameters["p1_width"],
    opt.result.fitted_parameters["p2_width"],
)
_ = opt.plot()

# %%
# A personal function uses the same configuration and result objects.


def decay(t, amplitude, rate=0.5, offset=0.0):
    return amplitude * np.exp(-rate * t) + offset


time = scp.Coord.linspace(0, 5, 101)
signal = scp.NDDataset(decay(time.data, 2, 0.8, 0.1), coordset=[time])
personal = scp.Model(decay, prefix="custom_")
initial = personal.make_params(custom_amplitude=1, custom_offset=0.1)
initial["custom_rate"].set(min=0.001, max=5)
initial["custom_offset"].set(vary=False)
fit = scp.Optimize(model=personal, params=initial, method="least_squares").fit(signal)
print(fit.report())
_ = fit.plot()
