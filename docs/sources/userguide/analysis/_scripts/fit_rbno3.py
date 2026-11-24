"""Fit one public RbNO3 MQMAS feature with a tilted 2D Gaussian."""

from cmath import phase
from math import degrees

import numpy as np

import spectrochempy as scp

path = scp.preferences.datadir / "nmrdata/bruker/tests/nmr/RbNO3/206"
raw = scp.nmr.read(path, phase_cycle="paired")
states = scp.nmr.Experiment(raw).select_coherence(3)
xf2 = scp.nmr.Experiment(states).xf2(size=4096)
row = xf2[0, 0].squeeze()
phc0 = -degrees(phase(row.data[abs(row).argmax()]))
phased = scp.nmr.Experiment(xf2).xf2p(phc0=phc0)
spectrum = scp.nmr.Experiment(phased).xfshear(size=128)
observed = scp.nmr.Experiment(spectrum).RR[-19.0:-25.0, -22.0:-34.0]
observed = observed / abs(observed).max()


def tilted_peak(template, f1, f2, width1, width2, tilt):
    """Return a full-grid response; widths are standard deviations in ppm."""
    y = template.y.data[:, None] - f1
    x = template.x.data[None, :] - f2 - tilt * y
    return {"peak": np.exp(-0.5 * ((y / width1) ** 2 + (x / width2) ** 2))}


result = (
    scp.Optimize(
        model=tilted_peak,
        parameters={
            "f1": {"value": -22.3, "bounds": (-24, -20), "unit": "ppm"},
            "f2": {"value": -27.6, "bounds": (-31, -25), "unit": "ppm"},
            "width1": {"value": 0.6, "bounds": (0.1, 2), "unit": "ppm"},
            "width2": {"value": 1.2, "bounds": (0.2, 4), "unit": "ppm"},
            "tilt": {"value": 0.0, "bounds": (-3, 3)},
        },
        linear_amplitudes=True,
        max_fun_calls=200,
    )
    .fit(observed)
    .result
)

print(f"Converged: {result.diagnostics['success']}")
print(f"RMS / observed peak: {result.diagnostics['rmse']:.2%}")
for name, value in result.fitted_parameters.items():
    print(f"{name}: {value:.3f}")
print(f"Peak amplitude: {result.amplitudes['peak']:.3f}")
print(f"Covariance: {result.diagnostics['covariance_reason']}")

_ = scp.multiplot(
    [observed, result.fitted, result.residuals],
    ncol=3,
    labels=["Measured RbNO3", "Tilted Gaussian", "Residual"],
    method="contour",
    levels=[0.05, 0.1, 0.2, 0.4, 0.6, 0.8],
    vmin=-1,
    vmax=1,
    positive_linestyle="-",
    negative_linestyle="--",
    colorbar=False,
    equal_aspect=False,
    figsize=(12, 3.5),
)
