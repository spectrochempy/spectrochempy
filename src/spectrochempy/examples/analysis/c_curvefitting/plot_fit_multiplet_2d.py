# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: D205, D400, D415, T201
"""
Fit overlapping peaks in a 2D spectrum
======================================

Fit two separable peaks jointly, then compare the reconstructed spectrum
and residuals on the input grid.
"""

import spectrochempy as scp

# %%
# Generate a reproducible spectrum
# --------------------------------
x = scp.Coord.linspace(-4, 4, 61, units="Hz")
y = scp.Coord.linspace(-3, 3, 51, units="ppm")
grid = scp.zeros((y.size, x.size), coordset=[y, x], units="V")
true_peaks = [
    {
        "height": 2.0,
        "axes": {
            "y": {"model": "gaussian", "center": -0.4, "width": 1.1},
            "x": {"model": "lorentzian", "center": -0.5, "width": 1.4},
        },
    },
    {
        "height": 1.3,
        "axes": {
            "y": {"model": "lorentzian", "center": 0.6, "width": 1.4},
            "x": {"model": "gaussian", "center": 0.8, "width": 1.6},
        },
    },
]
clean = scp.simulate_peaks(grid, true_peaks)
noise = scp.normal(scale=0.015, size=grid.shape, seed=42, units=grid.units)
spectrum = clean + noise

# %%
# Simulate missing observations near the first peak
# ------------------------------------------------
# Mask a small patch over the center of the first peak (x ≈ −0.5 Hz,
# y ≈ −0.4 ppm) to mimic a gap in the measurements. The surrounding
# points still constrain the fit. The slice uses row and column indices.
spectrum[20:24, 25:28] = scp.MASKED

# %%
# Fit both peaks together
# -----------------------
# Each starting width has a positive lower bound. The points under the
# mask contribute neither to the fit nor to its residual statistics.
starting_peaks = [
    {
        "height": {"value": height, "bounds": (0, 4)},
        "axes": {
            dim: {
                "model": model,
                "center": center,
                "width": {"value": width, "bounds": (0.3, 3)},
            }
            for dim, model, center, width in profiles
        },
    }
    for height, profiles in [
        (1.7, [("y", "gaussian", -0.3, 1.3), ("x", "lorentzian", -0.3, 1.6)]),
        (1.1, [("y", "lorentzian", 0.5, 1.6), ("x", "gaussian", 0.6, 1.8)]),
    ]
]
result = scp.Optimize(peaks=starting_peaks).fit(spectrum).result
print(f"Converged: {result.diagnostics['success']}")
print(f"Residual RMSE: {result.diagnostics['rmse']:.4f} V")
for i, peak in enumerate(result.peaks, start=1):
    print(f"Peak {i} height: {peak['height']:.3f} V")

# %%
# Compare observations, fit and residuals
# ---------------------------------------
# The fitted model is evaluated across the whole grid. Its output keeps
# the mask so reconstructed values inside the gap remain distinguishable
# from fitted values at observed points. No residual is reported there.
for dataset, title in [
    (spectrum, "Observed"),
    (result.fitted, "Fitted"),
    (result.residuals, "Residuals"),
]:
    ax = dataset.plot_image(figsize=(5, 3.5), colorbar=True)
    _ = ax.set_title(title)
