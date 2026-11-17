# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: D205, D400, D415
"""
Fit overlapping peaks in a 2D spectrum
======================================

Fit two peaks jointly and compare the data, reconstruction, and residuals.
See the :doc:`fitting tutorial </userguide/analysis/fitting>` for
parameter bounds and fitting when parts of the map are missing.
"""

import spectrochempy as scp


def peak(height, x_center, y_center):
    return {
        "height": height,
        "axes": {
            "x": {"model": "gaussian", "center": x_center, "width": 1.5},
            "y": {"model": "gaussian", "center": y_center, "width": 1.2},
        },
    }


x = scp.Coord.linspace(-4, 4, 61, units="Hz")
y = scp.Coord.linspace(-3, 3, 51, units="ppm")
grid = scp.zeros((y.size, x.size), coordset=[y, x], units="V")
signal = scp.simulate_peaks(grid, [peak(2.0, -0.5, -0.4), peak(1.3, 0.5, 0.4)])
noise = scp.normal(scale=0.015, size=grid.shape, seed=42, units=grid.units)
spectrum = signal + noise

starting_peaks = [peak(1.7, -0.3, -0.2), peak(1.1, 0.7, 0.5)]
for guess in starting_peaks:
    for profile in guess["axes"].values():
        profile["width"] = {"value": profile["width"], "bounds": (0.3, 3.0)}
result = scp.Optimize(peaks=starting_peaks).fit(spectrum).result

_ = scp.multiplot(
    [spectrum, result.fitted, result.residuals],
    labels=["Observed", "Fitted", "Residuals"],
    ncol=3,
    method="image",
    colorbar=True,
    figsize=(12, 4),
)
