# ruff: noqa: D205, D400, D415, T201
"""
Fit spectral components and recovery times jointly
==================================================

Use four Voigt components, independent recovery times and one shared mixing
parameter. See the global fitting guide for model conventions and callbacks.
"""

import spectrochempy as scp

x = scp.Coord.linspace(-8, 8, 101, units="Hz")
t = scp.Coord([0, 0.3, 0.8, 1.6, 3, 6, 12, 30], units="s")
grid = scp.zeros((t.size, x.size), coordset=[t, x], units="V")
peaks = [
    {
        "height": amplitude,
        "axes": {
            "x": {
                "model": "voigt",
                "parameters": {"center": center, "width": 1.5, "ratio": 0.7},
            },
            "y": {"model": "saturation_recovery", "parameters": {"tau": tau}},
        },
    }
    for amplitude, center, tau in [(1, -5, 1), (2, -2, 2), (1.5, 1, 4), (0.8, 4, 8)]
]
spectrum = scp.simulate_peaks(grid, peaks)
spectrum += scp.normal(scale=0.003, size=grid.shape, seed=42, units=grid.units)
for peak in peaks:
    peak["height"] = {"value": 1, "bounds": (0, 4)}
    peak["axes"]["x"]["parameters"]["width"] = {"value": 1.8, "bounds": (0.3, 4)}
    peak["axes"]["x"]["parameters"]["ratio"] = {"shared": "mixing"}
    tau = peak["axes"]["y"]["parameters"]["tau"]
    peak["axes"]["y"]["parameters"]["tau"] = {"value": tau * 1.2, "bounds": (0.1, 30)}
result = (
    scp.Optimize(
        peaks=peaks,
        shared_parameters={
            "mixing": {"value": 0.5, "bounds": (0, 1)},
        },
    )
    .fit(spectrum)
    .result
)
print("Converged:", result.diagnostics["success"])
_ = scp.plot_multiple(
    [spectrum[-1].squeeze(), result.fitted[-1].squeeze()],
    labels=["Observed", "Global fit"],
    method="pen",
    legend="best",
)
_ = result.residuals.plot(
    method="lines", colorbar=True, title="Residuals across all delays"
)

# %%
# Uncomment the following line to display all figures when running the script
# directly with Python.

# %%

# scp.show()
