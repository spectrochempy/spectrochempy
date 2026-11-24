"""Fit a synthetic COSY-style spectrum with shared doublet parameters."""

import spectrochempy as scp


def doublet(axis, center, coupling, width):
    """Return in-phase and antiphase Gaussian doublets (FWHM in Hz)."""
    left = scp.gaussian(
        axis, pos=center - coupling / 2, width=width, normalized=False
    ).data
    right = scp.gaussian(
        axis, pos=center + coupling / 2, width=width, normalized=False
    ).data
    return left + right, left - right


def cosy(template, frequency_a, frequency_b, coupling, width):
    """Tie the two axes, both cross peaks and all doublet splittings."""
    ax, adx = doublet(template.x, frequency_a, coupling, width)
    ay, ady = doublet(template.y, frequency_a, coupling, width)
    bx, bdx = doublet(template.x, frequency_b, coupling, width)
    by, bdy = doublet(template.y, frequency_b, coupling, width)
    return {
        "diagonal_a": ay[:, None] * ax[None, :],
        "diagonal_b": by[:, None] * bx[None, :],
        "cross_pair": ady[:, None] * bdx[None, :] + bdy[:, None] * adx[None, :],
    }


f1 = scp.Coord.linspace(0, 100, 121, units="Hz", title="F1 frequency", rounding=False)
f2 = scp.Coord.linspace(0, 100, 121, units="Hz", title="F2 frequency", rounding=False)
observed = scp.zeros((f1.size, f2.size), coordset=[f1, f2])
truth = {"frequency_a": 30.0, "frequency_b": 70.0, "coupling": 8.0, "width": 3.0}
responses = cosy(observed, **truth)
observed.data = (
    responses["diagonal_a"]
    + 0.8 * responses["diagonal_b"]
    + 0.35 * responses["cross_pair"]
)
observed += scp.normal(
    scale=0.005, size=observed.shape, seed=42, coordset=observed.coordset
)
result = (
    scp.Optimize(
        model=cosy,
        parameters={
            "frequency_a": {"value": 28, "bounds": (25, 35), "unit": "Hz"},
            "frequency_b": {"value": 72, "bounds": (65, 75), "unit": "Hz"},
            "coupling": {"value": 7, "bounds": (5, 11), "unit": "Hz"},
            "width": {"value": 4, "bounds": (2, 6), "unit": "Hz"},
        },
        linear_amplitudes=True,
        max_fun_calls=100,
    )
    .fit(observed)
    .result
)

print(f"Converged: {result.diagnostics['success']}")
print(f"Residual RMS: {result.diagnostics['rmse']:.4f} (noise SD: 0.005)")
for name, value in result.fitted_parameters.items():
    print(f"{name}: {value:.3f}; injected: {truth[name]:.3f} Hz")
for name, value in result.amplitudes.items():
    print(f"{name}: {value:.3f}")
print(f"Covariance: {result.diagnostics['covariance_reason']}")

_ = scp.multiplot(
    [observed, result.fitted, result.residuals],
    ncol=3,
    labels=["Synthetic COSY-style spectrum", "Optimize fit", "Residual"],
    method="contour",
    levels=[-0.8, -0.4, -0.2, -0.1, -0.05, -0.02, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8],
    vmin=-1,
    vmax=1,
    positive_color="#b2182b",
    negative_color="#2166ac",
    positive_linestyle="-",
    negative_linestyle="--",
    colorbar=False,
    equal_aspect=False,
    figsize=(12, 3.5),
)
