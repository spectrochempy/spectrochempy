# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: D205, D400, D415, T201
"""
Fit a synthetic COSY-style 2D spectrum
======================================

Fit two diagonal features and a symmetric pair of signed cross peaks with
shared resonance frequencies, doublet splitting and line width. The complete
model uses ``Optimize(model=..., parameters=...)`` and public SCP helpers;
no input data, NMR plugin or additional imports are needed.

This is a pedagogical first-order line-shape model, not a COSY pulse-sequence
simulation. It omits coherence-transfer amplitudes, relaxation and strong
coupling. The fitted splitting is the injected synthetic line separation;
experimental coupling estimates require a validated forward model.
"""

import spectrochempy as scp


# %%
# Define the shared model
# -----------------------
# The sum and difference of two Gaussian lines produce in-phase and
# antiphase doublets. Both splitting and Gaussian FWHM are in Hz.
# Broadcasting their numerical values assembles full-grid responses from
# the public one-dimensional line shapes.
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


# %%
# Generate a reproducible spectrum
# --------------------------------
# The same resonance frequencies appear on both axes. The two diagonals
# have amplitudes 1.0 and 0.8; both cross peaks share amplitude 0.35.
# All features share an 8 Hz splitting and 3 Hz line FWHM. Independent
# normal noise has standard deviation 0.005 and a fixed seed.
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

# %%
# Fit the complete map
# ---------------------
# The callback receives a detached, zero-valued template with the input
# grid and effective mask, then numerical parameters in their declared units.
# Starting values differ from the injected values. Bounds keep resonances
# distinguishable and widths and splittings positive.
# Named responses allow positive amplitude projection at each nonlinear
# step. The cross-pair response retains both positive and negative lobes.
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

# %%
# Inspect parameter recovery and residuals
# ----------------------------------------
# Compare fitted parameters with their injected values and residual RMS
# with the known noise standard deviation. Generation and fitting use the
# same assumptions: this checks recovery, not experimental model validity.
# Four nonlinear parameters and three projected amplitudes count as seven
# estimated parameters. The retained Jacobian has four profiled columns;
# joint covariance is unavailable with reason ``profiled_amplitudes``.
print(f"Converged: {result.diagnostics['success']}")
print(f"Residual RMS: {result.diagnostics['rmse']:.4f} (noise SD: 0.005)")
for name, value in result.fitted_parameters.items():
    print(f"{name}: {value:.3f}; injected: {truth[name]:.3f} Hz")
for name, value in result.amplitudes.items():
    print(f"{name}: {value:.3f}")
print(f"Covariance: {result.diagnostics['covariance_reason']}")

# %%
# Compare the signed contours on a common scale
# ---------------------------------------------
# Solid red and dashed blue contours show positive and negative intensities.
# Fitted spectra and residuals preserve the input grid, units and mask.
# Nonfinite and masked observations are excluded consistently.
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
