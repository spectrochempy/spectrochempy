# %%
# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
"""
Investigating a Bruker COSY 2D processing replay
=============================================

This example compares the bundled Bruker ``exam2d_HH/1`` QF/COSY TopSpin
``pdata/1/2rr`` spectrum with a dataset-specific raw ``ser`` processing replay.

The candidate replay uses operations recorded for ``xfb`` in ``pdata/1/auditp.txt``:
sine apodization, F2 Fourier transform, F1 first-point correction, F1 linear
prediction to ``LPBIN=256`` with ``NCOEF=32``, zero filling to ``SI=1024``, F1
Fourier transform with the QF convention, and magnitude
calculation. The local linear-prediction algorithm has not been established as
equivalent to TopSpin. Significant local differences remain despite a high
global overlap; this example investigates them rather than validating a TopSpin
reproduction. It remains deliberately bounded to this dataset.

Requires the official ``spectrochempy-nmr`` plugin.
Install with: ``pip install spectrochempy[nmr]``.
"""

# %%
import matplotlib.pyplot as plt

import spectrochempy as scp

SI = 1024
LPBIN_F1 = 256
NCOEF_F1 = 32


# %%
# Temporary local linear prediction
# ---------------------------------
# SpectroChemPy does not yet expose the required LP operation. NumPy/SciPy are
# used only inside this local prototype; the surrounding processing uses SCP.
def _lp_coefficients(trace, order):
    import numpy as np
    import scipy.linalg

    size = len(trace)
    rows = size - order
    matrix = scipy.linalg.hankel(trace[:rows], trace[rows - 1 : -1])
    rhs = trace[order:].reshape(rows, 1)
    u, singular, vh = scipy.linalg.svd(matrix)
    cutoff = singular[0] * np.finfo(np.float64).eps * 1.0e6
    sigma_inverse = np.zeros((order, rows), dtype=singular.dtype)
    for i, value in enumerate(singular):
        if value > cutoff:
            sigma_inverse[i, i] = 1.0 / np.conj(value)
    coefficients = np.asarray(
        np.asmatrix(vh).H
        * np.asmatrix(sigma_inverse)
        * np.asmatrix(u).H
        * np.asmatrix(rhs)
    ).ravel()

    polynomial = np.empty(len(coefficients) + 1, dtype=coefficients.dtype)
    polynomial[0] = 1.0 + 0.0j
    polynomial[1:] = -coefficients[::-1]
    roots = np.roots(polynomial)
    for i, root in enumerate(roots):
        if abs(root) > 1.0:
            roots[i] = root / abs(root)
    return np.squeeze(-np.poly(roots)[:0:-1])


def _lp_predict_after(trace, final_size, order):
    import numpy as np

    pred = final_size - len(trace)
    if pred <= 0:
        return trace
    coefficients = _lp_coefficients(trace, order)
    predicted = np.empty(final_size, dtype=trace.dtype)
    predicted[: len(trace)] = trace
    for i in range(pred):
        predicted[len(trace) + i] = np.sum(
            predicted[len(trace) - order + i : len(trace) + i] * coefficients
        )
    return predicted


def _lp_f1(dataset, final_time_size):
    predicted = scp.zf_size(dataset, size=final_time_size, dim="y")
    for i in range(dataset.shape[1]):
        predicted.data[:, i] = _lp_predict_after(
            dataset.data[:, i], final_time_size, NCOEF_F1
        )
    predicted.history = (
        f"Filled F1 extension with local forward LP: order={NCOEF_F1}, "
        f"size={final_time_size}, SVD coefficients and root projection"
    )
    return predicted


def _qf_cosy_exam2d_hh(raw):
    f2 = scp.sp(raw, ssb=0, pow=1, dim="x")
    f2 = scp.fft(f2, size=SI, dim="x")
    f2[0, :] *= 0.5
    f2.history = "Applied F1 first-point correction FCOR=0.5 before local LP"
    predicted = _lp_f1(f2, LPBIN_F1)
    f1 = scp.sp(predicted, ssb=0, pow=1, dim="y")
    f1 = scp.fft(f1, size=SI, dim="y")
    return scp.absolute(f1)


def _diagonal_cut(dataset):
    x_ppm = dataset.x.data
    y_ppm = dataset.y.data
    common = dataset.x[(x_ppm >= y_ppm.min()) & (x_ppm <= y_ppm.max())]
    aligned = scp.interpolate(
        dataset, dims=["x", "y"], coord={"x": common, "y": common}
    )
    return scp.diagonal(aligned, dim="x")


def _scaled_metrics(calculated, reference):
    scale = (calculated * reference).sum() / (calculated**2).sum()
    scaled = calculated * scale
    overlap = float(
        (scaled * reference).sum() / ((scaled**2).sum() * (reference**2).sum()) ** 0.5
    )
    rms = float((((scaled - reference) ** 2).sum() / (reference**2).sum()) ** 0.5)
    return scaled, overlap, rms


# %%
# Read the raw dataset and the processed TopSpin reference from the existing
# bundled test data. No data file is modified.
dataset = "nmrdata/bruker/tests/nmr/exam2d_HH"
raw = scp.nmr.read(dataset, expno=1, remove_digital_filter=True)
topspin_2rr = scp.nmr.read(
    "nmrdata/bruker/tests/nmr/exam2d_HH/1/pdata/1/2rr",
    proc_axis=True,
)

topspin_raw = topspin_2rr.real
replayed_spectrum = _qf_cosy_exam2d_hh(raw)

# %%
# Keep the public FFT result and its acquisition-derived coordinates intact.
# For the bin-wise oracle comparison only, assign the processed vendor grid to
# a copy. This explicit calibration does not shift or interpolate its intensities.
replayed_raw = replayed_spectrum.copy()
replayed_raw.set_coordset(x=topspin_2rr.x.copy(), y=topspin_2rr.y.copy())
replayed_raw.history = "Assigned TopSpin processed grid for bin-wise comparison"
replayed_scaled, overlap, rms_rel = _scaled_metrics(replayed_raw, topspin_raw)

reference_max = scp.absolute(topspin_raw).max()
topspin = topspin_raw / reference_max
replayed = replayed_scaled / reference_max
residual = replayed - topspin

x_ppm = topspin_2rr.x.data
y_ppm = topspin_2rr.y.data
topspin_diagonal = _diagonal_cut(topspin)
replayed_diagonal = _diagonal_cut(replayed)
diagonal_ppm = topspin_diagonal.x.data
residual_diagonal = replayed_diagonal - topspin_diagonal
diagonal_rms_rel = float(
    ((residual_diagonal**2).sum() / (topspin_diagonal**2).sum()) ** 0.5
)
window = (8.8, 0.5)
levels = scp.linspace(0.08, 0.80, 10).data


# %%
# Processing path used for the replay
# -----------------------------------
#
# Apodization, zero filling, both FFTs, and magnitude use the public SCP API.
# The FFT's ``size`` argument handles zero filling. Only the missing F1 LP
# operation remains a local prototype:
#
# * F2: sine window, zero-fill to ``SI=1024``, Fourier transform.
# * F1: first point correction ``FCOR=0.5``.
# * F1: linear prediction from 128 to ``LPBIN=256`` with ``NCOEF=32``.
# * F1: sine window, zero-fill to ``SI=1024``, Fourier transform.
# * Magnitude. The QF/DQD metadata select the Fourier conventions; no manual
#   flip, roll, or conjugation is applied by the example.
#
# The LP step is the important difference from the earlier simplified replay:
# without it, the F1 peaks are visibly broader than the TopSpin reference.
# This local SVD prediction with root projection is an approximation: the
# vendor parameters do not establish its solver, stabilization, or the placement
# of FCOR relative to coefficient estimation. This example now exercises the
# public QF/DQD FFT dispatch corrected by #1771, while retaining the same local
# LP approximation and the unresolved discrepancy with TopSpin.
#
# One least-squares amplitude factor is fitted over the entire 2D matrix.
# Both spectra and every residual below then use the same TopSpin maximum as
# intensity unit. No further peak-wise or profile-wise scaling is applied.
# Coordinates for both matrices come from the processed vendor grid; these
# plots do not validate the coordinates produced by the public FFT API.

print(f"Global overlap: {overlap:.6f}; relative L2 residual: {rms_rel:.2%}")
print(f"Diagonal relative L2 residual: {diagonal_rms_rel:.2%}")
print(
    f"Maximum absolute residual / TopSpin maximum: {float(scp.absolute(residual).max()):.2%}"
)


# %%
# Plot the TopSpin reference, the bounded replay, and the scaled residual.
fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

for ax, data, title in [
    (axes[0], topspin, "TopSpin processed 2rr"),
    (axes[1], replayed, "ser replay with F1 LP"),
]:
    ax.contour(x_ppm, y_ppm, data.data, levels=levels, colors="black", linewidths=0.8)
    ax.plot(diagonal_ppm, diagonal_ppm, color="tab:red", linewidth=0.8, alpha=0.6)
    ax.set_title(title)
    ax.set_xlim(window)
    ax.set_ylim(window)
    ax.set_xlabel("F2 / ppm")
    ax.set_ylabel("F1 / ppm")
    ax.set_aspect("equal", adjustable="box")

residual_levels = scp.linspace(-0.12, 0.12, 13).data
mesh = axes[2].contourf(
    x_ppm,
    y_ppm,
    residual.data,
    levels=residual_levels,
    cmap="coolwarm",
    extend="both",
)
axes[2].set_title("Scaled residual")
axes[2].plot(diagonal_ppm, diagonal_ppm, color="black", linewidth=0.8, alpha=0.6)
axes[2].set_xlim(window)
axes[2].set_ylim(window)
axes[2].set_xlabel("F2 / ppm")
axes[2].set_ylabel("F1 / ppm")
axes[2].set_aspect("equal", adjustable="box")
fig.colorbar(mesh, ax=axes[2], shrink=0.8)

_ = fig.suptitle(
    "exam2d_HH/1 QF/COSY: TopSpin 2rr vs candidate ser replay\n"
    f"Global relative L2 residual: {rms_rel:.2%}; local differences unresolved"
)

# %%
# Diagonal profile
# ----------------
# The diagonal is F1 = F2 in ppm, interpolated on the two vendor axes, which
# differ slightly. A matrix index diagonal need not follow this physical line.
# A separate residual panel and a magnified region expose differences that are
# hard to distinguish in a full-range overlay.
fig_diag, diagonal_axes = plt.subplots(
    2,
    2,
    figsize=(12, 5),
    sharex="col",
    constrained_layout=True,
    gridspec_kw={"height_ratios": [3, 1]},
)

for column, limits in enumerate((window, (1.5, 0.9))):
    ax_diag = diagonal_axes[0, column]
    ax_res = diagonal_axes[1, column]
    ax_diag.plot(
        diagonal_ppm, topspin_diagonal.data, label="TopSpin 2rr", linewidth=1.4
    )
    ax_diag.plot(
        diagonal_ppm,
        replayed_diagonal.data,
        label="candidate ser replay",
        linewidth=1.2,
        linestyle="--",
    )
    ax_res.plot(diagonal_ppm, residual_diagonal.data, color="tab:red", linewidth=0.9)
    ax_res.axhline(0.0, color="0.4", linewidth=0.6)
    ax_diag.set_xlim(limits)
    ax_diag.set_ylabel("Intensity / TopSpin maximum")
    ax_res.set_ylabel("Difference")
    ax_res.set_xlabel("F1 = F2 / ppm")
    ax_diag.legend()
_ = fig_diag.suptitle("Physical diagonal: full range and 1.5–0.9 ppm detail")

# %%
# Off-diagonal diagnostic
# ----------------------
# Select the largest absolute residual inside the displayed window, excluding
# a 0.2 ppm band around the diagonal. Both F1/F2 cuts pass through this same
# vendor-grid point; no peak shifts or local amplitude fits are used.
off_diagonal = (
    (abs(y_ppm[:, None] - x_ppm[None, :]) > 0.2)
    & (x_ppm[None, :] >= window[1])
    & (x_ppm[None, :] <= window[0])
    & (y_ppm[:, None] >= window[1])
    & (y_ppm[:, None] <= window[0])
)
off_diagonal_residual = scp.absolute(residual)
off_diagonal_residual.mask = ~off_diagonal
cut_y, cut_x = off_diagonal_residual.argmax()
print(
    f"Off-diagonal residual maximum at F1={y_ppm[cut_y]:.4f}, "
    f"F2={x_ppm[cut_x]:.4f} ppm: "
    f"difference / TopSpin maximum = {float(residual[cut_y, cut_x].sum()):.2%}"
)
fig_cuts, cut_axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
for ax, ppm, reference_cut, replay_cut, center, title in (
    (
        cut_axes[0],
        y_ppm,
        topspin[:, cut_x].squeeze(),
        replayed[:, cut_x].squeeze(),
        y_ppm[cut_y],
        f"F1 cut at F2 = {x_ppm[cut_x]:.4f} ppm",
    ),
    (
        cut_axes[1],
        x_ppm,
        topspin[cut_y, :].squeeze(),
        replayed[cut_y, :].squeeze(),
        x_ppm[cut_x],
        f"F2 cut at F1 = {y_ppm[cut_y]:.4f} ppm",
    ),
):
    difference = replay_cut - reference_cut
    ax.plot(ppm, reference_cut.data, label="TopSpin 2rr")
    ax.plot(ppm, replay_cut.data, "--", label="candidate ser replay")
    ax.plot(ppm, difference.data, color="tab:red", label="difference")
    ax.axhline(0, color="0.4", linewidth=0.6)
    ax.set_xlim(center + 0.15, center - 0.15)
    visible = abs(ppm - center) <= 0.15
    low = min(
        float(cut[visible].min()) for cut in (reference_cut, replay_cut, difference)
    )
    high = max(
        float(cut[visible].max()) for cut in (reference_cut, replay_cut, difference)
    )
    margin = 0.05 * (high - low)
    ax.set_ylim(low - margin, high + margin)
    ax.set_xlabel("Chemical shift / ppm")
    ax.set_ylabel("Intensity / TopSpin maximum")
    ax.set_title(title)
    ax.legend()
_ = fig_cuts.suptitle("Largest off-diagonal discrepancy (same global intensity scale)")

# %%
# Uncomment the following line to display all figures when running the script
# directly with Python.

# %%

# scp.show()
