# %%
# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
"""
Bounded Bruker COSY 2D processing replay
========================================

This example compares the bundled Bruker ``exam2d_HH/1`` QF/COSY TopSpin
``pdata/1/2rr`` spectrum with a dataset-specific raw ``ser`` processing replay.

The replay follows the ``xfb`` processing recorded in ``pdata/1/auditp.txt``:
sine apodization, F2 Fourier transform, F1 first-point correction, F1 linear
prediction to ``LPBIN=256`` with ``NCOEF=32``, zero filling to ``SI=1024``, F1
Fourier transform, orientation to the TopSpin ``2rr`` side, and magnitude
calculation. It remains deliberately bounded to this dataset and does not define
a general TopSpin replay API.

Requires the official ``spectrochempy-nmr`` plugin.
Install with: ``pip install spectrochempy[nmr]``.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

import spectrochempy as scp

SI = 1024
LPBIN_F1 = 256
NCOEF_F1 = 32


def _sine(data):
    size = data.shape[-1]
    return data * np.sin(np.pi * np.arange(size) / (size - 1)).astype(data.dtype)


def _zf_size(data, size):
    filled = np.zeros(data.shape[:-1] + (size,), dtype=data.dtype)
    filled[..., : data.shape[-1]] = data
    return filled


def _fft(data):
    return np.fft.fftshift(np.fft.fft(data, axis=-1).astype(data.dtype), axes=-1)


def _lp_coefficients(trace, order):
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


def _zf_f1(data, size):
    return _zf_size(data.T, size).T


def _lp_f1(data, final_time_size):
    predicted = np.empty((final_time_size, data.shape[1]), dtype=data.dtype)
    for i in range(data.shape[1]):
        predicted[:, i] = _lp_predict_after(data[:, i], final_time_size, NCOEF_F1)
    return predicted


def _qf_cosy_exam2d_hh(raw):
    data = np.asarray(raw.data).copy()
    data = _sine(data)
    data = _zf_size(data, SI)
    data = _fft(data)

    data = data.copy()
    data[0, :] *= 0.5
    data = _lp_f1(data, LPBIN_F1)
    data = _sine(data.T).T
    data = _zf_f1(data, SI)
    data = _fft(data.T).T
    # Match the TopSpin 2rr side/orientation for this QF dataset.
    return np.roll(np.abs(data)[::-1, :], 1, axis=0)


def _normalized(data):
    data = np.asarray(data, dtype=np.float64)
    return data / np.max(np.abs(data))


def _scaled_metrics(calculated, reference):
    calculated = np.asarray(calculated, dtype=float)
    reference = np.asarray(reference, dtype=float)
    scale = float(
        np.vdot(calculated.ravel(), reference.ravel()).real
        / np.vdot(calculated.ravel(), calculated.ravel()).real
    )
    scaled = calculated * scale
    overlap = float(
        np.vdot(scaled.ravel(), reference.ravel()).real
        / (np.linalg.norm(scaled.ravel()) * np.linalg.norm(reference.ravel()))
    )
    rms = float(
        np.sqrt(np.mean((scaled - reference) ** 2)) / np.sqrt(np.mean(reference**2))
    )
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

topspin_raw = np.asarray(topspin_2rr.data).real
replayed_raw = _qf_cosy_exam2d_hh(raw)
replayed_scaled, overlap, rms_rel = _scaled_metrics(replayed_raw, topspin_raw)

topspin = _normalized(topspin_raw)
replayed = _normalized(replayed_scaled)
residual = replayed - topspin
topspin_diagonal = np.diag(topspin)
replayed_diagonal = np.diag(replayed)
residual_diagonal = replayed_diagonal - topspin_diagonal

x_ppm = np.asarray(topspin_2rr.x.data, dtype=float)
y_ppm = np.asarray(topspin_2rr.y.data, dtype=float)
window = (8.8, 0.5)
levels = np.linspace(0.08, 0.80, 10)


# %%
# Processing path used for the replay
# -----------------------------------
#
# This is not a call to a SpectroChemPy processing API: SpectroChemPy does not
# yet expose the F1 linear-prediction step needed by this TopSpin ``xfb`` audit
# trail. The replay therefore keeps the missing operation local to this gallery
# example:
#
# * F2: sine window, zero-fill to ``SI=1024``, Fourier transform.
# * F1: first point correction ``FCOR=0.5``.
# * F1: linear prediction from 128 to ``LPBIN=256`` with ``NCOEF=32``.
# * F1: sine window, zero-fill to ``SI=1024``, Fourier transform.
# * Magnitude and orientation matching to the TopSpin ``2rr`` side.
#
# The LP step is the important difference from the earlier simplified replay:
# without it, the F1 peaks are visibly broader than the TopSpin reference.


# %%
# Plot the TopSpin reference, the bounded replay, and the scaled residual.
fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

for ax, data, title in [
    (axes[0], topspin, "TopSpin processed 2rr"),
    (axes[1], replayed, "ser replay with F1 LP"),
]:
    ax.contour(x_ppm, y_ppm, data, levels=levels, colors="black", linewidths=0.8)
    ax.plot(x_ppm, y_ppm, color="tab:red", linewidth=0.8, alpha=0.6)
    ax.set_title(title)
    ax.set_xlim(window)
    ax.set_ylim(window)
    ax.set_xlabel("F2 / ppm")
    ax.set_ylabel("F1 / ppm")
    ax.set_aspect("equal", adjustable="box")

residual_levels = np.linspace(-0.12, 0.12, 13)
mesh = axes[2].contourf(
    x_ppm,
    y_ppm,
    residual,
    levels=residual_levels,
    cmap="coolwarm",
    extend="both",
)
axes[2].set_title("Scaled residual")
axes[2].plot(x_ppm, y_ppm, color="black", linewidth=0.8, alpha=0.6)
axes[2].set_xlim(window)
axes[2].set_ylim(window)
axes[2].set_xlabel("F2 / ppm")
axes[2].set_ylabel("F1 / ppm")
axes[2].set_aspect("equal", adjustable="box")
fig.colorbar(mesh, ax=axes[2], shrink=0.8)

fig.text(
    0.5,
    -0.02,
    f"auditp xfb: F1 LP to {LPBIN_F1} with NCOEF={NCOEF_F1}; "
    f"overlap={overlap:.6f}, RMS_rel={rms_rel:.6f}; "
    "bounded to exam2d_HH/1 with F1 side/orientation matched to 2rr.",
    ha="center",
)

fig.suptitle("exam2d_HH/1 QF/COSY: TopSpin 2rr vs bounded ser replay")

# %%
# Diagonal profile
# ----------------
# The global residual is small, but the diagonal profile exposes remaining local
# differences in peak shape and intensity along the COSY autocorrelation line.
fig_diag, ax_diag = plt.subplots(figsize=(7, 3), constrained_layout=True)

ax_diag.plot(x_ppm, topspin_diagonal, label="TopSpin 2rr", linewidth=1.4)
ax_diag.plot(
    x_ppm,
    replayed_diagonal,
    label="bounded ser replay",
    linewidth=1.2,
    linestyle="--",
)
ax_diag.plot(
    x_ppm,
    residual_diagonal,
    label="replay - TopSpin",
    linewidth=0.9,
    color="tab:red",
    alpha=0.8,
)
ax_diag.axhline(0.0, color="0.4", linewidth=0.6)
ax_diag.set_xlim(window)
ax_diag.set_xlabel("F2/F1 diagonal / ppm")
ax_diag.set_ylabel("normalized intensity")
ax_diag.set_title("Diagonal cut through the TopSpin and replayed spectra")
ax_diag.legend(loc="upper left")

# %%
# The F1 linear prediction step is essential here: without it, the indirect
# dimension peaks are visibly broader than the TopSpin reference.

# %%
# Uncomment the following line to display all figures when running the script
# directly with Python.

# %%

# scp.show()
