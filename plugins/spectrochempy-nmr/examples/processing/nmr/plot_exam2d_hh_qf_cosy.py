# %%
# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
"""
Bounded visual comparison of a Bruker COSY 2D
=============================================

This example compares two visual renderings of the bundled Bruker
``exam2d_HH/1`` QF/COSY dataset:

* the TopSpin processed ``pdata/1/2rr`` spectrum;
* a bounded FFT reconstruction from the raw ``ser`` file.

The figure is intended for human visual inspection of the dominant 2D peak
topology. It is not the primary scientific proof, does not claim an exact
TopSpin replay, and does not establish cross-vendor, States-TPPI,
Echo-Antiecho, pseudo-2D, P33/CENAPT, 17O J-HMQC, or fractional-``GRPDLY``
validity.

Requires the official ``spectrochempy-nmr`` plugin.
Install with: ``pip install spectrochempy[nmr]``.
"""

# %%
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import spectrochempy as scp


# %%
# The reconstruction below mirrors the bounded ``exam2d_HH/1`` QF/COSY path
# characterized in PR #1769. It is deliberately local to this example and is not
# a public TopSpin replay API.
def _sine_window(size):
    return np.sin(np.pi * (np.arange(size) + 0.5) / size)


def _qf_cosy_exam2d_hh(raw):
    data = np.asarray(raw.data).copy()
    direct = np.zeros((1024, 1024), dtype=np.complex128)
    direct[: data.shape[0], : data.shape[1]] = data * _sine_window(data.shape[1])
    f2 = np.fft.fftshift(np.fft.fft(direct, axis=1), axes=1)
    f2 *= _sine_window(f2.shape[0])[:, None]
    return np.abs(np.fft.fftshift(np.fft.ifft(f2, axis=0), axes=0))


def _normalized(data):
    data = np.asarray(data, dtype=np.float64)
    return data / np.max(np.abs(data))


# %%
# Read the raw dataset and the processed TopSpin reference from the existing
# bundled test data. No data file is modified.
dataset = "nmrdata/bruker/tests/nmr/exam2d_HH"
raw = scp.nmr.read(dataset, expno=1, remove_digital_filter=True)
topspin_2rr = scp.nmr.read(
    "nmrdata/bruker/tests/nmr/exam2d_HH/1/pdata/1/2rr",
    proc_axis=True,
)

topspin = _normalized(np.asarray(topspin_2rr.data).real)
reconstructed = _normalized(_qf_cosy_exam2d_hh(raw))

x_ppm = np.asarray(topspin_2rr.x.data, dtype=float)
y_ppm = np.asarray(topspin_2rr.y.data, dtype=float)
window = (8.8, 0.5)
levels = np.linspace(0.08, 0.80, 10)


# %%
# Plot both spectra with the same ppm window and contour levels. The third panel
# overlays the contours as a qualitative alignment aid only.
fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

panels = [
    (axes[0], topspin, "TopSpin processed 2rr"),
    (axes[1], reconstructed, "FFT reconstruction from ser"),
]

for ax, data, title in panels:
    ax.contour(x_ppm, y_ppm, data, levels=levels, colors="black", linewidths=0.8)
    ax.set_title(title)
    ax.set_xlim(window)
    ax.set_ylim(window)
    ax.set_xlabel("F2 / ppm")
    ax.set_ylabel("F1 / ppm")
    ax.set_aspect("equal", adjustable="box")

axes[2].contour(
    x_ppm,
    y_ppm,
    reconstructed,
    levels=levels,
    colors="C1",
    linewidths=0.8,
    linestyles="--",
)
axes[2].contour(
    x_ppm,
    y_ppm,
    topspin,
    levels=levels,
    colors="black",
    linewidths=0.8,
)
axes[2].set_title("Qualitative contour overlay")
axes[2].set_xlim(window)
axes[2].set_ylim(window)
axes[2].set_xlabel("F2 / ppm")
axes[2].set_ylabel("F1 / ppm")
axes[2].set_aspect("equal", adjustable="box")
axes[2].legend(
    handles=[
        Line2D([0], [0], color="black", lw=1.0, label="TopSpin 2rr"),
        Line2D([0], [0], color="C1", lw=1.0, ls="--", label="FFT from ser"),
    ],
    loc="lower left",
    frameon=False,
)

fig.suptitle("exam2d_HH/1 QF/COSY: visual topology comparison")

# %%
# The matching scientific evidence remains the bounded regression from PR #1769.
# This gallery figure only illustrates that the dominant COSY topology is similar
# on the same processed ppm axes.

# %%
# Uncomment the following line to display all figures when running the script
# directly with Python.

# %%

# scp.show()
