# %%
# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: D205, D400, D415, T201
"""
Compare spectral smoothers on noisy peaks
=========================================

Use a seeded synthetic spectrum to compare Butterworth, Savitzky-Golay,
Whittaker and Gaussian smoothing. The noiseless spectrum provides reference
peak positions, heights and full widths at half maximum (FWHM).
"""

import numpy as np

import spectrochempy as scp

coordinate = scp.Coord.linspace(1000, 1255.75, 1024, units="cm^-1")
axis = coordinate.data
reference = np.exp(-0.5 * ((axis - 1075) / 1.5) ** 2)
reference += 0.65 * np.exp(-0.5 * ((axis - 1160) / 3.0) ** 2)

filters = {
    "Butterworth": scp.Filter(method="butterworth", cutoff=0.18, order=3),
    "Savitzky-Golay": scp.Filter(method="savgol", size=17, order=3),
    "Whittaker": scp.Filter(method="whittaker", lamb=32, order=2),
    "Gaussian": scp.Filter(method="gaussian", sigma=2.0),
}

# %%
# Measure RMS error in peak-free regions and near the edges. Find the maximum
# in a window around each reference peak. Estimate its FWHM by linearly
# interpolating the nearest half-maximum crossings on both sides.
# The filter settings are illustrative. Each spectrum uses the same noise
# realization scaled to the specified standard deviation.

baseline = (axis < 1050) | (axis > 1200)
edges = (axis < 1005) | (axis > 1250)
windows = [(abs(axis - center) < 8).nonzero()[0] for center in (1075, 1160)]


def half_max_crossing(signal, indices, half_height):
    outer = np.flatnonzero(signal[indices] <= half_height)[0]
    inside, outside = indices[outer - 1 : outer + 1]
    fraction = (half_height - signal[inside]) / (signal[outside] - signal[inside])
    return axis[inside] + fraction * (axis[outside] - axis[inside])


def peak_metrics(signal):
    metrics = []
    for window in windows:
        offset = np.argmax(signal[window])
        peak = window[offset]
        half_height = signal[peak] / 2
        left = half_max_crossing(signal, window[: offset + 1][::-1], half_height)
        right = half_max_crossing(signal, window[offset:], half_height)
        metrics.append((axis[peak], signal[peak], right - left))
    return np.array(metrics).T


def rms(values):
    return np.sqrt(np.mean(values**2))


reference_metrics = peak_metrics(reference)
noise = np.random.default_rng(1852).standard_normal(axis.size)

for noise_std in (0.02, 0.06):
    dataset = scp.NDDataset(
        reference + noise_std * noise,
        coordset=[coordinate],
        units="absorbance",
        title="absorbance",
    )
    results = {name: model(dataset) for name, model in filters.items()}
    print(f"Noise standard deviation {noise_std:g}")
    print("Method            Baseline RMS  Edge RMS  Δposition   Δheight    ΔFWHM")
    for name, spectrum in {"Raw": dataset, **results}.items():
        residual = spectrum.data - reference
        position, height, width = np.max(
            np.abs(peak_metrics(spectrum.data) - reference_metrics), axis=1
        )
        print(
            f"{name:16} {rms(residual[baseline]):12.5f} {rms(residual[edges]):9.5f}"
            f" {position:10.3f} {height:9.5f} {width:9.3f}"
        )

# %%
# Plot the higher-noise comparison using the first reference peak.

target = abs(axis - 1075) < 12
comparison = scp.NDDataset(
    np.stack([reference, dataset.data, *(item.data for item in results.values())])[
        :, target
    ],
    coordset=[scp.Coord(labels=["Reference", "Raw", *results]), coordinate[target]],
    units="absorbance",
    title="absorbance",
)
_ = comparison.plot(method="lines", legend=True)
