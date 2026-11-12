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

import spectrochempy as scp

axis = scp.linspace(1000, 1255.75, 1024)
reference = scp.exp(-0.5 * ((axis - 1075) / 1.5) ** 2)
reference += 0.65 * scp.exp(-0.5 * ((axis - 1160) / 3.0) ** 2)
reference.x = scp.Coord(axis.data, units="cm^-1")
reference.units = "absorbance"
reference.title = "absorbance"

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

centers = (1075.0, 1160.0)


def peak_metrics(spectrum):
    metrics = []
    for center in centers:
        window = spectrum[center - 7.75 : center + 7.75]
        peak = window.characterize_peaks_1d(window.argmax()).rows[0]
        metrics.append(
            (
                peak["position"].magnitude,
                peak["sample_height"].magnitude,
                peak["fwhm"].magnitude,
            )
        )
    return metrics


def residual_rms(residual, left_end, right_start):
    region = scp.concatenate(residual[:left_end], residual[right_start:])
    return region.rms().magnitude


reference_metrics = peak_metrics(reference)
noise = scp.normal(
    size=reference.shape, seed=1852, units=reference.units, coordset=[reference.x]
)

for noise_std in (0.02, 0.06):
    dataset = reference + noise_std * noise
    results = {name: model(dataset) for name, model in filters.items()}
    print(f"Noise standard deviation {noise_std:g}")
    print("Method            Baseline RMS  Edge RMS  Δposition   Δheight    ΔFWHM")
    for name, spectrum in {"Raw": dataset, **results}.items():
        residual = spectrum - reference
        peak_pairs = zip(peak_metrics(spectrum), reference_metrics, strict=True)
        peak_errors = [
            tuple(
                abs(value - target)
                for value, target in zip(actual, expected, strict=True)
            )
            for actual, expected in peak_pairs
        ]
        position, height, width = map(max, zip(*peak_errors, strict=True))
        baseline = residual_rms(residual, 1049.75, 1200.25)
        edges = residual_rms(residual, 1004.75, 1250.25)
        print(
            f"{name:16} {baseline:12.5f} {edges:9.5f}"
            f" {position:10.3f} {height:9.5f} {width:9.3f}"
        )

# %%
# Plot the higher-noise comparison using the first reference peak.

comparison = scp.stack(reference, dataset, *results.values())
comparison.y = scp.Coord(labels=["Reference", "Raw", *results])
comparison.title = reference.title
comparison = comparison[:, 1063.25:1086.75]
_ = comparison.plot(method="lines", legend=True)
