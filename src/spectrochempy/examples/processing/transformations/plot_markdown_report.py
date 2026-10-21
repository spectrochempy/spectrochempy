# %%
# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
"""
Report a processed synthetic spectrum in Markdown
=================================================

``NDDataset.to_markdown_report()`` produces a deterministic Markdown summary of
one supplied dataset. It reports the dataset and its recorded processing history,
but does not itself analyse data, capture figures automatically, or establish
complete provenance.

This example builds a polymer-like IR spectrum with several line shapes, a
linear baseline and deterministic noise. It then corrects the baseline, smooths
the spectrum and locates peak maxima. The final report contains selected
metadata, processing history and an explicit reference to the exported figure.
"""

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

import spectrochempy as scp
from spectrochempy.reporting import FigureReference

# %%
# Create a synthetic polymer-like IR spectrum. The line-shape helpers use FWHM
# widths; the random generator has a fixed seed for reproducibility.
wavenumber = scp.Coord(
    np.linspace(4000.0, 600.0, 1401),
    title="Wavenumber",
    units="cm^-1",
)
bands = (
    scp.gaussian(wavenumber, ampl=0.65, pos=2950.0, width=90.0, normalized=False)
    + scp.voigt(
        wavenumber, ampl=1.0, pos=1720.0, width=65.0, ratio=0.35, normalized=False
    )
    + scp.gaussian(wavenumber, ampl=0.5, pos=1240.0, width=110.0, normalized=False)
)
baseline = 0.08 + 0.00001 * (4000.0 - wavenumber.data)
noise = np.random.default_rng(42).normal(0.0, 0.018, wavenumber.size)
raw = scp.NDDataset(
    bands.data + baseline + noise,
    coordset=[wavenumber],
    name="synthetic-polymer",
    title="Synthetic polymer spectrum",
    units="absorbance",
)
raw.meta.sample = "Synthetic polymer film"

# %%
# Correct the linear baseline, smooth the corrected spectrum, then locate peak
# maxima using coordinate-aware distance and prominence thresholds.
baseline_corrected = raw.basc()
processed = scp.savgol(baseline_corrected, size=21, order=3)
peaks, _ = processed.find_peaks(distance="120 cm^-1", prominence=0.08)

# %%
# Export the raw spectrum and the processed result. The detected maxima are
# visualized only in the final panel; the report receives the saved filename,
# not the live Matplotlib figure.
figure, (raw_axis, processed_axis) = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
raw_axis.plot(wavenumber.data, raw.data, color="0.35", label="Raw spectrum")
raw_axis.plot(wavenumber.data, baseline, color="C3", ls="--", label="Baseline")
raw_axis.set(title="Before processing", xlabel="Wavenumber / cm⁻¹", ylabel="Absorbance")
raw_axis.invert_xaxis()
raw_axis.legend()

processed_axis.plot(
    wavenumber.data,
    processed.data,
    color="C0",
    label="Baseline-corrected and smoothed",
)
processed_axis.scatter(
    peaks.x.values.m, peaks.data, color="C3", zorder=3, label="Peaks"
)
for position, intensity in zip(peaks.x.values.m, peaks.data, strict=False):
    processed_axis.annotate(
        f"{position:.0f}",
        (position, intensity),
        xytext=(0, 7),
        textcoords="offset points",
        ha="center",
        fontsize=8,
    )
processed_axis.set(title="After processing", xlabel="Wavenumber / cm⁻¹")
processed_axis.invert_xaxis()
processed_axis.legend()
figure.tight_layout()

figure_path = Path("processed-synthetic-polymer-spectrum.png")
figure.savefig(figure_path, dpi=150, bbox_inches="tight")

# %%
# Render and save the report. Metadata and figure references are opt-in, and
# no figure is captured implicitly. Keeping the PNG beside the Markdown file
# lets a Markdown viewer resolve the relative image reference.
report = processed.to_markdown_report(
    title="Processed synthetic polymer spectrum",
    metadata=("sample",),
    figures=(
        FigureReference(
            "Spectrum before and after processing",
            figure_path.as_posix(),
            "Left: raw spectrum and baseline. Right: baseline-corrected, "
            "Savitzky-Golay-smoothed spectrum with detected peak maxima.",
        ),
    ),
)
report_path = Path("processed-synthetic-polymer-spectrum-report.md")
report_path.write_text(report, encoding="utf-8")
print(report)
