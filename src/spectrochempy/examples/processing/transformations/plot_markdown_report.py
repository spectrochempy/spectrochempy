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
the spectrum and locates peak maxima. The final report presents each recorded
transformation as a readable section with its method, parameters, input sources
and traceability details, alongside selected metadata and an explicit reference
to the exported figure.
"""

from pathlib import Path

import spectrochempy as scp
from spectrochempy.reporting import FigureReference

# %%
# Create a synthetic polymer-like IR spectrum. The line-shape helpers use FWHM
# widths; the random generator has a fixed seed for reproducibility.
wavenumber = scp.Coord.linspace(
    4000.0,
    600.0,
    1401,
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
noise = scp.normal(0.0, 0.018, size=wavenumber.size, seed=42)
raw = scp.NDDataset(
    bands.data + baseline + noise.data,
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
# ``basc()`` and ``savgol()`` transform the dataset and append structured
# history entries identifying the polynomial baseline model and the
# Savitzky-Golay window and polynomial order. ``find_peaks()`` is an analysis:
# it returns a separate peak result and leaves the processed spectrum history
# unchanged. Plotting is a visualization step and is not recorded as a
# transformation.

# %%
# Export the raw spectrum and the processed result with SpectroChemPy's plotting
# helpers. ``multiplot`` creates the shared layout, while ``plot_multiple``
# overlays the fitted baseline and the detected maxima on the respective panels.
# ``plot_multiple`` applies one geometry to all traces, so the marker and
# line-style lists keep the spectrum as a line and render the peaks as markers
# only. Explicit axis limits preserve the full spectral range and keep both
# panels on the same absorbance scale after the overlays. The report receives
# the saved filename, not the live figure.
figure_path = Path("processed-synthetic-polymer-spectrum.png")
axes = scp.multiplot(
    [raw, processed],
    nrow=1,
    ncol=2,
    sharey=True,
    method="pen",
    figsize=(11, 4),
    show=False,
)
fitted_baseline = scp.get_baseline(raw)
scp.plot_multiple(
    [raw, fitted_baseline],
    ax=axes[0],
    clear=True,
    labels=["Raw spectrum", "Baseline"],
    legend="best",
    method="pen",
    color=["0.35", "C3"],
    ls=["-", "--"],
    title="Before processing",
    xlim=(4000, 600),
    ylim=(-0.05, 1.15),
    ylabel="Absorbance",
    show=False,
)
scp.plot_multiple(
    [processed, peaks],
    ax=axes[1],
    clear=True,
    labels=["Baseline-corrected and smoothed", "Peaks"],
    legend="best",
    method="pen",
    marker=["", "o"],
    ls=["-", "None"],
    color=["C0", "C3"],
    title="After processing",
    xlim=(4000, 600),
    ylim=(-0.05, 1.15),
    ylabel="Absorbance",
    output=figure_path,
    show=False,
)

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
