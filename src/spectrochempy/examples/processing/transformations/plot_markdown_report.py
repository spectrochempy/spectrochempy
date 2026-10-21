# %%
# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
"""
Report a preprocessed spectrum in Markdown
==========================================

``NDDataset.to_markdown_report()`` produces a deterministic Markdown summary of
one supplied dataset. It reports the dataset and its recorded processing history;
it does not analyse data, capture figures automatically, or establish complete
provenance.

This example smooths a synthetic spectrum, exports a comparison figure, then
renders selected metadata, the processing history, and an explicit reference to
that figure. The caller chooses which metadata and figures are safe to include.
"""

from pathlib import Path

import numpy as np

import spectrochempy as scp
from spectrochempy.reporting import FigureReference

# %%
# Create a spectrum and record its sample identifier as metadata.
wavenumber = scp.Coord(
    np.linspace(4000.0, 400.0, 101),
    title="Wavenumber",
    units="cm^-1",
)
raw = scp.NDDataset(
    np.sin(np.linspace(0.0, 4.0, 101)),
    coordset=[wavenumber],
    name="polymer-film",
    title="Absorbance",
    units="absorbance",
)
raw.meta.sample = "Polymer film"

# %%
# Apply a preprocessing step. The result retains a history entry describing the
# operation.
processed = scp.savgol(raw, size=5, order=2)

# %%
# Export an explicit before/after figure. The report receives its filename, not
# the live Matplotlib figure.
axis = raw.plot(color="0.5", label="Before smoothing")
_ = processed.plot(clear=False, color="C0", label="After smoothing")
_ = axis.legend()

figure_path = Path("preprocessed-polymer-film.png")
axis.get_figure().savefig(figure_path, dpi=150, bbox_inches="tight")

# %%
# Render and save the report. Metadata and figure references are opt-in, and
# no figure is captured implicitly. Keeping the PNG beside the Markdown file
# lets a Markdown viewer resolve the relative image reference.
report = processed.to_markdown_report(
    title="Preprocessed polymer-film spectrum",
    metadata=("sample",),
    figures=(
        FigureReference(
            "Spectrum before and after smoothing",
            figure_path.as_posix(),
            "Grey: input spectrum; blue: Savitzky-Golay-smoothed spectrum.",
        ),
    ),
)
report_path = Path("preprocessed-polymer-film-report.md")
report_path.write_text(report, encoding="utf-8")
print(report)
