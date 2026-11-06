# %%
# ========================================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ========================================================================================================
# ruff: noqa
"""
Report a processed IR spectrum in Markdown
==================================================

``NDDataset.to_markdown_report()`` summarizes a dataset and its recorded history.
It does not provide an executable workflow or a complete provenance record.
Figures are included only through explicit references.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import spectrochempy as scp
from spectrochempy.reporting import FigureReference

# %%
# Read a standard IR dataset and smooth its first spectrum. The public filter
# records its effective parameters in the resulting dataset's structured history.
dataset = scp.read("irdata/nh4y-activation.spg")
processed = scp.savgol(dataset[0], size=11, order=2)

# %%
# Save a simple plot and a Markdown report beside each other. Figure references
# are relative to the Markdown file, so both files can be moved together.
# Gallery exports are temporary; choose a persistent directory for your own use.
with TemporaryDirectory() as tmpdir:
    report_path = Path(tmpdir) / "ir-report.md"
    figure_path = report_path.with_name("spectrum.png")
    ax = processed.plot(output=figure_path, show=False)
    report = processed.to_markdown_report(
        title="Smoothed NH4Y IR spectrum",
        figures=(
            FigureReference(
                "Smoothed spectrum",
                figure_path.relative_to(report_path.parent).as_posix(),
                "Savitzky-Golay smoothing of the first measured spectrum.",
            ),
        ),
    )
    report_path.write_text(report, encoding="utf-8")
    print(report)
