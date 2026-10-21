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
it does not analyse data, capture figures, or establish complete provenance.

This example smooths a synthetic spectrum, then renders selected metadata and
the processing history. The caller chooses which metadata are safe to include.
"""

import numpy as np

import spectrochempy as scp

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
# Render the report. Metadata are opt-in, and no figure is captured implicitly.
report = processed.to_markdown_report(
    title="Preprocessed polymer-film spectrum",
    metadata=("sample",),
)
print(report)
