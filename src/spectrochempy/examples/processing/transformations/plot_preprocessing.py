# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: D205, D400, D415
"""
Compare IR spectra after normalization
======================================

Scale each spectrum by its own maximum absolute absorbance in a selected
region. Compare relative band intensities across the activation series.
See the :doc:`transformations tutorial </userguide/processing/transformations>`
for other methods and how to choose the axis.
"""

import spectrochempy as scp

series = scp.read_omnic("irdata/nh4y-activation.spg")
spectra = series[::10, 4000.0:2000.0]
normalized = spectra.normalize(method="max", dim="x")
normalized.title = "relative absorbance"
normalized.units = None

_ = scp.multiplot(
    [spectra, normalized],
    method="lines",
    ncol=2,
    labels=["Measured absorbance", "Max-normalized spectra"],
    x_reverse=True,
    figsize=(10, 4),
)
