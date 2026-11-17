# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: D205, D400, D415
"""
Denoise a Raman series with PCA
===============================

Reconstruct a series of Raman spectra from the principal components that
explain at least 98% of its variance. Compare the same acquisitions before
and after reconstruction in a region around two Raman bands. See the :doc:`denoising tutorial
</userguide/processing/denoising>` for choosing the retained variance and
handling cosmic ray spikes.
"""

import spectrochempy as scp

series = scp.read("ramandata/labspec/serie190214-1.txt")
spectra = series[0.0:6500.0, 70.0:].snip()
reconstructed = spectra.denoise(ratio=98)
reconstructed.title = spectra.title

_ = scp.multiplot(
    [spectra[::8, 500.0:760.0], reconstructed[::8, 500.0:760.0]],
    method="lines",
    ncol=2,
    labels=["Before PCA", "After PCA"],
    figsize=(10, 4),
)
