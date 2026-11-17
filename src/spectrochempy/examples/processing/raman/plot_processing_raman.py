# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: D205, D400, D415
"""
Preprocess a Raman activation series
====================================

Correct the baseline of a measured Raman series with SNIP, then denoise
the corrected spectra by PCA. Compare the same acquisitions before and
after processing. See the :doc:`baseline tutorial </userguide/processing/baseline>`
and the :doc:`denoising tutorial </userguide/processing/denoising>` for
parameter choices and checks.
"""

import spectrochempy as scp

spectra = scp.read_labspec("ramandata/labspec/Activation.txt")
corrected = spectra.snip(snip_width=55)
processed = corrected.denoise(ratio=98)
processed.title = spectra.title

_ = scp.multiplot(
    [spectra[::10], processed[::10]],
    method="lines",
    ncol=2,
    labels=["Measured spectra", "After SNIP and PCA"],
    figsize=(10, 4),
)
