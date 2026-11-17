# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: D205, D400, D415
"""
Autoscale held-out spectra with training statistics
===================================================

Fit an autoscaling transformer to the first 40 spectra in an IR series,
then apply the same scaling to the remaining spectra. Compare the held-out spectra before and after scaling with the training
mean and standard deviation at each wavenumber. See the :doc:`transformations tutorial
</userguide/processing/transformations>` for use in model validation.
"""

import spectrochempy as scp

series = scp.read_omnic("irdata/nh4y-activation.spg")
spectra = series[:, 4000.0:2000.0]
train, held_out = spectra[:40], spectra[40:]

scaler = scp.AutoscaleTransformer(dim="y")
_ = scaler.fit(train)
scaled = scaler.transform(held_out)
scaled.title = "standardized intensity"
scaled.units = None

_ = scp.multiplot(
    [held_out, scaled],
    method="lines",
    ncol=2,
    labels=["Measured held-out spectra", "After scaling with training statistics"],
    x_reverse=True,
    figsize=(10, 4),
)
