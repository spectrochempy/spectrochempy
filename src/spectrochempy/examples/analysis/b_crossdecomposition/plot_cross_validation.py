# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: D205, D400, D415
"""
Validate a PLS calibration on corn spectra
==========================================

Predict moisture for each corn sample from a model trained on other samples.
See the :doc:`cross-validation tutorial </userguide/analysis/cross_validation>`
for split design, error metrics, and preprocessing inside each fold.
"""

import spectrochempy as scp

datasets = scp.read("https://eigenvector.com/data/Corn/corn.mat", merge=False)
by_name = {dataset.name: dataset for dataset in datasets}
spectra = by_name["m5spec"]
moisture = by_name["propvals"][:, "Moisture"].squeeze()
moisture.title = "Moisture content"
moisture.units = "%"

# %%
# A subset of the measured NIR spectra shows the variation across corn
# samples. Each spectrum has one reference moisture value.

_ = spectra[::10].plot(
    method="lines",
    title="Measured corn NIR spectra",
    xlabel="Wavelength / nm",
    ylabel="NIR signal",
)

# %%
# Predict each sample using a model fitted without that sample.

model = scp.PLSRegression(n_components=5)
splitter = scp.KFold(n_splits=5, shuffle=True, random_state=7)
result = scp.cross_validate(model, spectra, moisture, cv=splitter)
_ = model.plot_parity(result.observed, result.oof_predictions, label="Out-of-fold")
