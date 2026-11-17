# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: D205, D400, D415
"""
Predict corn moisture from NIR spectra
======================================

Fit a PLS regression on 57 corn samples and compare the predictions for the
remaining 23 samples with their measured moisture values.
See the :doc:`PLS tutorial </userguide/analysis/pls>` for the dataset,
the calibration split, and how to interpret the parity plot.
"""

import spectrochempy as scp

datasets = scp.read("https://eigenvector.com/data/Corn/corn.mat", merge=False)
by_name = {dataset.name: dataset for dataset in datasets}
spectra = by_name["m5spec"]
moisture = by_name["propvals"][:, "Moisture"]

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
# Fit the calibration and compare held-out predictions with measured moisture.

model = scp.PLSRegression(n_components=5)
_ = model.fit(spectra[:57], moisture[:57])
predicted = model.predict(spectra[57:])
_ = model.plot_parity(moisture[57:], predicted, label="Held-out samples")
