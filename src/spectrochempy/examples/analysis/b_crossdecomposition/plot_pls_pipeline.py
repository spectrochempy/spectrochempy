# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: D205, D400, D415
"""
Predict corn moisture with a PLS pipeline
=========================================

Fit autoscaling and PLS on 57 corn samples, then predict moisture for the
remaining 23. See the :doc:`pipeline tutorial </userguide/analysis/pipeline>`
for fitted steps and how preprocessing is reused at prediction time.
"""

import spectrochempy as scp

datasets = scp.read("https://eigenvector.com/data/Corn/corn.mat", merge=False)
by_name = {dataset.name: dataset for dataset in datasets}
spectra = by_name["m5spec"]
moisture = by_name["propvals"][:, "Moisture"]
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
# Fit autoscaling and PLS on the training samples.

pipeline = scp.Pipeline(
    [
        ("scale", scp.AutoscaleTransformer(dim="y")),
        ("pls", scp.PLSRegression(n_components=5, scale=False)),
    ]
)
_ = pipeline.fit(spectra[:57], moisture[:57])
predicted = pipeline.predict(spectra[57:])
pls = pipeline.fitted_named_steps_["pls"]
_ = pls.plot_parity(moisture[57:], predicted, label="Held-out samples")
