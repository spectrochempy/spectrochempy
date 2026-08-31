# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
"""
PLS regression with a Pipeline
==============================

This example predicts corn moisture from NIR spectra using a SpectroChemPy
``Pipeline``. The pipeline learns preprocessing from the calibration spectra
only, then reuses the fitted preprocessing when predicting validation spectra.
"""

# %%
# Import the packages
import numpy as np

import spectrochempy as scp

# %%
# Load the corn NIR dataset
# -------------------------
# The data is available from the Eigenvector archive:
try:
    ds_list = scp.read("http://www.eigenvector.com/data/Corn/corn.mat", merge=False)
except FileNotFoundError:
    ds_list = None
    print("Eigenvector corn dataset not reachable; skipping the remote Pipeline example.")
else:
    ds_list_names = [f"{i} : {ds.name}({ds.shape})" for i, ds in enumerate(ds_list)]
    print(ds_list_names)

# %%
if ds_list is not None:
    # %%
    # Inspect the spectra and select moisture
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # The 5th dataset ``m5spec`` contains NIR spectra from 80 corn samples
    # recorded on the same instrument. The properties to predict are in the
    # ``propval`` dataset.
    X = ds_list[4]
    X.title = "reflectance"
    X.x.title = "Wavelength"
    X.x.units = "nm"
    _ = X.plot(cmap=None, show=False)

    Y = ds_list[3]
    y = Y[:, "Moisture"]

    # %%
    # Split calibration and validation samples
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Use regularly spaced samples after sorting by moisture so the validation
    # set spans the concentration range instead of becoming a simple high-index
    # holdout.
    moisture_order = np.argsort(np.asarray(y.data).ravel())
    validation_indices = sorted(moisture_order[3::4].tolist())
    calibration_indices = [i for i in range(X.shape[0]) if i not in validation_indices]

    X_cal = X[calibration_indices]
    y_cal = y[calibration_indices]
    X_val = X[validation_indices]
    y_val = y[validation_indices]

    # %%
    # Fit preprocessing and PLS together
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # ``AutoscaleTransformer`` owns the centering/scaling step, so the internal
    # scaling of ``PLSRegression`` is disabled to keep responsibilities clear.
    pipeline = scp.Pipeline(
        [
            ("scale", scp.AutoscaleTransformer(dim="y")),
            ("pls", scp.PLSRegression(n_components=5, scale=False)),
        ]
    )
    pipeline.fit(X_cal, y_cal)

    # %%
    # Predict validation samples
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^
    y_val_pred = pipeline.predict(X_val)
    residuals = y_val - y_val_pred
    rmse_val = np.sqrt(np.mean(np.asarray(residuals.data) ** 2))
    units = f" {y_val.units}" if y_val.units else ""
    print(f"Validation RMSE: {rmse_val:.3f}{units}")

    # %%
    # Validate with a parity plot
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^
    fitted_pls = pipeline.fitted_named_steps_["pls"]
    ax = fitted_pls.plot_parity(label="calibration", s=150, show=False)
    _ = fitted_pls.plot_parity(
        y_val,
        y_val_pred,
        ax=ax,
        s=150,
        c="red",
        label="validation",
        clear=False,
        show=False,
    )
    _ = ax.legend(loc="lower right")
    _ = ax.set_title(f"Corn moisture prediction, RMSE = {rmse_val:.3f}{units}")

    # %%
    # Template and fitted steps
    # ^^^^^^^^^^^^^^^^^^^^^^^^^
    # The public ``steps`` and ``named_steps`` are templates. The fitted scaler
    # used for validation is a fitted clone.
    print(pipeline.named_steps["scale"] is pipeline.fitted_named_steps_["scale"])

    # %%
    # Updating a nested parameter
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Any effective ``set_params()`` call invalidates fitted state. Fit again
    # before predicting with the updated configuration.
    pipeline.set_params(pls__n_components=4)
    pipeline.fit(X_cal, y_cal)
    _ = pipeline.predict(X_val)

# %%
# Uncomment the following line to display all figures when running the script
# directly with Python.

# %%

# scp.show()
