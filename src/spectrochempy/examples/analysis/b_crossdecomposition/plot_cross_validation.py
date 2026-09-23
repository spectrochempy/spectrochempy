# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
"""
Cross-validation of corn moisture models
========================================

This example evaluates fixed PLS configurations on real near-infrared corn
spectra. It shows out-of-fold predictions, global metrics, and fold-local
preprocessing without using the validation observations during fitting.

The public `Eigenvector Corn data
<https://www.eigenvector.com/Docs/EigenNews_3.html>`_ contain 80 corn samples
measured on three instruments, with reference values for moisture, oil,
protein, and starch. The measurements were made at Cargill and are distributed
by Eigenvector with permission. Here we use only the M5 spectra and moisture.
"""

# %%
# Import packages
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold

import spectrochempy as scp

# %%
# Load and identify the data
# --------------------------
# The remote archive is an existing SpectroChemPy example dependency. A failed
# download raises normally, so a documentation build cannot silently publish a
# partially executed analysis.
datasets = scp.read("https://www.eigenvector.com/data/Corn/corn.mat", merge=False)
X = next(dataset for dataset in datasets if dataset.name == "m5spec").copy()
properties = next(dataset for dataset in datasets if dataset.name == "propvals").copy()
y = properties[:, "Moisture"].squeeze()

X.title = "NIR absorbance"
X.x.title = "Wavelength"
X.x.units = "nm"
X.y.title = "sample"
y.title = "Moisture content"
y.units = "%"
y.y.title = "sample"

print(f"Spectra: {X.shape}, dimensions {tuple(X.dims)}")
print(f"Target: {y.shape}, dimensions {tuple(y.dims)}")
print(f"Wavelength range: {X.x.data[0]:.0f}–{X.x.data[-1]:.0f} {X.x.units}")
print(f"Moisture range: {y.data.min():.3f}–{y.data.max():.3f} {y.units}")

ax = X[::8].plot(cmap="viridis", show=False)
_ = ax.set_title("Every eighth M5 corn spectrum")

# %%
# Define one validation design
# ----------------------------
# There is one M5 spectrum per sample and the distributed file does not expose
# a batch or replicate grouping. We therefore use a reproducible shuffled
# K-fold design, without inventing groups. A domain analysis should use grouped
# cross-validation when its experimental design identifies dependent samples.
#
# Five PLS components match the established fixed configuration used in the
# SpectroChemPy corn tutorial. They are fixed before this comparison and are
# not selected from the scores below.
splitter = KFold(n_splits=5, shuffle=True, random_state=7)
metrics = ("rmsecv", "r2", "bias", "mae")

pls = scp.PLSRegression(n_components=5)
pls_result = scp.cross_validate(pls, X, y, cv=splitter, metrics=metrics)

# %%
# Add fold-local scatter correction
# ---------------------------------
# NIR spectra commonly exhibit multiplicative scatter. With no explicit
# reference, ``MSCTransformer`` learns the mean reference spectrum during
# ``fit``. Keeping it inside the Pipeline therefore learns a different
# reference from each fold's calibration spectra only; there is no global MSC
# reference and no validation leakage.
msc_pls = scp.Pipeline(
    [
        ("msc", scp.MSCTransformer(dim="y")),
        ("pls", scp.PLSRegression(n_components=5)),
    ]
)
msc_result = scp.cross_validate(msc_pls, X, y, cv=splitter, metrics=metrics)


def _metric_value(result, name):
    return float(np.asarray(result.metric(name).values.data).squeeze())


# %%
# Inspect global OOF metrics
# --------------------------
# RMSECV, bias, and MAE retain the moisture unit; R² is unitless. Global
# RMSECV is calculated from all OOF residuals together: it is the square root
# of the fold-size-weighted mean of squared fold RMSE values, not their
# arithmetic mean. The comparison is descriptive: this example does not claim
# that MSC should improve this dataset or use the displayed scores to select a
# model.
print("model            RMSECV (%)      R2     bias (%)      MAE (%)")
for label, result in (("PLS", pls_result), ("MSC + PLS", msc_result)):
    print(
        f"{label:12s} "
        f"{_metric_value(result, 'rmsecv'):12.4f} "
        f"{_metric_value(result, 'r2'):7.4f} "
        f"{_metric_value(result, 'bias'):12.4f} "
        f"{_metric_value(result, 'mae'):12.4f}"
    )

# %%
# Plot parity and residuals
# -------------------------
# Both result objects preserve the original sample order and target geometry.
observed = np.asarray(pls_result.observed.data).squeeze()
pls_predicted = np.asarray(pls_result.oof_predictions.data).squeeze()
msc_predicted = np.asarray(msc_result.oof_predictions.data).squeeze()

limits = [
    min(observed.min(), pls_predicted.min(), msc_predicted.min()),
    max(observed.max(), pls_predicted.max(), msc_predicted.max()),
]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
axes[0].plot(limits, limits, color="0.35", linestyle="--", label="identity")
axes[0].scatter(observed, pls_predicted, label="PLS", alpha=0.8)
axes[0].scatter(observed, msc_predicted, label="MSC + PLS", alpha=0.8)
axes[0].set(
    xlabel="Reference moisture (%)",
    ylabel="OOF-predicted moisture (%)",
    title="Out-of-fold parity",
)
axes[0].legend()

axes[1].axhline(0.0, color="0.35", linestyle="--")
axes[1].scatter(observed, observed - pls_predicted, label="PLS", alpha=0.8)
axes[1].scatter(observed, observed - msc_predicted, label="MSC + PLS", alpha=0.8)
axes[1].set(
    xlabel="Reference moisture (%)",
    ylabel="OOF residual (%)",
    title="Out-of-fold residuals",
)
axes[1].legend()

# %%
# The template estimators remain unfitted. ``cross_validate`` fits independent
# clones inside each fold and does not perform a final fit on all 80 samples.
print(f"Supplied PLS fitted: {pls._fitted}")
print(f"Supplied Pipeline fitted: {msc_pls._fitted}")
print(f"OOF sample order preserved: {pls_result.observed.y == y.y}")

# %%
# Uncomment the following line to display all figures when running the script
# directly with Python.

# scp.show()
