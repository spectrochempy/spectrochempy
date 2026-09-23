# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   language_info:
#     name: python
#     version: 3.10.8
# ---

# %% [markdown]
# # Cross-validation of supervised models
#
# Cross-validation estimates how one **fixed** model configuration predicts
# observations that were not used to fit it. In each fold, the calibration
# observations fit a fresh estimator and the validation observations are
# predicted once. The predictions from all folds are then restored to the
# original sample order; these are out-of-fold (OOF) predictions.
#
# Any preprocessing that learns from data must be inside a `Pipeline`. It is
# then fitted from the calibration part of each fold, preventing validation
# spectra from leaking into centering, scaling, or reference estimates.
# See the API reference for
# [scp.cross_validate](../../reference/generated/spectrochempy.cross_validate.rst)
# and
# [CrossValidationResult](../../reference/generated/spectrochempy.CrossValidationResult.rst).

# %%
from sklearn.model_selection import KFold

import spectrochempy as scp

sample = scp.Coord.arange(18, title="sample")
wavenumber = scp.Coord.linspace(1000.0, 1200.0, 24, title="wavenumber", units="cm^-1")
concentration = scp.linspace(0.2, 1.8, sample.size).data
band = scp.exp(-0.5 * ((wavenumber.data - 1100.0) / 22.0) ** 2)
noise = scp.normal(scale=0.015, size=(sample.size, wavenumber.size), seed=7).data
spectra = concentration[:, None] * band + noise

X = scp.NDDataset(
    spectra,
    coordset=[sample, wavenumber],
    dims=["y", "x"],
    units="absorbance",
    title="spectra",
)
y = scp.NDDataset(
    concentration,
    coordset=[sample.copy()],
    dims=["y"],
    units="mol/L",
    title="concentration",
)

# %% [markdown]
# ## A fixed validation design
#
# `cv` may be an integer, `KFold`, `GroupKFold`, or `LeaveOneOut`. An integer
# creates an unshuffled `KFold`, or a `GroupKFold` when `groups` is supplied.
# Pass one group identity per observation when experimental batches, subjects,
# or replicate families must stay together. Do not create arbitrary groups to
# make a grouped analysis possible.
#
# The default observation dimension is `"y"`; use `sample_dim` when observations
# use another dimension name. SpectroChemPy checks that X, y, their observation
# coordinates, and any groups are aligned in the same order.

# %%
splitter = KFold(n_splits=6, shuffle=True, random_state=7)
pipeline = scp.Pipeline(
    [
        ("center", scp.CenterTransformer(dim="y")),
        ("pls", scp.PLSRegression(n_components=2, scale=False)),
    ]
)
result = scp.cross_validate(
    pipeline,
    X,
    y,
    cv=splitter,
    metrics=("rmsecv", "r2", "bias", "mae"),
)

print(result.oof_predictions)
print(result.metric("rmsecv").values)

# %% [markdown]
# `CrossValidationResult` contains:
#
# - `observed`, `oof_predictions`, and `residuals`, in the original target
#   geometry, order, coordinates, and units;
# - `metric(name)`, returning values plus per-target `defined` flags and
#   `reasons` for `rmsecv`, `r2`, `bias`, or `mae`;
# - `folds`, whose read-only calibration/validation positions, `n_valid`, and
#   fold metrics make the split auditable;
# - `n_valid`, the number of finite, unmasked observed/predicted pairs per
#   target;
# - snapshots of the estimator and splitter configuration, plus optional
#   independent fitted fold estimators when `return_estimators=True`.
#
# RMSECV, bias, and MAE carry the target units; R² is unitless. Global RMSECV
# is computed directly from all OOF residuals. Equivalently, it is the square
# root of the fold-size-weighted mean of the squared fold RMSE values. It is
# **not** their arithmetic mean, even when every fold has the same size.

# %%
for name in ("rmsecv", "r2", "bias", "mae"):
    metric = result.metric(name)
    value = float(metric.values.data.squeeze())
    units = "" if metric.values.units is None else f" {metric.values.units}"
    print(f"{name}: {value:.4f}{units}; defined={metric.defined[0]}")

print("validation positions:", [fold.validation_positions for fold in result.folds])
print("valid pairs:", int(result.n_valid.data.squeeze()))

# %% [markdown]
# ## Undefined values, masks, and limits
#
# Observed targets containing masks or non-finite values are rejected, as are
# non-finite predictors and predictor masks that vary between observations.
# Features masked for every observation are retained for preprocessing and
# handled consistently by PLS. A masked or non-finite prediction reduces
# `n_valid` and makes the requested metrics for that target explicitly
# undefined; metrics are not silently recomputed on the remaining finite subset.
# `defined`, `reasons`, and `undefined_metrics` expose why. For example, fold R²
# is undefined for a leave-one-out fold because that fold has only one validation
# observation; global OOF R² can still be defined.
#
# The supplied estimator and datasets are not fitted or mutated. Result
# datasets are isolated copies, but remain mutable `NDDataset` objects.
# `cross_validate` does not refit a final model on all observations, select
# hyperparameters, implement nested cross-validation, persist results, or
# capture complete provenance. Selection performed after inspecting these
# scores is subject to selection bias; use a separate design, such as nested
# cross-validation, when model selection itself must be evaluated.
#
# A complete application to real NIR spectra, including OOF predictions and
# parity/residual plots, is available in the
# [Gallery example](../../gettingstarted/examples/gallery/auto_examples_analysis/b_crossdecomposition/plot_cross_validation.rst).
