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
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.10.8
# ---

# %% [markdown]
# # Pipeline
#
# ``Pipeline`` composes a small, reproducible sequence of SpectroChemPy
# preprocessing transformers and a final transformer or supervised estimator.
# It is useful when you want to fit preprocessing and a model together, then
# reuse exactly the preprocessing learned from calibration spectra on new
# spectra.
#
# The first public version is intentionally modest. Its goal is not to replace a
# workflow engine or scikit-learn's full ``Pipeline`` API, but to make common
# SpectroChemPy estimator workflows easier to repeat without data leakage:
# preprocessing steps are fitted only when ``Pipeline.fit()`` is called, so test
# spectra do not accidentally influence centering or scaling statistics.

# %%
import numpy as np

import spectrochempy as scp

rng = np.random.default_rng(42)
wavenumbers = scp.Coord.linspace(1000.0, 1200.0, 80, title="wavenumber", units="cm^-1")
concentration = np.linspace(0.1, 1.2, 12)
samples = scp.Coord.arange(concentration.size, title="sample")

band_a = np.exp(-0.5 * ((wavenumbers.data - 1060.0) / 12.0) ** 2)
band_b = np.exp(-0.5 * ((wavenumbers.data - 1140.0) / 18.0) ** 2)
baseline = 0.03 + 0.0005 * (wavenumbers.data - wavenumbers.data.mean())
spectra = (
    baseline
    + concentration[:, None] * band_a
    + 0.35 * concentration[:, None] * band_b
    + rng.normal(scale=0.015, size=(concentration.size, wavenumbers.size))
)

dataset = scp.NDDataset(
    spectra,
    coordset=[samples, wavenumbers],
    dims=["y", "x"],
    units="absorbance",
    title="calibration spectra",
)
target = scp.NDDataset(
    concentration,
    coordset=[samples.copy()],
    dims=["y"],
    units="mol/L",
    title="concentration",
)
_ = dataset.plot(show=False)

# %% [markdown]
# A complete calibration/test example
# -----------------------------------
#
# A typical supervised use is to fit preprocessing and regression together on a
# calibration subset, then apply the fitted pipeline to separate test spectra.
# The scaler below learns its statistics from ``X_cal`` only; those fitted
# statistics are reused when predicting ``X_test``.

# %%
test_indices = [2, 5, 8, 11]
cal_indices = [i for i in range(concentration.size) if i not in test_indices]

X_cal = dataset[cal_indices]
y_cal = target[cal_indices]
X_test = dataset[test_indices]
y_test = target[test_indices]

regression_pipeline = scp.Pipeline(
    [
        ("scale", scp.AutoscaleTransformer(dim="y")),
        ("pls", scp.PLSRegression(n_components=2, scale=False)),
    ]
)

regression_pipeline.fit(X_cal, y_cal)
y_pred = regression_pipeline.predict(X_test)
residuals = y_test - y_pred
rmse = np.sqrt(np.mean(np.asarray(residuals.data) ** 2))

summary = scp.NDDataset(
    np.column_stack([y_test.data, y_pred.data, residuals.data]),
    coordset=[
        y_test.coordset[0].copy(),
        scp.Coord(
            np.arange(3),
            labels=["observed", "predicted", "residual"],
            title="quantity",
        ),
    ],
    dims=["y", "x"],
    units=y_test.units,
    title=f"test predictions, RMSE = {rmse:.3f}",
)
summary

# %% [markdown]
# The fitted final estimator can still be inspected directly. Here we reuse its
# parity-plot helper and add the independent test predictions in red.

# %%
fitted_pls = regression_pipeline.fitted_named_steps_["pls"]
ax = fitted_pls.plot_parity(label="calibration", s=120, show=False)
_ = fitted_pls.plot_parity(
    y_test,
    y_pred,
    ax=ax,
    s=120,
    c="red",
    label="test",
    clear=False,
    show=False,
)
_ = ax.legend(loc="lower right")

# %% [markdown]
# Transformer-final pipelines
# ---------------------------
#
# A transformer-final pipeline ends with a preprocessing transformer or with
# ``PCA``. ``fit_transform(X)`` is equivalent to ``fit(X).transform(X)``.

# %%
pca_pipeline = scp.Pipeline(
    [
        ("center", scp.CenterTransformer(dim="y")),
        ("pca", scp.PCA(n_components=3)),
    ]
)
scores = pca_pipeline.fit_transform(dataset)
scores

# %% [markdown]
# Template and fitted steps
# -------------------------
#
# The steps passed to ``Pipeline`` are templates. Calling ``fit()`` clones those
# templates, fits the clones, and leaves the original step objects unchanged.
# Template steps remain available through ``steps`` and ``named_steps``.
# Learned runtime state is available after fitting through ``fitted_steps_``
# and ``fitted_named_steps_``.

# %%
print(
    regression_pipeline.named_steps["scale"]
    is regression_pipeline.fitted_named_steps_["scale"]
)
print(regression_pipeline.steps[0][1] is regression_pipeline.fitted_steps_[0][1])

# %% [markdown]
# Nested parameters
# -----------------
#
# Template parameters are visible and editable with ``step__parameter`` names,
# using the step name followed by a double underscore and the parameter name.
# Any effective change invalidates the fitted state, so you must call
# ``fit()`` again before using ``predict()`` or ``transform()``.

# %%
regression_pipeline.get_params(deep=True)["pls__n_components"]
regression_pipeline.set_params(pls__n_components=1)
regression_pipeline.fit(X_cal, y_cal)
regression_pipeline.predict(X_test)

# %% [markdown]
# What Pipeline v1 does and does not do
# -------------------------------------
#
# ``Pipeline`` is deliberately linear. Steps are ordered ``(name, step)`` pairs,
# intermediate steps must transform an ``NDDataset`` into another
# ``NDDataset``, and the final step is either a transformer or a supervised
# estimator. A final transformer supports ``transform()`` and
# ``fit_transform()``. A final supervised estimator supports ``predict()`` and,
# when available on that estimator, ``score()``.
#
# ``Pipeline`` does not choose train/test splits, run cross-validation, search
# hyperparameters, cache intermediate results, branch into multiple paths, or
# route arbitrary fit parameters. For cross-validation, the split loop must fit
# a fresh pipeline inside each fold so that every preprocessing statistic stays
# fold-local.

# %% [markdown]
# Supported v1 classes
# --------------------
#
# Intermediate positions accept ``CenterTransformer``, ``AutoscaleTransformer``,
# ``ParetoScaleTransformer``, ``RangeScaleTransformer``,
# ``RobustScaleTransformer``, ``SNVTransformer``, ``NormalizeTransformer``,
# ``MSCTransformer`` and ``LogTransformer``.
#
# Final positions accept those transformers, ``PCA``, ``PLSRegression``,
# ``LSTSQ`` and ``NNLS``.
#
# Version 1 intentionally excludes ``Baseline``, ``SVD``, ``PSD``, ``Filter``,
# functional processing wrappers, ``MCRALS``, ``Optimize``, optional steps,
# branching, caching, persistence guarantees and arbitrary fit-parameter
# routing.
