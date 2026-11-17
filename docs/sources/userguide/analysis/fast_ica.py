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
# ---

# %% [markdown]
# # Independent Component Analysis (FastICA)
#
# FastICA estimates statistically independent sources from linear mixtures.
# Unlike a variance ranking, independence describes the joint distribution
# of the sources. In the standard ICA model, at most one source can be
# Gaussian. Spectral or chemical assignments need separate evidence.
#
# ## Separate two signals
#
# Mix a sine wave and a switching signal in different proportions to
# obtain two observed channels. Orient the matrix passed to `fit()` with
# observations in rows (time samples) and channels in columns. The
# transpose before plotting puts time back on the horizontal axis.

# %%
import numpy as np

import spectrochempy as scp

time = scp.Coord.linspace(0.0, 6.0, 601, title="time", units="s")
smooth = scp.sin(2.0 * np.pi * time.data)
switching = (scp.sin(5.0 * np.pi * time.data) >= 0) * 2.0 - 1.0
mixtures = scp.NDDataset(
    [smooth + 0.6 * switching, 0.4 * smooth + switching],
    coordset=[scp.Coord(labels=["Mixture 1", "Mixture 2"]), time],
    title="relative signal",
)
model = scp.FastICA(n_components=2, random_state=42)
sources = model.fit_transform(mixtures.T).T
sources.title = "relative signal"

# %% [markdown]
# The [short gallery example](../../gettingstarted/examples/gallery/auto_examples_analysis/a_decomposition/plot_fast_ica.rst)
# plots the mixtures and the estimated sources. The source order, sign,
# and scale are ambiguous. For example, changing the sign of one source
# and of its mixing coefficients leaves the mixtures unchanged. Compare
# the shapes with the known signals rather than matching their amplitudes.

# %%
reference = scp.NDDataset([smooth, switching], coordset=[None, time])
_ = scp.multiplot(
    [reference, sources],
    labels=["Known sources", "Estimated sources"],
    ncol=2,
    method="pen",
    figsize=(10, 4),
)

# %% [markdown]
# ## Reconstruct the mixtures
#
# The columns of `model.transform()` are the estimated source signals,
# sampled along the observations. `model.result.St` holds the transposed
# mixing matrix. `inverse_transform()` reconstructs the input layout
# and restores the mean removed during fitting. Inspect the reconstruction
# error when retaining fewer sources than channels. A low error alone
# gives no evidence that the recovered sources are independent.

# %%
reconstructed = model.inverse_transform()
residual = mixtures.T - reconstructed
print(f"Reconstruction RMSE: {float(residual.rms()):.2e}")
