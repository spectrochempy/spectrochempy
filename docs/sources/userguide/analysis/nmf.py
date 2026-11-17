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
# # Non-negative matrix factorization (NMF)
#
# NMF approximates a matrix of spectra as $X \approx WH$. Each row of $H$
# is a component spectrum. The corresponding column of $W$ gives its
# weight in each observation. Both matrices contain non-negative values.
# This model is useful when additive spectral contributions are plausible,
# such as absorbance bands whose amplitudes change across a series.
#
# ```{=rst}
# See :cite:t:`lee-seung:1999` for the non-negative factorization model.
# ```

# %% [markdown]
# ## Prepare an infrared series
#
# Use a 2D dataset with observations in rows and spectral channels in
# columns. NMF expects non-negative input. Inspect the baseline and
# exclude unusable regions before fitting. Mean centering can introduce
# negative intensities.
#
# For this NH₄Y series, take every fifth spectrum and use a window above
# the saturated region. Thin the spectral grid to keep the example quick.
# The retained signal has a small negative minimum, so shift every point
# by the same constant. The shift changes the matrix being fitted and can
# influence the recovered components. For measured data, check the origin
# of negative values before choosing how to handle them.

# %%
import spectrochempy as scp

spectra = scp.read_omnic("irdata/nh4y-activation.spg")[::5, 4000.0:1280.0:3]
offset = min(0.0, float(spectra.data.min()))
nonnegative = spectra - offset

# %% [markdown]
# ## Compare two component counts
#
# Compare two and four components on the same preprocessed matrix.
# Use the same initialization strategy for both fits. The reported
# reconstruction error measures the discrepancy between the input and
# the fitted product. A lower error shows that the model can reproduce
# the input more closely. Look at residual structure and whether the
# estimated profiles are stable before deciding how many factors to keep.

# %%
two = scp.NMF(n_components=2, init="nndsvda", random_state=0, max_iter=1500)
four = scp.NMF(n_components=4, init="nndsvda", random_state=0, max_iter=1500)
_ = two.fit(nonnegative)
_ = four.fit(nonnegative)

result_two = two.result
result_four = four.result
print(f"Two components: reconstruction error = {result_two.reconstruction_error:.3g}")
print(f"Four components: reconstruction error = {result_four.reconstruction_error:.3g}")

# %% [markdown]
# ## Read the spectra, weights, and residuals
#
# For the four-component model, `result.components` is $H$ and
# `result.W` is $W$. Compare component spectra with known bands and look
# at how their weights change across the observations. Different
# factorizations can reconstruct similar data, and rescaling a component
# can be compensated by rescaling its weights. Treat the weights as
# relative contributions unless you have established a calibration.

# %%
components = result_four.components
weights = result_four.W
_ = components.plot(method="pen", title="Component spectra", legend=components.k.labels)
_ = weights.T.plot(method="pen", title="Component weights", legend=weights.k.labels)

# %% [markdown]
# Reconstruct the fitted matrix and restore the offset before comparing
# it with the original signal. Compare residuals at the same observation.
# Here the four-component model reduces the error of the two-component
# model. Assigning a chemical meaning to the factors requires checking
# the recovered bands and their evolution.

# %%
reconstructed_two = two.inverse_transform(result_two.W) + offset
reconstructed_four = four.inverse_transform(weights) + offset
residual_two = spectra - reconstructed_two
residual_four = spectra - reconstructed_four
_ = scp.plot_multiple(
    datasets=[residual_two[0], residual_four[0]],
    labels=["Two components", "Four components"],
    legend="best",
    title="Residuals for the first retained spectrum",
)

# %% [markdown]
# The [short gallery example](../../gettingstarted/examples/gallery/auto_examples_analysis/a_decomposition/plot_nmf.rst)
# fits two non-negative bands and plots the component spectra. See
# the [NMF API](../../reference/generated/spectrochempy.NMF.rst) for fit options.
