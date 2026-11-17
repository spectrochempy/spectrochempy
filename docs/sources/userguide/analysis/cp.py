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
# # CP/PARAFAC decomposition of spectral series
#
# Suppose you record a time series of spectra for each sample on the same
# wavenumber grid. Organize the data as a three-dimensional
# `NDDataset` with sample, time, and wavenumber axes. Use CP/PARAFAC when
# you expect each contribution to have one profile along each of these axes.
# For example, the same spectral band can decay with the same time profile
# in every sample while its initial amplitude varies across samples.
#
# [Install the Tensor plugin](../plugins/tensor.rst#installation) to use
# `scp.tensor.CP`. The estimator expects a three-dimensional dataset.
# Align the time and spectral grids before assembling the tensor, and
# treat baseline shifts or changing peak positions before assigning
# component profiles to chemical species.

# %% [markdown]
# ## Build a dataset with two contributions
#
# The model approximates the signal at sample $i$, time $j$, and wavenumber
# $k$ as a sum of $R$ contributions:
#
# $$
# \widehat{X}_{ijk} = \sum_{r=1}^{R} w_r A_{ir} B_{jr} C_{kr}.
# $$
#
# Here $A$ describes sample amplitudes, $B$ describes time profiles, $C$
# describes spectral profiles, and $w_r$ is the weight of component $r$.
# We know the rank of the synthetic data because we construct it from two
# distinct bands with different sample and time profiles.
#
# ```{=rst}
# See :cite:t:`kolda:2009` for the CP model and its scaling and permutation properties.
# ```

# %%
import spectrochempy as scp

samples = scp.Coord([1, 2, 3, 4], title="sample")
time = scp.Coord.linspace(0.0, 10.0, 16, title="time", units="s")
wavenumber = scp.Coord.linspace(1800.0, 1400.0, 101, title="wavenumber", units="cm^-1")
band_a = scp.gaussian(wavenumber.data, pos=1660.0, width=40.0, normalized=False)
band_b = scp.gaussian(wavenumber.data, pos=1530.0, width=30.0, normalized=False)
decay_a = 1.0 / (1.0 + time.data)
decay_b = 1.0 / (1.0 + time.data / 5.0)

contribution_a = (
    samples.data[:, None, None] * decay_a[None, :, None] * band_a[None, None, :]
)
contribution_b = (
    samples.data[::-1, None, None] * decay_b[None, :, None] * band_b[None, None, :]
)
spectra = scp.NDDataset(
    contribution_a + contribution_b,
    coordset=[samples, time, wavenumber],
    title="absorbance",
)

# %% [markdown]
# ## Compare two ranks
#
# Set the candidate rank with `n_components`. Fit both models to the same
# tensor and compare their sum of squared reconstruction errors (SSE).
# Since this dataset was generated without noise, the two-component fit
# should reconstruct it much more closely than the one-component fit.

# %%
rank_one = scp.tensor.CP(n_components=1, random_state=0)
_ = rank_one.fit(spectra)
rank_two = scp.tensor.CP(n_components=2, random_state=0)
_ = rank_two.fit(spectra)

print(f"Rank 1 SSE: {rank_one.result.SSE:.3g}")
print(f"Rank 2 SSE: {rank_two.result.SSE:.3g}")

# %% [markdown]
# For measured data, inspect how much the residuals improve as the rank
# increases. Prefer a rank whose profiles are stable across initializations
# and whose contributions make sense for the experiment. A newly resolved
# band in the residual can justify another component. A lower error on its
# own is insufficient to assign a new chemical species.

# %% [markdown]
# ## Interpret the factors and check the reconstruction
#
# Each factor has one row per position on its original axis and one column
# per component. Read the same column across all three factors to see a
# component's sample, time, and spectral profiles. The column order, signs,
# and scales can vary between equivalent fits. Interpret a contribution
# through the product of its factors and weight, and use the reconstructed
# signal when comparing to measured absorbance.

# %%
sample_factor, time_factor, spectral_factor = rank_two.result.factors
weights = rank_two.result.weights
print("Factor shapes:", sample_factor.shape, time_factor.shape, spectral_factor.shape)
print(f"Component weights: {weights}")
_ = scp.plot_multiple(
    datasets=[spectral_factor[:, 0], spectral_factor[:, 1]],
    labels=["Component 1", "Component 2"],
    legend="best",
)

# %% [markdown]
# Reconstruct both models with `inverse_transform()` and compare a spectrum
# from the same sample and time in each tensor. Here the rank-one model
# compromises between bands because it has a single spectral profile for
# the entire tensor. In a real analysis, inspect several samples and times
# to see where the residual structure persists.

# %%
reconstructed_one = rank_one.inverse_transform()
reconstructed_two = rank_two.inverse_transform()
_ = scp.plot_multiple(
    datasets=[spectra[0, 0], reconstructed_one[0, 0], reconstructed_two[0, 0]],
    labels=["Original", "Rank 1", "Rank 2"],
    legend="best",
)

# %% [markdown]
# The [short CP/PARAFAC gallery example](../../gettingstarted/examples/gallery/auto_examples_analysis/a_decomposition/plot_tensor_cp.rst)
# shows one fit and one reconstruction plot. See the
# [CP API reference](../../reference/generated/spectrochempy.tensor.CP.rst)
# for initialization, constraints, and fit diagnostics.
