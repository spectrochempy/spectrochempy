# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: title,-all
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
#     version: 3.13.2
# ---

# %% [markdown]
# # Denoising

# %%
import spectrochempy as scp

# %% [markdown]
# ## Denoising 2D spectra
# Denoising 2D spectra can be done using the above filtering techniques which can be applied sequentially
# to each rows of a 2D dataset.
#
# e.g., let's take a series of Raman spectra for demonstration: These spectra present both a significant noise and cosmic rays spikes.

# %%
# Load the data
dataset = scp.read("ramandata/labspec/serie190214-1.txt")
# select the useful region (in particular spectra are 0 after 6500 s)
nd = dataset[0.0:6500.0, 70.0:]
# baseline correction the data (for a easier comparison)
nd1 = nd.snip()
# plot
prefs = scp.preferences
prefs.figure.figsize = (9, 5)
_ = nd1.plot()

# %% [markdown]
# We can apply a Savgol filter to denoise the spectra

# %%
nd2 = nd1.savgol(size=7, order=2)
_ = nd2.plot()

# %% [markdown]
# The problem is that, not only the spikes are not removed, but they are also broadened.

# %% [markdown]
# ### Reconstruct a series with PCA
#
# The `denoise()` method fits PCA to a 2D dataset of spectra. It reconstructs
# the series using enough principal components to meet the explained variance
# target set by `ratio`, expressed as a percentage (99.8 by default).
#
# A lower ratio can remove more noise and can also erase weak bands or
# changes that occur in only a few acquisitions. Compare the same spectra
# before and after reconstruction, checking peak shapes and trends across
# the series before using the denoised data for quantitative analysis.

# %%
nd3 = nd1.denoise(ratio=90)
_ = nd3.plot()

# %% [markdown]
# Here PCA reduces the noise but some cosmic ray spikes are still visible.
# Use the dedicated despiking methods below to target those isolated spikes.

# %% [markdown]
# ## Removing cosmic rays spike from Raman spectra

# %% [markdown]
# ### Median filter

# %% [markdown]
# A first way to perform this is to apply a median-filter to the data

# %%
filter = scp.Filter(method="median", size=5)
nd4 = filter(nd1)
_ = nd4.plot()

# %% [markdown]
# However, the spike are not fully removed, and are broadened.

# %% [markdown]
# ### `despike` method
#
# ```{=rst}
# The default ``katsumoto`` method (:cite:t:`katsumoto:2003`) detects
# sharp deviations of either sign from a locally smoothed spectrum. The
# ``whitaker`` method (:cite:t:`whitaker:2018`) detects outliers in
# differences between adjacent samples.
# ```
#
# For these methods, `delta` sets the detection threshold and `size`
# sets the window length in samples. Start with a conservative threshold
# and check the result against the input, especially near narrow bands.
#
# When a spike affects one acquisition in an ordered series, use
# `series.despike(method="neighbors", dim="y")` if `y` is the acquisition
# axis. This method compares each intensity with the two neighboring
# acquisitions at the same wavenumber. It uses `delta` to detect positive
# excursions above both neighbors and replaces them with the neighbors' mean.
# Inspect rapid changes across the series before accepting the correction.
#
# %%
X = nd1[0]
nd5 = scp.despike(X, size=11, delta=5)
_ = X.plot()
_ = nd5.plot(clear=False, ls="-", c="r")

# %% [markdown]
# Getting the desired results require the tuning of size and delta parameters. And sometimes may need to repeat the procedure on a previously filtered spectra.
#
# For example, it size or delta are badly chosen, valid peaks could be removed. So careful inspection of the results is crucial.

# %%
nd5b = scp.despike(X, size=21, delta=2)
_ = X.plot()
_ = nd5b.plot(clear=False, ls="-", c="r")

# %% [markdown]
# Last we can apply it to the full 2D dataset

# %%
nd6 = scp.despike(nd1, size=11, delta=5)
_ = nd6.plot()

# %% [markdown]
# It is however rarely perfect as the setting of size and delta may be depending on the row.
#
# A possibility to improve it is to apply a denoise filter afterward.

# %%
nd7 = nd6.denoise(ratio=92)
_ = nd7.plot()

# %% [markdown]
# The 'whitaker' method is also available:
# %%
nd8 = scp.despike(nd1, size=11, delta=5, method="whitaker")
_ = nd8.plot()
