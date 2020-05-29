# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Peak finding
#
# This tutorial shows how to find peaks and determine peak maxima with spectrochempy. As prerequisite, the user is
# expected to have read the [Import](../IO/import.ipynb), [Import IR](../IO/importIR.ipynb),  [slicing](../process/slicing.ipynb) tutorials. 
#
# First lets import the modules that will be used in this tutorial

# %%
import spectrochempy as scp           
import matplotlib.pyplot as plt  # will be used for some plots

# %% [markdown]
# and import and plot a typical IR dataset (CO adsorption on supported CoMo catalyst in the 2300-1900 cm-1 region) that will be used throughout the tutorial:

# %%
# %matplotlib qt
X = scp.read_omnic("irdata/CO@Mo_Al2O3.SPG")[:,2300.0:1900.0]
subplot1 = X.plot()   # colormap and linewdth can be changed for better visibility cmap='Dark2'

# %% [markdown]
# ## 1 find maxima with a direct method
#
# Maxima of a given spectrum can be found by find_peaks() which is based on [scpy.signal.find_peaks()]() . It returns two outputs: a NDDataset grouping the the peak maxima (wavenumbers and absorbances) and A dictionary containing properties of the returned peaks (empty if no particula option is selected). Applying this method on the last spectrum without any option will give the following results:

# %%
peaks, properties = X[-1].find_peaks(thre)
print(peaks.x.data)
print(peaks.data)

# %% [markdown]
# ## 2 find extrema with derivatives 

# %%
plot = X.savgol_filter(window_length = 31, polyorder=2, deriv=2).plot()

# %%
deriv = -1 * X.savgol_filter(window_length = 31, polyorder=2, deriv=2)
#peaks, prop = deriv[-1].find_peaks(height=7e-5)
#peaks.x[0].data[0]
