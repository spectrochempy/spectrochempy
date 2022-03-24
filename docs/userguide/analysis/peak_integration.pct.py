# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Peak integration
#
# This tutorial shows how to find peak maxima and determine peak areas with spectrochempy. As prerequisite,
# the user is expected to have read the [Import](../importexport/import.ipynb),
# [Import IR](../importexport/importIR.ipynb),
# [slicing](../processing/slicing.ipynb) and [baseline correction](../processing/baseline.ipynb) tutorials.

# %% [markdown]
# First lets import the SpectroChemPy API

# %%
import spectrochempy as scp

# %% [markdown]
# Now import some 2D data into a NDDataset object

# %%
ds = scp.read_omnic("irdata/nh4y-activation.spg")
ds

# %% [markdown]
# It's a series of 55 spectra.
#
# For the demonstration select only the first 20 on a limited region from 1250 to 1800 cm$^{-1}$ (Do not forget to
# use floating numbers for slicing)

# %%
X = ds[:20, 1250.0:1800.0]

# %% [markdown]
# We can also eventually remove offset on the acquisition time dimension (y)

# %%
X.y -= X.y[0]
X.y.ito("min")
X.y.title = "acquisition time"

# %% [markdown]
# We set some plotting preferences and then plot the raw data

# %%
prefs = X.preferences
prefs.figure.figsize = (6, 3)
prefs.colormap = "Dark2"
prefs.colorbar = True
X.plot()

# %% [markdown]
# Now we can perform some baseline correction

# %%
blc = scp.BaselineCorrection(X)
regions = (
    [1740.0, 1800.0],
    [1550.0, 1570.0],
    [1250.0, 1300.0],
)  # define 3 regions where we want the baseline to reach zero.
Xcorr = blc.compute(*regions)  # compute the corrected NDDataset

Xcorr.plot()

# %% [markdown]
# To integrate each row on the full range, we can use the sum or trapz method of a NDDataset.

# %%
inttrapz = Xcorr.trapz(dim="x")
intsimps = Xcorr.simps(dim="x")

# %% [markdown]
# As you can see both method give almost the same results in this case

# %%
scp.plot_multiple(
    method="scatter",
    ms=5,
    datasets=[inttrapz, intsimps],
    labels=["trapzoidal rule", "simpson' rule"],
    legend="best",
)

# %% [markdown]
# The difference between the trapezoidal and simpson integration methods is visualized below. In this case they are
# extremely close.

# %%
diff = (inttrapz - intsimps) * 100.0 / intsimps
diff.title = "difference"
diff.units = "percent"
diff.plot(scatter=True, ms=5)
