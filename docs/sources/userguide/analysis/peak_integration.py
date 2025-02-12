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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Peak Integration
#
# This tutorial shows how to find peak maxima and determine peak areas with
# SpectroChemPy. As a prerequisite,
# the user is expected to have read the [Import](../importexport/import.rst),
# [Import IR](../importexport/importIR.rst),
# [Slicing](../processing/slicing.rst), and
# [Baseline Correction](../processing/baseline.rst) tutorials.

# %% [markdown]
# First, let's import the SpectroChemPy API.

# %%
import spectrochempy as scp

# %% [markdown]
# Now, import some 2D data into an NDDataset object.

# %%
ds = scp.read_omnic("irdata/nh4y-activation.spg")
ds

# %% [markdown]
# It's a series of 55 spectra.
#
# For the demonstration, select only the first 20 on a limited region from 1250 to
# 1800 cm$^{-1}$ (Do not forget to
# use floating numbers for slicing).

# %%
X = ds[:20, 1250.0:1800.0]

# %% [markdown]
# We can also eventually remove the offset on the acquisition time dimension (y).

# %%
X.y -= X.y[0]
X.y.ito("min")
X.y.title = "acquisition time"

# %% [markdown]
# We set some plotting preferences and then plot the raw data.

# %%
prefs = X.preferences
prefs.figure.figsize = (6, 3)
prefs.colormap = "Dark2"
prefs.colorbar = True
X.plot()

# %% [markdown]
# Now we can perform some baseline correction.

# %%
blc = scp.Baseline()
blc.ranges = (
    [1740.0, 1800.0],
    [1550.0, 1570.0],
    [1250.0, 1300.0],
)  # define 3 regions where we want the baseline to reach zero.
blc.model = "polynomial"
blc.order = 3

blc.fit(X)  # fit the baseline

Xcorr = blc.corrected  # get the corrected dataset
Xcorr.plot()

# %% [markdown]
# To integrate each row on the full range, we can use the sum or trapz method of an
# NDDataset.

# %%
inttrapz = Xcorr.trapezoid(dim="x")
intsimps = Xcorr.simpson(dim="x")

# %% [markdown]
# As you can see, both methods give almost the same results in this case.

# %%
scp.plot_multiple(
    method="scatter",
    ms=5,
    datasets=[inttrapz, intsimps],
    labels=["trapezoidal rule", "Simpson's rule"],
    legend="best",
)

# %% [markdown]
# The difference between the trapezoidal and Simpson integration methods is visualized
# below. In this case, they are
# extremely close.

# %%
diff = (inttrapz - intsimps) * 100.0 / intsimps
diff.title = "difference"
diff.units = "percent"
diff.plot(scatter=True, ms=5)
