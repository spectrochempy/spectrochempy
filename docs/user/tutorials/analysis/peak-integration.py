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
# # Peak integration
#
# This tutorial shows how to find peaks and determine peak areas with spectrochempy. As prerequisite, the user is
# expected to have read the [Import](../IO/import.ipynb), [Import IR](../IO/importIR.ipynb),  [slicing](../process/slicing.ipynb) and [baseline corrrection](../process/baseline_correct.ipynb) tutorials. 
#
# First lets import the modules that will be used in this tutorial

# %%
import spectrochempy as scp           
import matplotlib.pyplot as plt  # will be used for some plots 

# %%
X = scp.read_omnic("irdata//nh4y-activation.SPG")[:20,1300.0:1750.]
subplot1 = X.plot()

# %%
X2 = X[:,1800.0:1550.]
blc = scp.BaselineCorrection(X2)
X2corr = blc.compute([1750.,1730.0], [1570.,1550.])    # compute the corrected NDDataset 
subplot5 = X2corr.plot()


# %%
aires = X2corr.trapz()
aires

# %%
X2corr.write_jdx("fichier.jdx")


# %%
temps = (X.y - X.y[0]).to("minute")
temps
plt.plot(temps.data, aires)
