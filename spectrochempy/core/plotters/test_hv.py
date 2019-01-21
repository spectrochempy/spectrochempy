# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Use of HV

# %%
from spectrochempy import *

# %%
nd= NDDataset.read_omnic('irdata/nh4y-activation.spg')
nd.y -= nd.y[0] 
nd.y.title = 'Time'
da = nd.to_xarray()
da

# %%
da.attrs['units'][1]

# %%
nd.plot()

# %%
import hvplot as hv
import hvplot.xarray

# %%
da.hvplot(width=400)

# %%
from spectrochempy import *
fs = FileSelector(filters='.scp')
fs

# %%
fs.value, fs.path, fs.fullpath
