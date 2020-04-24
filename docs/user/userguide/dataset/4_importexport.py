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
# # Import and export of NDDataset objects

# %% [markdown]
# As usual we start by importing the SpectroChemPy API

# %%
from spectrochempy import * 

# %% [markdown]
# ## Data directory
#
# The builtin **datadir** variable contains a path to our *test*'s data.
#
# However it is always possible to specify alternative locations: Any existing file path can be specified in import
# functions calls.
#
# <div class='alert alert-info'>
#     
# **NOTE:**
# In import function calls, if we do not specify the **datadir**, the application will first look in this directory by default, if it doesn't find the path in the current directory.
#
# </div>

# %%
# let check if the `datadir` directory exists
import os
datadir = general_preferences.datadir
if os.path.exists(datadir):
    print(datadir)

# %%
# !cd C:\Users\christian\anaconda3\envs\scpy\lib\site-packages\scp_data\testdata

# %%
# !dir

# %% [markdown]
# ## File selector widget

# %% [markdown]
# A widget is provided to help with the selection of file names or directory. 
#
# <div class ="alert alert-warning">
#     
# **WARNING:**
# Experimental feature - subject to changes
#     
# </div>

# %%
path = general_preferences.datadir
fs = FileSelector(path = path, filters=['spg','spa'])   
fs

# %% [markdown]
# After validation of the selection, one can read the path and name of the selected files. 

# %%
fs.value, fs.path, fs.fullpath


# %% [markdown]
# ##  Infrared spectroscopy OMNIC file Import (.spg extension)
#

# %%
dataset = NDDataset.read_omnic(os.path.join('irdata', 'nh4y-activation.spg'))
dataset

# %%
# view it...
_ = dataset.plot(method='stack')

# %% [markdown]
#
# ## NMR Bruker data Import

# %% [markdown]
# Now, lets load a NMR dataset (in the Bruker format).

# %%
path = os.path.join(datadir, 'nmrdata','bruker', 'tests', 'nmr','bruker_1d')
ndd = NDDataset.read_bruker_nmr(path, expno=1, remove_digital_filter=True)
ndd

# %%
# view it...
_ = ndd.plot(color='blue')
