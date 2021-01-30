# -*- coding: utf-8 -*-
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
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python 3
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
#     version: 3.9.0
#   widgets:
#     application/vnd.jupyter.widget-state+json:
#       state: {}
#       version_major: 2
#       version_minor: 0
# ---

# %% [markdown]
# # Fourier transformation

# %% [markdown]
# In this notebook, we are going to transform time-domain NMR data into 1D or 2Dspectra using SpectroChemPy processing tools

# %%
import spectrochempy as scp

# %% [markdown]
# First we open read some time domain data. Here is a NMD free induction decay (FID):

# %%
path = scp.preferences.datadir / 'nmrdata' / 'bruker' / 'tests' / 'nmr' / 'topspin_1d'
fid = scp.read_topspin(path)

# %% [markdown]
# The type of the data is complex:

# %%
fid.dtype

# %% [markdown]
# We can represent both real and imaginary parts on the same plot using the `show_complex` parameter.

# %%
prefs = fid.preferences
prefs.figure.figsize = (6,4)
_ = fid.plot(show_complex=True, xlim=(0,15000))

# %%
spec = scp.fft(fid)

_ = spec.plot()

# %%
scp.show()
