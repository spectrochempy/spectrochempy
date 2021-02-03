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
# # N-d Fourier transformation (NMR)

# %%
import spectrochempy as scp

# %% [markdown]
# ## FFT of 2D spectra

# %% [markdown]
# We will process a 2D HMQC spectrum:

# %%
path = scp.preferences.datadir / 'nmrdata' / 'bruker' / 'tests' / 'nmr' / 'topspin_2d'
ser = scp.read_topspin(path)
ser.plot_map()
ser

# %% [raw]
# Extraction and Fourier transformation of the first row :

# %%
row0 = ser[0]
_ = row0.plot()

# %%
row0.fft()

# %%
row0.is_quaternion

# %%
