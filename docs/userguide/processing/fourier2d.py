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
#     version: 3.9.1
#   widgets:
#     application/vnd.jupyter.widget-state+json:
#       state: {}
#       version_major: 2
#       version_minor: 0
# ---

# %% [markdown]
# # 2D Fourier transformation (NMR)

# %%
import spectrochempy as scp

# %% [markdown]
# ## Process step by step

# %%
# !python --version

# %% [markdown]
# As a first example, we will process a 2D HMQC spectrum:

# %%
path = scp.preferences.datadir / 'nmrdata' / 'bruker' / 'tests' / 'nmr' / 'topspin_2d'
ser = scp.read_topspin(path)
ser.plot_map()
ser

# %% [markdown]
# Extraction and Fourier transformation of the first row (row index:0):

# %%
row0 = ser[0]
_ = row0.plot()
row0

# %%
r = row0.fft()
_ = r.plot()
r

# %% [markdown]
# FFT along dimension x for the whole 2D dataset

# %%
spec = ser.fft()
_ = spec.plot()
spec

# %% [markdown]
# By default, plot() use the `stack`method, which is not the best in this case. Use `map` or `image`instead:

# %%
_ = spec.plot_map()
# TODO: change preferences for contour levels!

# %% [markdown]
# Now we can perform a FFT in the second dimension. We must take into account the encoding:

# %%
spec.meta.encoding[0]

# %% [markdown]
# This type of encoding is however taken into account automatically.

# %%
sp = spec.fft(dim=0)

# %%
sp.plot_image()
sp
