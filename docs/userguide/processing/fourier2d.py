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
# As a first exmple, we will process a 2D HMQC spectrum:

# %%
path = scp.preferences.datadir / 'nmrdata' / 'bruker' / 'tests' / 'nmr' / 'topspin_2d'
ser = scp.read_topspin(path)
ser.plot_map()
ser

# %% [markdown]
# Extraction and Fourier transformation of the first row :

# %%
row0 = ser[0]
_ = row0.plot()


# %%
r = row0.fft()
_ = r.plot()

# %% [markdown]
# FFT along dimension x for the whole 2D dataset

# %%
spec = ser.fft()
spec

# %%
_ = spec.plot()

# %%
_ = spec.plot_map()

# %% [markdown]
# Now we can perform a FFT in the second dimension. We must take into account the encoding:

# %%
spec.meta.encoding[0]

# %% [markdown]
# This type of encoding is however taken into account automatically.

# %%
sp = spec.fft(dim=0)
sp.plot_map()
sp

# %%
spec.meta.si

# %%
