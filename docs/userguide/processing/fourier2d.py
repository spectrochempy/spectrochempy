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
# # 2D Fourier transformation (NMR)

# %%
import spectrochempy as scp
from spectrochempy import ur 


# %% [markdown]
# ## Processing of Hypercomplex dataset

# %% [markdown]
# As a first example, we will process a 2D HMQC spectrum which has been acquired using a phase sensitive detection method : STATES-TPPI encoding.  The STATES (States, Ruben, Haberkorn) produce an hypercomplex dataset which need to be processed in a specific way, that SpectroChemPy handle automatically. TPPI (for Time Proportinal Phase Increment) is also handled.

# %%
path = scp.preferences.datadir / 'nmrdata' / 'bruker' / 'tests' / 'nmr' / 'topspin_2d'
ser = scp.read_topspin(path, expno=1)
prefs= ser.preferences 
prefs.figure.figsize = (7,3)
_ = ser.plot_map()

# %% [markdown]
# ### Processing steps
#
# * Optional : Apply some broadening by apodization in the time domain.
# * Optional : DC correction in the time domain.
# * Optional : Zero-filling.
# * Fourier transform in the F2 (x) dimension.
# * Phase  the first transformed dimension in the frequency domain.
# * Optional: Apply some apodization in the time domain for the F1 (y) dimension.
# * Optional: DC Correction in F1.
# * Optional: Zero-filling.
# * Fourier transform the second dimension F1.
# * Phase correct the second transformed dimension in the frequency domain.
#
#

# %% [markdown]
# ### Apodization, DC correction, Zero-filling

# %% [markdown]
# For this step we can first extract and Fourier transformation of the first row (row index:0).

# %%
row0 = ser[0]
_ = row0.plot()

# %%
row0 = ser[0]  
row0.dc(inplace=True)
row0.zf_size(size=2048, inplace=True)

# %%
shifted = row0.coordmax()

# %%
row0.em(lb=20*ur.Hz, shifted=shifted)
f0 = row0.fft()
_ = f0.plot()

# %%

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

# %% [markdown]
# ## Other encoding

# %% [markdown]
# ### Echo-antiecho

# %%
path = scp.preferences.datadir / 'nmrdata' / 'bruker' / 'tests' / 'nmr' / 'exam2d_HC'
ser = scp.read_topspin(path)
spec = ser.fft()
sp = spec.fft(dim='y')
_ = sp.plot_map()

# %% [markdown]
# ## Processing of Echo-AntiEcho encoded SER

# %% [markdown]
# In this second example, we will process a HSQC spectrum of Cyclosporin wich has been acquired using a Rance-Kay quadrature scheme, also known as Echo-Antiecho. (The original data is extracted from the examples of the Bruker Topspin software). 

# %%
path = scp.preferences.datadir / 'nmrdata' / 'bruker' / 'tests' / 'nmr' / 'exam2d_HC'
ser = scp.read_topspin(path)
ser.sp(ssb=2, inplace=True)  # Sine apodization
s2 = ser.fft(1024)
s2.pk(phc0=-38, inplace=True)
s2[0].plot()
s2.meta.pivot

# %%
s2.sp(ssb=2, dim='y', inplace=True)  # Sine apodization in the y dimension
spec = s2.fft(1024, dim='y')
_ = spec.plot_map()
spec.meta.pivot

# %%
s = spec.pk(phc0=45, dim='y')
_ = s.plot_map() #xlim=(3.5, 2.5), ylim=(20,45))
