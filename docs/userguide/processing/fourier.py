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
# # Fourier transformation (NMR)

# %% [markdown]
# In this notebook, we are going to transform time-domain NMR data into 1D or 2D spectra using SpectroChemPy processing tools

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
prefs.figure.figsize = (6,3)
_ = fid.plot(show_complex=True, xlim=(0,15000))
print("td = ", fid.size)

# %% [markdown]
# ## FFT 
#
# Now we perform a Fast Fourier Fransform (FFT):

# %%
spec = scp.fft(fid)
_ = spec.plot(xlim=(100,-100))
print("size = ", spec.size)

# %% [markdown]
# Alternative notation

# %%
k = 1024
spec = fid.fft(size=32*k )
_ = spec.plot(xlim=(100,-100))
print("size = ", spec.size)

# %%
newfid = spec.ifft()
# x coordinateis in second (base units) so lets transform it
newfid.x.ito('us')
_ = newfid.plot(show_complex=True, xlim=(0,15000))

# %% [markdown]
# Let's compare fid and newfid. There differs as a rephasing has been automatically applied after the first FFT (with the parameters found in the original fid metadata: PHC0 and PHC1).
#
# First point in the tme domain of the real part, is at the maximum. 

# %%
fid.real.plot(xlim=(0,5000), ls='--', label='original real part')
ax = newfid.real.plot(clear=False, data_only=True, c='r', label='fft + ifft')
_ = ax.legend()

# %% [markdown]
# First point in the time domain of the imaginary part is at the minimum. 

# %%
fid.imag.plot(xlim=(0,5000), ls='--',  label='original imaginary part')
ax = newfid.imag.plot(clear=False, data_only=True, c='r', label='fft + ifft')
_ = ax.legend(loc='lower right')

# %% [markdown]
# ## Preprocessing
#
# ### Line broadening
# Often before applying a FFT, some exponential multiplication `em`or other broadening filters such as `gm` or `sp` are applied. 

# %%
fidtrans = fid.em(lb='300. Hz')
spec = fidtrans.fft(size=32*1024)
_ = spec.plot(xlim=(100,-100))
print("size = ", spec.size)

# %% [markdown]
# ### Time domain baseline correction
# See the dedicated [tutorial](td_baseline).
