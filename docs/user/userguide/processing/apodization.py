# -*- coding: utf-8 -*-
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
# # Apodization

# %%
from spectrochempy import *

# %% [markdown]
# ## Introduction 
#
# As an example, apodization is a transformation particularly useful for preprocessing NMR time domain data before Fourier transformation. It generally help for signal to noise improvement.

# %%
# reead an experimental spectra
path = os.path.join('nmrdata', 'bruker', 'tests', 'nmr', 'bruker_1d')
dataset = NDDataset.read_bruker_nmr(path, expno=1, remove_digital_filter=True)
dataset = dataset/dataset.max()     # normalization
# store original data
nd = dataset.copy()

# show data
nd

# %% [markdown]
# ### Plot of the Real and Imaginary original data

# %%
_ = nd.plot(xlim=(0.,15000.))
_ = nd.plot(imag=True, data_only=True, clear=False, color='r')

# %% [markdown]
# ### Exponential multiplication

# %%
_ = nd.plot(xlim=(0.,15000.))
_ = nd.em(lb=300. * ur.Hz)
_ = nd.plot(data_only=True, clear=False, color='g')

# %% [markdown]
# **Warning:** processing function are most of the time applied inplace. Use `inplace=False` option to avoid this if necessary

# %%
nd = dataset.copy()  # to go back to the original data
_ = nd.plot(xlim=(0.,5000.))
ndlb = nd.em(lb=300. * ur.Hz, inplace=False) # ndlb contain the processed data
_ = nd.plot(data_only=True, clear=False, color='g')  # nd dataset remain unchanged 
_ = ndlb.plot(data_only=True, clear=False, color='b')

# %% [markdown]
# Of course, imaginary data are also transformed at the same time

# %%
_ = nd.plot(imag=True, xlim=(0, 5000), color='r')
_ = ndlb.plot(imag=True, data_only=True, clear=False, color='b')

# %% [markdown]
# If we want to display the apodization function, we can use the `retfunc=True` parameter.

# %%
nd = dataset.copy()
_ = nd.plot(xlim=(0.,5000.))
ndlb, apod = nd.em(lb=300. * ur.Hz, inplace=False, retfunc=True) # ndlb contain the processed data and apod the apodization function
_ = ndlb.plot(data_only=True, clear=False, color='b')
_ = apod.plot(data_only=True, clear=False, color='m', linestyle='--')

# %% [markdown]
# #### Shifted apodization

# %%
nd = dataset.copy()
_ = nd.plot(xlim=(0.,5000.))
ndlb, apod = nd.em(lb=300. * ur.Hz, shifted=1000*ur.us, inplace=False, retfunc=True) # ndlb contain the processed data and apod the apodization function
_ = ndlb.plot(data_only=True, clear=False, color='b')
_ = apod.plot(data_only=True, clear=False, color='m', linestyle='--')

# %% [markdown]
# ### Other apodization functions

# %% [markdown]
# #### Gaussian-Lorentzian appodization

# %%
nd = dataset.copy()
lb = 10.
gb = 200.
ndlg, apod = nd.gm(lb=lb, gb=gb, inplace=False, retfunc=True)
_ = nd.plot(xlim=(0.,5000.))
_ = ndlg.plot(data_only=True, clear=False, color='b')
_ = apod.plot(data_only=True, clear=False, color='m', linestyle='--')

# %% [markdown]
# #### Shifted Gaussian-Lorentzian apodization

# %%
nd = dataset.copy()
lb = 10.
gb = 200.
ndlg, apod = nd.gm(lb=lb, gb=gb, shifted=2000*ur.us, inplace=False, retfunc=True)
_ = nd.plot(xlim=(0.,5000.))
_ = ndlg.plot(data_only=True, clear=False, color='b')
_ = apod.plot(data_only=True, clear=False, color='m', linestyle='--')

# %% [markdown]
# #### Apodization using sine window multiplication
#
# The`sp`  apodization is by default performed on the last dimension.
#
# Functional form of apodization window (cfBruker TOPSPIN manual): $sp(t) = \sin(\frac{(\pi - \phi) t }{\text{aq}} + \phi)^{pow}$
#
# where 
# * $0 < t < \text{aq}$ and  $\phi = \pi ⁄ \text{sbb}$ when $\text{ssb} \ge 2$ 
#
# or
# *    $\phi = 0$ when $\text{ssb} < 2$
#         
# $\text{aq}$ is an acquisition status parameter and $\text{ssb}$ is a processing parameter (see below) and $\text{pow}$ is an exponent equal to 1 for a sine bell window or 2 for a squared sine bell window.
#
# The $\text{ssb}$ parameter mimics the behaviour of the `SSB` parameter on bruker TOPSPIN software:
# * Typical values are 1 for a pure sine function and 2 for a pure cosine function.
# * Values greater than 2 give a mixed sine/cosine function. Note that all values smaller than 2, for example 0, have the same effect as $\text{ssb}=1$, namely a pure sine function.
#     
# **Shortcuts**:
# * `sine` is strictly a alias of `sp`
# * `sinm` is equivalent to `sp` with $\text{pow}=1$
# * `qsin` is equivalent to `sp` with $\text{pow}=2$
#
# Below are several examples of `sinm` and `qsin` apodization functions.

# %%
nd = dataset.copy()
_ = nd.plot()

new, curve = nd.qsin(ssb=3, retfunc=True)
_ = curve.plot(color='r', clear=False)
_ = new.plot(xlim=(0, 25000), zlim=(-2, 2), data_only=True, color='r', clear=False)

# %%
nd = dataset.copy()
_ = nd.plot()

new, curve = nd.sinm(ssb=1, retfunc=True)
_ = curve.plot(color='b', clear=False)
_ = new.plot(xlim=(0, 25000), zlim=(-2, 2), data_only=True, color='b', clear=False)

# %%
nd = dataset.copy()
_ = nd.plot()

new, curve = nd.sinm(ssb=3, retfunc=True)
_ = curve.plot(color='b', ls='--', clear=False)
_ = new.plot(xlim=(0, 25000), zlim=(-2, 2), data_only=True, color='b', clear=False)

# %%
nd = dataset.copy()
_ = nd.plot()

new, curve = nd.qsin(ssb=2, retfunc=True)
_ = curve.plot(color='m', clear=False)
_ = new.plot(xlim=(0, 25000), zlim=(-2, 2), data_only=True, color='m', clear=False)

# %%
nd = dataset.copy()
_ = nd.plot()

new, curve = nd.qsin(ssb=1, retfunc=True)
_ = curve.plot(color='g', clear=False)
_ = new.plot(xlim=(0, 25000), zlim=(-2, 2), data_only=True, color='g', clear=False)

# %%

# %%
