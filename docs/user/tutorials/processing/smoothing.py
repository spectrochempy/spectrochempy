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
# # Smoothing 
#
# This tutorial shows how to smooth spectra and compute their derivatives. 

# %%
import spectrochempy as scp
X = scp.read_omnic('irdata//nh4y-activation.SPG')
_ = X.plot()
                   

# %%
# We select a noisy part
X2 = X[0:5,3700.0:2800.0]
_ = X2.plot()

# %% [markdown]
#  A first algorithm will smooth using a moving window of a given 'length' (i.e. number of points)

# %%
_ = X2.smooth(length=11).plot()          #the default window length is 11

# %% [markdown]
# The other one is the Savitky-Golay algorithm which use ploynomlial interpolation in the moving window. Its interest is that it can also be used to compute derivatives)

# %%
_ = X2.savgol_filter(window_length = 21, polyorder=3, deriv=0).plot()
