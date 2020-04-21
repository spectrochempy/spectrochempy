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
# # The isotope database for NMR

# %%
from spectrochempy import *

# %% [markdown]
# If you are interested by **NMR**, it might be useful to have information about NMR isotopes, *e.g.*, to include this in some further calculations.
#
# Let's get information about the $^{27}$Al nucleus. 
#
# We use the **Isotopes** class:

# %%
al = Isotopes('27Al')
al

# %%
print(al.symbol, al.spin)

# %%
quadrupolar_moment = al.Q.to('m^2') 
quadrupolar_moment

# %%
gyromagnetic_ratio = al.gamma
gyromagnetic_ratio

# %% [markdown]
# It is then obvious how to get the Larmor frequency of aluminium:

# %%
B0 = Quantity('9.4 tesla')
print("{freq:~.2f} at {B0:~.2f}".format(freq=al.gamma * B0, B0=B0))
