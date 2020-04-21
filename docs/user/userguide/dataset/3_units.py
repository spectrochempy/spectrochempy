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
# # Units

# %% [markdown]
# Spectrochempy can do calculations with units - it uses [pint](https://pint.readthedocs.io) to define and perform operation on data with units.

# %%
from spectrochempy import *

# %% [markdown]
# ## Create quantities
#
# to create quantity, use for instance, one of the following expression:

# %%
Quantity('10.0 cm^-1')

# %%
Quantity(1.0, 'cm^-1/hour')

# %%
Quantity(10.0, ur.cm/ur.km)

# %% [markdown]
# or may be (?) simpler,

# %%
10.0 * ur.meter/ur.gram/ur.volt

# %% [markdown]
# `ur` stands for **unit registry**, which handle many type of units
# (and conversion between them)

# %% [markdown]
# ## Do arithmetics with units

# %%
a = 900 * ur.km
b = 4.5 * ur.hours
a/b

# %% [markdown]
# Such calculations can also be done using the following syntax, using a string expression

# %%
Quantity("900 km / (8 hours)")

# %% [markdown]
# ## Convert between units

# %%
c = a/b
c.to('cm/s')

# %% [markdown]
# We can make the conversion *inplace* using *ito* instead of *to*

# %%
c.ito('m/s')
c

# %% [markdown]
# ## Do math operations with consistent units

# %%
x = 10 * ur.radians
np.sin(x)

# %% [markdown]
# Consistency of the units are checked!

# %%
x = 10 * ur.meters
np.sqrt(x)

# %% [markdown]
# but this is wrong...

# %%
x = 10 * ur.meters
try:
    np.cos(x)
except DimensionalityError as e:
    error_(e)

# %% [markdown]
# Units can be set for NDDataset data and/or Coordinates

# %%
ds = NDDataset([1., 2., 3.], units='g/cm^3', title='concentration')
ds

# %%
ds.to('kg/m^3')

# %% [markdown]
# One can do transparent calculation using the units

# %%
volume = Quantity("2 m^3")
ds1 = ds * volume
ds1

# %%
ds1/ds
