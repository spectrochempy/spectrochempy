# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
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
#     version: 3.9.9
#   widgets:
#     application/vnd.jupyter.widget-state+json:
#       state: {}
#       version_major: 2
#       version_minor: 0
# ---
# %%
import numpy as np

# %%
import spectrochempy as scp
from spectrochempy import Quantity, ur  # to simplify further writing

# %% [markdown]
# # Units & Quantities

# %% [markdown]
# SpectroChemPy can do calculations with units - it uses [pint](https://pint.readthedocs.io) to define and perform
# operation on data with units.
#
# Two objects, `ur` and `Quantity` , allow the manipulation of data with units:
#
# * `ur`: stands for **unit registry**, is used to define and handle many type of units as well as making conversion
#  between them.
# * `Quantity`: is a scalar or an array of scalars with some units.

# %% [markdown]
# ## Units
# For instance, a unit of speed can be defined as:
# %%
ur.cm / ur.s


# %% [markdown]
# ## Create quantities
#
# to create quantity, use for instance, one of the following expression:

# %%
Quantity("10.0 cm^-1")

# %%
Quantity(1.0, "cm^-1/hour")

# %%
Quantity(10.0, ur.cm / ur.km)

# %% [markdown]
# or may be (?) simpler,

# %%
10.0 * ur.cm / ur.km

# %% [markdown]
# ## Do arithmetic with units

# %%
a = 900 * ur.km
b = 4.5 * ur.hours
a / b

# %% [markdown]
# Such calculations can also be done using the following syntax, using a string expression

# %%
Quantity("900 km / (4.5 hours)")

# %% [markdown]
# ## Conversion between units

# %%
c = a / b
d = c.to("cm/s")


# %% [markdown]
# As shown below `to()` has generated a new variable and does not affect the initial one:

# %%
print(f"initial quantity: c = {c}")
print(f"converted quantity: d = {d}")

# %% [markdown]
# We can make the conversion *inplace* using `ito()` instead of `to()`:

# %%
c.ito("m/s")
print(f"converted quantity: c = {c}")

# %% [markdown]
# ## Do math operations with consistent units
# The units are transformed consistently in maths operations:

# %%
x = 10 * ur.meters
np.sqrt(x)

# %%
x = 10 * ur.radians
np.sin(x)

# %% [markdown]
# Consistency of the units are checked and errors are generated if quantities have not appropriate units
# with the math operation...

# %%
x = 10 * ur.meters
try:
    np.cos(x)
except scp.DimensionalityError as e:  # catch the error
    scp.error_(
        scp.DimensionalityError, e
    )  # generate the error message (see API configuration)

# Consistency of the units are checked and errors are generated if quantities have not appropriate units
# with the math operation...

# %% [markdown]
# ## Stripping the units
#
# If for any reason - including quick and dirty checks -
# unitless numbers are needed, the `magnitude` field can be used:

# %%
x = 10 * ur.meters
np.cos(x.magnitude)

# %% [markdown]
# Units can be set for NDDataset data and/or Coordinates

# %%
ds = scp.NDDataset([1.0, 2.0, 3.0], units="g/cm^3", title="concentration")
ds

# %%
ds.to("kg/m^3")

# %% [markdown]
# One can do transparent calculation using the units

# %%
volume = Quantity("2 m^3")
ds1 = ds * volume
ds1

# %%
ds1 / ds
