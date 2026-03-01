# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# flake8: noqa
"""
Units manipulation examples
===========================

In this example, we show how units can be used in SpectroChemPy

"""

# %%

import spectrochempy as scp

# %%
# Spectrochempy can do calculations with units - it uses [pint](https://pint.readthedocs.io) to define and perform
# operation on data with units.

# %%
# Create quantities
# -----------------
# To create quantity, use for instance, one of the following expression:

scp.Quantity("10.0 cm^-1")

""
scp.Quantity(1.0, "cm^-1/hour")

# %%
# or may be simpler using `ur`:

ur = scp.ur
10.0 * ur.meter / ur.gram / ur.volt

# %%
# `ur` stands for **unit registry**, which handle many type of units (and conversion between them)

# %%
# Units for dataset
# -----------------
#
# When loading experimental dataset using the `read` method, units are generally attributed to coordinates and data

ds = scp.read("wodger.spg")[0]
prefs = ds.preferences
prefs.figure.figsize = (7, 3)
_ = ds.plot()

# %%
# * `wavenumbers` (`x` ) coordinates are here expressed in $cm^{-1}$
# * and `data` are in absorbance ($a.u.$) units.

# %%
# Convert between units
# ----------------------
#
# Here are some examples

x = 36 * ur("km/hr")
x.to("cm/s")

# %%
# We can make the conversion *inplace* using *ito* instead of *to*

x.ito("m/s")
x

# %%
# Obviously you cannot convert between incompatible units

try:
    x.to("hour")
except scp.DimensionalityError as e:
    scp.error_(scp.DimensionalityError, e)

# %%
# This, of course, also applies to NDDataset.
# Let's try for the `x` coordinate. It is `wavenumber` in $cm^{-1}$ that can be transformed in $Hz$ for instance:

ds.x.ito("terahertz")
_ = ds.plot()

# %%
# We can also change the wavenumbers (or frequency units), to energy units or wavelength as
# Spectrochempy (thanks to [pint](https://pint.readthedocs.io)) knows how to make the transformation.

ds.x.ito("eV")
_ = ds.plot()

""
try:
    ds.x.ito("nanometer")
except Exception as e:
    scp.error_(Exception, e)

""
ds.x = ds.x.to("nanometer")
print(ds.x)
_ = ds.plot()

# %%
# `absorbance` units (the units of the data) can also be transformed into `transmittance`

ds.ito("transmittance")
_ = ds.plot()

""
ds.ito("absorbance")
ds.x.ito("cm^-1")
_ = ds.plot()

# %%
# This ends the example ! The following line can be uncommented if no plot shows when
# running the .py script with python

# scp.show()
