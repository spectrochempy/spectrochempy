# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
"""
Units manipulation examples
===========================

In this example, we show how units can be used in SpectroChemPy

"""

# %%

import spectrochempy as scp

# %%
# SpectroChemPy can do calculations with units. It uses
# [pint](https://pint.readthedocs.io) to define and perform operations on
# data with units.

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
# `ur` stands for **unit registry**, which handles many types of units
# and conversions between them.

# %%
# Units for dataset
# -----------------
#
# When loading experimental dataset using the `read` method, units are generally attributed to coordinates and data

ds = scp.read("wodger.spg")[0]
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
except Exception as e:
    print(f"Expected incompatible-unit conversion error: {e}")

# %%
# This, of course, also applies to NDDataset.
# Let's try for the `x` coordinate. It is a `wavenumber` in $cm^{-1}$ that
# can be transformed to $Hz$, for instance:

ds.x.ito("terahertz")
_ = ds.plot()
# %%
# We can also change wavenumber or frequency units to energy or wavelength,
# as SpectroChemPy, thanks to [pint](https://pint.readthedocs.io), knows how
# to make the transformation.

ds.x.ito("eV")
_ = ds.plot()
# %%
try:
    ds.x.ito("nanometer")
except Exception as e:
    print(f"Expected incompatible coordinate conversion error: {e}")

ds.x = ds.x.to("nanometer")
ds.x

# %%
_ = ds.plot()
# %%
# `absorbance` units (the units of the data)
# can also be transformed into `transmittance`
ds.ito("transmittance")
_ = ds.plot()
# %%
# Back to `absorbance`.
ds.ito("absorbance")
ds.x.ito("cm^-1")
_ = ds.plot()

# %%
# This ends the example ! The following line can be uncommented if no plot shows when
# running the .py script with python

# scp.show()
