# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
"""
Removing cosmic ray spikes from a Raman spectrum
================================================

In this example, we use the `despike` method to remove the noise from a Raman
spectrum.
"""

import spectrochempy as scp

# %%
# Load and prepare the data
# --------------------------
dataset = scp.read("ramandata/labspec/serie190214-1.txt")

# %%
# Keep one spectrum and select the useful region:
X = dataset[0, 70.0:]

# %%
# Apply a fast SNIP baseline correction:
X1 = X.snip()

# %%
prefs = scp.preferences
prefs.figure.figsize = (8, 4)
_ = X1.plot()

# %%
# Despike with the default method
# --------------------------------
# The ``despike`` method has two key parameters: ``size`` (Savitzky-Golay filter
# width) and ``delta`` (detection threshold in standard deviation units):
X2 = scp.despike(X1, size=11, delta=5)
_ = X1.plot()
_ = X2.plot(clear=False, ls="-", c="r")

# %%
# Despike with the Whitaker method
# ---------------------------------
X3 = scp.despike(X1, size=11, delta=5, method="whitaker")
_ = X1.plot()
_ = X3.plot(clear=False, ls="-", c="r")

# %%
# The importance of parameter tuning
# -----------------------------------
# Poorly chosen parameters may remove valid peaks:
X4 = scp.despike(X1, size=21, delta=2)
_ = X1.plot()
_ = X4.plot(clear=False, ls="-", c="r")


# %%
# This ends the example ! The following line can be uncommented if no plot shows when
# running the .py script with python

# scp.show()

# sphinx_gallery_thumbnail_number = -1
