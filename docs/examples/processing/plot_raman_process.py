# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# flake8: noqa
"""
Processing RAMAN spectra
========================
Various examples of processing RAMAN spectra
"""
# %%
# Import API
import spectrochempy as scp

# %%
# ## Importing spectra
# Define the folder where are the spectra
datadir = scp.preferences.datadir
ramandir = datadir / "ramandata"

# %%
# Read a single spectrum
A = scp.read_labspec("SMC1-Initial_RT.txt", directory=ramandir)

# %%
# Plot the spectrum
_ = A.plot()
scp.show()

# %%
# Crop the spectrum to a useful region
B = A[100.0:]
_ = B.plot()
scp.show()

# %%
# ## Baseline correction
# Let's try to remove the baseline using differents methods
# For this we use the `Baseline` processor
#
# First, we define the baseline processor
blc = scp.Baseline()

# %%
# Now we can try the various baseline methods
# ### Detrending
blc.interpolation = "detrend"
blc.order = 1  # linear detrending
blc.fit(B)
Bcorr = blc.transform()
baseline = blc.baseline
Bcorr.plot()
baseline.plot(clear=False, color="r")
scp.show()

# %%
# Clearly, this is not a good method for this spectrum because the baseline is not linear
# let's try with a polynomial detrend of order 2
blc.order = 2
blc.fit(B)
Bcorr = blc.transform()
baseline = blc.baseline
Bcorr.plot()
baseline.plot(clear=False, color="r")
scp.show()

# ### Asymmetric Least Squares
blc.interpolation = "als"

# %%
# Not much better, let's try with the asymmetric least squares method

# %%
scp.show()
