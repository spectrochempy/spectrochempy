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
# ## Importing a 1D spectra
# Define the folder where are the spectra
datadir = scp.preferences.datadir
ramandir = datadir / "ramandata"

# %%
# Read a single spectrum
A = scp.read_labspec("SMC1-Initial_RT.txt", directory=ramandir)

# %%
# Plot the spectrum
_ = A.plot()

# %%
# Crop the spectrum to a useful region
B = A[100.0:]
_ = B.plot()

# %%
# ## Baseline correction
# Let's try to remove the baseline using differents methods
# For this we use the `Baseline` processor
#
# First, we define the baseline processor
blc = scp.Baseline(log_level="INFO")

# %%
# Now we can try the various baseline methods
# ### Detrending
blc.model = "detrend"
blc.order = 1  # linear detrending
blc.fit(B)

# %%
# The baseline is now stored in the `baseline` attribute of the processor
corr = blc.transform()
baseline = blc.baseline


# %%
# Let's plot the result of the correction
#
# As we will use thii type of plot several times,
# we define a function for it
def plot_result():
    B.plot()
    corr.plot(clear=False, color="g")
    baseline.plot(clear=False, color="r")


plot_result()

# %%
# Clearly, this is not a good method for this spectrum because the baseline is not linear
# let's try with a polynomial detrend of order 2
blc.order = 2  # quadratic detrending
blc.fit(B)
corr = blc.transform()
baseline = blc.baseline
plot_result()

# %%
# Not much better, let's try with the asymmetric least squares method

# ### Asymmetric Least Squares
blc.model = "als"
blc.mu = 10**7  # smoothness
blc.asymmetry = 0.01
blc.fit(B)
corr = blc.transform()
baseline = blc.baseline
plot_result()

# %%
# This is better

# %%
# ## ALS on 2D spectra (series of spectra)

# First, we read the series of spectra

C = scp.read_labspec("Activation.txt", directory=ramandir)
C = C[::10]  # for sake of speed, we keep only 1 spectrum over 10

# %%
# Let's plot the series of spectra
C.plot()

# %%
# Now we apply the ALS method on the series of spectra
#
# We keep the same parameters as before and fit the new dataset

blc.fit(C)
corr = blc.transform()
baseline = blc.baseline
corr.plot()


# %%
# This ends the example ! The following line can be uncommented if no plot shows when
# running the .py script with python

# scp.show()
