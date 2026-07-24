# %%
# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
"""
Denoising a 2D Raman spectrum
=============================

In this example, we use the `denoise` method to remove the noise from a 2D Raman
spectrum.
"""

# %%
# Import spectrochempy
import spectrochempy as scp

scp.set_loglevel("INFO")

# %%
# Load and prepare the data
# --------------------------
dataset = scp.read("ramandata/labspec/serie190214-1.txt")

# %%
# Select the useful spectral region:
nd = dataset[:, 60.0:]

# %%
# Plot the original data:
_ = nd.plot(title="original data")

# %%
# Detrend for easier comparison:
nd1 = nd.detrend(title="detrended data")
_ = nd1.plot()

# %%
# Denoise with default parameters
# --------------------------------
# The default ``ratio`` is 99.8:
nd2 = nd1.denoise()
_ = nd2.plot(title="denoised data")

# %%
# Denoise with different ratios
# ------------------------------
nd3 = nd1.denoise(ratio=95)
_ = nd3.plot(title="denoised data")
# sphinx_gallery_thumbnail_number = 5

nd4 = nd1.denoise(ratio=90)
_ = nd4.plot(title="denoised data")

# %%
# Denoising increases the signal-to-noise ratio effectively but has limited
# effect on cosmic ray spikes.  Consider using ``despike`` methods for those.

# %%
# This ends the basic example of denoising a 2D Raman spectrum.
# scp.show()  # uncomment to show plot if running from a script
# sphinx_gallery_thumbnail_number = -1
