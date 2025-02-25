# -*- coding: utf-8 -*-
# %%
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# flake8: noqa
"""
Denoising a 2D Raman spectrum
=============================

In this example, we use the `denoise` method to remove the noise from a 2D Raman
spectrum.
"""

# %%
# Import spectrochempy

import spectrochempy as scp

scp.set_loglevel("INFO")  # to see information

# %%
# Load the data (should be a 2D spectrum or a list of datasets that can be merged):

dataset = scp.read("ramandata/labspec/serie190214-1.txt")

# %%
# select the useful region

nd = dataset[:, 60.0:]

# %%
# Basic plot

_ = nd.plot(title="original data")

# %%
# Detrend the data (for a easier comparison)

nd1 = nd.detrend(title="detrended data")
_ = nd1.plot()

# %%
# Denoise the data using the `denoise` method with the default parameters
# i.e., ratio=99.8
nd2 = nd1.denoise()
_ = nd2.plot(title="denoised data")

# %%
# Denoise the data using a different ratio
nd3 = nd1.denoise(ratio=95)
_ = nd3.plot(title="denoised data")

# sphinx_gallery_thumbnail_number = 5

nd4 = nd1.denoise(ratio=90)
_ = nd4.plot(title="denoised data")

# %%
# This example shows that denoising can be used effectively on such spectra to increase the signal-to-noise ratio.
# However, it apparently has a poor effect on eliminating cosmic ray peaks.
# For the latter, it may be useful to use `despike` methods as seen in another example.

# %%
# This ends the basic example of denoising a 2D Raman spectrum.

# scp.show()  # uncomment to show plot if running from a script

# sphinx_gallery_thumbnail_number = -1
