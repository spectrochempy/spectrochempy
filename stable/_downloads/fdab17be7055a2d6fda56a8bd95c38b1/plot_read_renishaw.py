# -*- coding: utf-8 -*-
# %%
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# flake8: noqa
"""
Reading Renishaw WiRE files
===========================
This example shows how to read a Renishaw WiRE files (*.wdf format).

The spectral data used has been downloaded from
https://github.com/alchem0x2A/py-wdf-reader (MIT License) and stored in the
local directory ``ramandata/wire``.

The examples given here is a modified version of the original examples:
See https://github.com/alchem0x2A/py-wdf-reader/blob/master/examples
"""

import numpy as np

# %%
# First we need to import the spectrochempy API package
import spectrochempy as scp

# %%
# **Import dataset from local files**
# Read Raman data recorded in WiRe format (``.wdf`` extension).
# We just pass the file name as parameter.

# %%
# First read a single spectrum (measurement type : single)
dataset = scp.read_wire("ramandata/wire/sp.wdf")  # or read_wdf (or read)
_ = dataset.plot()

# %%
# Now read a series of spectra (measurement type : series) from a Z-depth scan.
dataset = scp.read_wdf("ramandata/wire/depth.wdf")
_ = dataset.plot_image()

# %%
# filter blank spectra
keep_rows = np.where(dataset.data.mean(axis=1) > 0)[0]
dataset = dataset[keep_rows]
_ = dataset.plot_image()

# %%
# extract a line scan data from a StreamLine HR measurement
dataset = scp.read("ramandata/wire/line.wdf")
_ = dataset.plot_image()

# %%
# finally extract grid scan data from a StreamLine HR measurement
dataset = scp.read_wdf("ramandata/wire/mapping.wdf")
# plot the dataset as an image (summming all wavenumbers)
_ = dataset.sum(dim=2).plot_image()
# plot the image taken at 1529cm-1
_ = dataset[..., 1529.0].plot_image()


# %%
# This ends the example ! The following line can be uncommented if no plot shows when
# running the .py script with python

# scp.show()
