# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# flake8: noqa
"""
NDDataset baseline correction
==============================

In this example, we perform a baseline correction of a 2D NDDataset
interactively, using the `multivariate` method and a `pchip` interpolation.

"""

# %%
# As usual we start by importing the useful library, and at least  the
# spectrochempy library.

# %%

import spectrochempy as scp

# %%
# Load data:

datadir = scp.preferences.datadir
nd = scp.NDDataset.read_omnic(datadir / "irdata" / "nh4y-activation.spg")

# %%
# Do some slicing to keep only the interesting region:

ndp = (nd - nd[-1])[:, 1291.0:5999.0]
# Important:  notice that we use floating point number
# integer would mean points, not wavenumbers!

# %%
# Define the BaselineCorrection object:

ibc = scp.BaselineCorrection(ndp)

# %%
# Launch the interactive view, using the `BaselineCorrection.run` method:

ranges = [
    [1556.30, 1568.26],
    [1795.00, 1956.75],
    [3766.03, 3915.81],
    [4574.26, 4616.04],
    [4980.10, 4998.01],
    [5437.52, 5994.70],
]  # predefined ranges
span = ibc.run(
    *ranges, method="multivariate", interpolation="pchip", npc=5, zoompreview=3
)

# %%
# Print the corrected dataset:

print(ibc.corrected)
_ = ibc.corrected.plot()

# %%
# This ends the example ! The following line can be uncommented if no plot shows when running
# the .py script

# %%
# scp.show()
