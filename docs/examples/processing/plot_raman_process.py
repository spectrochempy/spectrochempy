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
# temporaire
import os
os.environ["SCPY_TESTING"] = "1"

# %%
import spectrochempy as scp

# %%
# define the folder where are the spectra
datadir = scp.preferences.datadir
ramandir = datadir / "ramandata"

# %%
# read a single spectrum
A = scp.read_labspec("SMC1-Initial_RT.txt", directory=ramandir)

# %%
# plot the spectrum
_ = A.plot()


