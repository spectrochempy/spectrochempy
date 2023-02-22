# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# flake8: noqa
"""
Loading of experimental 1D NMR data
===================================

In this example, we load a NMR dataset (in the Bruker format) and plot it.

"""
# %%
import spectrochempy as scp

# %%
# `datadir.path` contains the path to a default data directory.

datadir = scp.preferences.datadir

path = datadir / "nmrdata" / "bruker" / "tests" / "nmr" / "topspin_1d"

# %%
# load the data in a new dataset

ndd = scp.NDDataset.read_topspin(path, expno=1, remove_digital_filter=True)

# %%
# view it...

_ = scp.plot(ndd)

# %%
# Now load a 2D  dataset

path = datadir / "nmrdata" / "bruker" / "tests" / "nmr" / "topspin_2d"
ndd = scp.NDDataset.read_topspin(path, expno=1, remove_digital_filter=True)
_ = scp.plot(ndd)

# scp.show()  # uncomment to show plot if needed (not necessary in jupyter notebook)
