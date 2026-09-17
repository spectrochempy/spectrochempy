# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# flake8: noqa
"""
EFA (Keller and Massart original example)
=========================================

In this example, we perform the Evolving Factor Analysis of a TEST dataset
(ref. Keller and Massart, Chemometrics and Intelligent Laboratory Systems,
12 (1992) 209-224 )

"""
# %%
import numpy as np

import spectrochempy as scp

# sphinx_gallery_thumbnail_number = 5

# %%
# Generate a test dataset
# -----------------------
# 1) simulated chromatogram
# *************************

t = scp.Coord(np.arange(15), units="minutes", title="time")  # time coordinates
c = scp.Coord(range(2), title="components")  # component coordinates

data = np.zeros((2, 15), dtype=np.float64)
data[0, 3:8] = [1, 3, 6, 3, 1]  # compound 1
data[1, 5:11] = [1, 3, 5, 3, 1, 0.5]  # compound 2

dsc = scp.NDDataset(data=data, coords=[c, t])
dsc.plot(title="concentration")

# %%
# 2) absorption spectra
# **********************

spec = np.array([[2.0, 3.0, 4.0, 2.0], [3.0, 4.0, 2.0, 1.0]])
w = scp.Coord(np.arange(1, 5, 1), units="nm", title="wavelength")

dss = scp.NDDataset(data=spec, coords=[c, w])
dss.plot(title="spectra")

# %%
# 3) simulated data matrix
# ************************

dataset = scp.dot(dsc.T, dss)
dataset.data = np.random.normal(dataset.data, 0.1)
dataset.title = "intensity"

dataset.plot(title="calculated dataset")

# %%
# 4) evolving factor analysis (EFA)
# *********************************
efa = scp.EFA()
efa.fit(dataset)

# %%
# Plots of the log(EV) for the forward and backward analysis
#
efa.f_ev.T.plot(yscale="log", legend=efa.f_ev.k.labels)

# %%
efa.b_ev.T.plot(yscale="log", legend=efa.b_ev.k.labels)

# %%
# Looking at these EFA curves, it is quite obvious that only two components
# are really significant, and this corresponds to the data that we have in
# input.
# We can consider that the third EFA components is mainly due to the noise,
# and so we can use it to set a cut of values
n_pc = efa.n_components = 2

efa.cutoff = np.max(efa.f_ev[:, n_pc].data)
f2 = efa.f_ev[:, :n_pc]
b2 = efa.b_ev[:, :n_pc]

# %%
# we concatenate the datasets to plot them in a single figure
both = scp.concatenate(f2, b2)
both.T.plot(yscale="log")

# %%
# Get the abstract concentration profile based on the FIFO EFA analysis
#
C = efa.transform()
C.T.plot(title="EFA concentration")

# %%
# This ends the example ! The following line can be uncommented if no plot shows when
# running the .py script with python

# scp.show()
