# -*- coding: utf-8 -*-
# %%
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# flake8: noqa
"""
Processing NMR spectra (Relaxation measurement)
================================================
Processing NMR spectra taken for relaxation measurements
"""
# %%
# Import API
# ----------
import spectrochempy as scp

# %%
# Importing a pseudo 2D NMR spectra
# ---------------------------------
# Define the folder where are the spectra
datadir = scp.preferences.datadir
nmrdir = datadir / "nmrdata" / "bruker" / "tests" / "nmr"

dataset = scp.read_topspin(nmrdir / "relax" / "100" / "ser")

# Change the y coordinate using the vdlist
with open(nmrdir / "relax" / "100" / "vdlist", mode="r") as f:
    vd = [float(val) for val in f.readlines()]

dataset.y = scp.Coord(vd, title="time", units="s")


# %%
# Analysing the data
# ---------------------
# Print dataset summary
dataset

# %%
# Plot the dataset
ds = dataset.em(lb=15)
ds = ds.fft()
ds = ds.pk(phc0=-10)
_ = ds.plot(xlim=(-60, -140))


# %%
# Integrate a region
dsint = ds[:, -90.0:-115.0].simpson()
_ = dsint.plot(marker="^")
dsint

# %%
# Fit a model
# (option not yet included to SpectroChemPy)

import numpy as np
from scipy.optimize import curve_fit

# see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html


def T1_model(t, I_0, T1):
    I = I_0 * (1 - np.exp(-t / T1))
    return I


# %%
xdata = np.append(
    0, dsint.y.data.squeeze()
)  # we add a zero at the begining to complete the series of values
ydata = np.append(0, dsint.real.data)  # data to fit

# initial parameters
I_0 = np.max(ydata)
T1 = 10
p0 = [I_0, T1]
popt, pcov = curve_fit(T1_model, xdata, ydata, p0, bounds=(0, 20000))
I_0, T1 = popt
ymodel = T1_model(xdata, I_0, T1)
ax = dsint.plot(marker="o", ls="")
_ = ax.plot(xdata, ymodel)
_ = ax.set_xlim(0, 50)
_ = ax.set_ylim(0, 5000)
_ = ax.set_title(f"I$_0$: {I_0}  T$_1$: {T1}s", fontsize=9)

# %%
# This ends the example ! The following line can be removed or commented
# when the example is run as a notebook (*.ipynb).

# scp.show()
