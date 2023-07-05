# -*- coding: utf-8 -*-
# %%
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# flake8: noqa
"""
Analysis CP NMR spectra
========================
Example with handling of a series of CP NMR spectra.
"""
# %%
# Import API
# ----------
import spectrochempy as scp

# %%
# Import NMR spectra
# ------------------
# Define the folder where are the spectra
datadir = scp.preferences.datadir
nmrdir = datadir / "nmrdata" / "bruker" / "tests" / "nmr" / "CP"

# %%
# Set the `glob` pattern in order to load a series of spectra of given type
# in the given directory (here we read fid, but we could also read "1r" files
# when available)
dataset = scp.read_topspin(nmrdir, glob="**/fid")

# %%
# 15 fids have been read and merged into a single dataset
dataset

# %%
# The new dimension (y) have several coordinates corresponding to all metadata that change from fid to fid.
#
# In the present case, the relevant coordinates is given by the `p15` array which is the array of CP contact times.
#
# To have y using this coordinates, we need to select it
dataset.y.select(3)

# %%
# plot the dataset (zoom on the begining of the fid)
prefs = dataset.preferences
prefs.figure.figsize = (9, 4)
_ = ax = dataset.plot(colorbar=True)
_ = ax.set_xlim(0, 5000)

# %%
# Process a fourier transform along the x dimension

# exponential multiplication
nd1 = scp.em(dataset, lb=50)

# %%
# fourier transform
nd2 = scp.fft(nd1, si=4096)

# %%
# perform a phase correction of order 0 (need to be tuned carefully)
nd3 = scp.pk(nd2, phc0=-118)

# %%
# plot
_ = nd3.plot()

# %%
# snip baseline correction
nd4 = scp.snip(nd3, snip_width=200)

ax = nd4.plot()
_ = ax.set_xlim(225, 25)
_ = ax.set_ylim(-1, 10)

# %%
# Peak peaking
# we use the max of each spectra for this pp
peaks, properties = nd4.max(dim=0).find_peaks(height=2.0, distance=10.0)
peaks.x.data, properties

# %%
# plot with peak markers
ax = nd4.plot()  # output the spectrum on ax. ax will receive next plot too
pks = peaks + 0.5  # add a small offset on the y position of the markers
_ = pks.plot_scatter(
    ax=ax,
    marker="v",
    color="black",
    clear=False,  # we need to keep the previous output on ax
    data_only=True,  # we don't need to redraw all things like labels, etc...
    ylim=(-0.1, 13),
    xlim=(225, 25),
)

for p in pks:
    x, y = p.coord(-1).values, p.values + 0.5
    _ = ax.annotate(
        f"{x.m:0.1f}",
        xy=(x, y),
        xytext=(-5, 5),
        rotation=90,
        textcoords="offset points",
    )

# %%
# Get the section at once using fancy indexing
sections = nd4[:, peaks.x.data]

# The array sections has a shape (15, 3).
# We must transpose it to plot the three sections has a function of contact time
sections = sections.T

# now plot it
_ = sections.plot(marker="o", lw="1", ls=":", legend="best", colormap="jet")

# %%
# Fitting a model to these data
# Fitting of arbitrary model is not yet implemented in SpectroChemPy, but we can use scipy for that purpose

import numpy as np
from scipy.optimize import curve_fit

# see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html


def cp_model(t, I_0, T_IS, T1_Irho, T1_Srho):
    I = (
        I_0
        * (np.exp(-t / T1_Irho) - np.exp(-t * (1 / T_IS + 1 / T1_Srho)))
        / (1 + T_IS / T1_Srho - T_IS / T1_Irho)
    )
    return I


# %%
for section in sections:
    xdata = (
        section.y.data
    )  # Note we use axis `y` which correcponds to the contact times (data have been transposed!)
    ydata = section.data.squeeze()  # data to fit

    # initial parameters
    I_0 = np.max(ydata)
    T_IS = 10
    T1_Irho = 100
    T1_Srho = 10000
    p0 = [I_0, T_IS, T1_Irho, T1_Srho]
    popt, pcov = curve_fit(cp_model, xdata, ydata, p0, bounds=(0, 20000))

    I_0, T_IS, T1_Irho, T1_Srho = popt
    print(popt)
    ymodel = cp_model(xdata, I_0, T_IS, T1_Irho, T1_Srho)

    ax = section.plot(marker="o", ls="")
    ax.plot(xdata, ymodel)


# %%
# This ends the example ! The following line can be removed or commented
# when the example is run as a notebook (*.ipynb).

# scp.show()
