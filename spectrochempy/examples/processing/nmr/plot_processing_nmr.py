# -*- coding: utf-8 -*-
# %%
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# flake8: noqa
"""
Processing NMR spectra
========================
Various examples of processing NMR spectra
"""
# %%
# Import API
# ----------
import spectrochempy as scp

# %%
# Importing a 2D NMR spectra
# --------------------------
# Define the folder where are the spectra
datadir = scp.preferences.datadir
nmrdir = datadir / "nmrdata"

dataset = scp.read_topspin(
    nmrdir / "bruker" / "tests" / "nmr" / "topspin_2d" / "1" / "pdata" / "1" / "ser"
)


# %%
# Analysing the 2D NMD dataset
# ----------------------------
# Print dataset summary
dataset

# %%
# Plot the dataset
_ = dataset.plot_map()

# %%
# Extract slices along x
s = dataset[-27.6, :]
_ = s.plot()

# %%
# Baseline correction of this slice
# Note that only the real part is corrected
sa = s.snip(snip_width=100)
_ = sa.plot()

# %%
# apply this correction to the whole dataset
sb = dataset.snip(snip_width=100)
_ = sb.plot_map()

# %%
# Select a region of interest
nd = sb[
    -40.0:-15.0, 55.0:20.0
]  # note the use of float to delect using coordinates (not poitn indexes)
_ = nd.plot_map()

# %%
# Extract slices along x
s1 = nd[-27.6, :]
_ = s1.plot()

# %%
s2 = nd[-25.7, :]
_ = s2.plot()

# %%
# plot two slices on the same figure
_ = s1.plot()
_ = s2.plot(
    clear=False,
    color="red",
    linestyle="-",
)

# %%
# Now slice along y
s3 = nd[:, 40.0]
s4 = nd[:, 36.0]
_ = s3.plot(color="violet", ls="-", lw="2")
_ = s4.plot(clear=False, color="green", ls="-", lw="2")

# %%
# Peak picking
# ------------
peaks, _ = s2.find_peaks()


# %%
# plot the position of the peaks
# For this we will define a plot function that we be reused later


def plot_with_pp(s, peaks):
    ax = s.plot()  # output the spectrum on ax. ax will receive next plot too
    pks = peaks + 0.2  # add a small offset on the y position of the markers
    _ = pks.plot_scatter(
        ax=ax,
        marker="v",
        color="black",
        clear=False,  # we need to keep the previous output on ax
        data_only=True,  # we don't need to redraw all things like labels, etc...
        ylim=(-0.1, 7),
    )

    for p in pks:
        x, y = p.coord(-1).values, p.values + 0.2
        _ = ax.annotate(
            f"{x.m:0.1f}",
            xy=(x, y),
            xytext=(-5, 5),
            rotation=90,
            textcoords="offset points",
        )


_ = plot_with_pp(s2, peaks)

# %%
# Set some parameters to get less but significant peaks
peaks, _ = s2.find_peaks(height=1.0, distance=1.0)
_ = plot_with_pp(s2, peaks)

# %%
# Now look in the other dimension using slice s4
peaks, _ = s4.find_peaks(height=1.0, distance=1.0)
_ = plot_with_pp(s4, peaks)

# %%

# %%
