# %%
# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
"""
Processing a 1D NMR spectrum
============================
Basic 1D NMR processing and inspection with the official
``spectrochempy-nmr`` plugin.

This example stays within the currently validated public scope of the plugin:

* reading a 1D NMR dataset;
* processing the FID to a frequency-domain spectrum;
* inspecting slices, correcting the baseline and picking peaks.
"""

# %%
# Import API
# ----------
import spectrochempy as scp

# %%
# Import a 1D NMR FID
# -------------------
datadir = scp.preferences.datadir
nmrdir = datadir / "nmrdata"

dataset = scp.nmr.read(nmrdir / "bruker" / "tests" / "nmr" / "topspin_1d" / "1" / "fid")

# %%
# Process the 1D FID
# ------------------
experiment = scp.nmr.Experiment(dataset)
dataset = experiment.process(apodization="em", lb=2.0, size=16384, phase="metadata")


# %%
# Analyse the processed 1D spectrum
# ---------------------------------
# Print dataset summary
dataset

# %%
# Plot the dataset
_ = dataset.plot()

# %%
# Select a region of interest
spectrum = dataset[-40.0:-15.0]
_ = spectrum.plot()

# %%
# Peak picking
# ------------
peaks, _ = spectrum.find_peaks()


# %%
# plot the position of the peaks
# For this we will define a plot function that we be reused later


def plot_with_pp(s, peaks):
    ax = s.plot()  # output the spectrum on ax. ax will receive next plot too;
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
        x, y = p.coord(-1).values.m, (p + 0.2).values.m
        ax.annotate(
            f"{x:0.1f}",
            xy=(x, y),
            xytext=(-5, 5),
            rotation=90,
            textcoords="offset points",
        )


plot_with_pp(spectrum, peaks)

# %%
# Set some parameters to get less but significant peaks
peaks, _ = spectrum.find_peaks(height=1.0, distance=1.0)
plot_with_pp(spectrum, peaks)

# %%
# Peak fitting
# ------------
#
# Fit parameters are defined in a script (a single text as below)
script = """
#-----------------------------------------------------------
# syntax for parameters definition:
# name: value, low_bound,  high_bound
# available prefix:
#  # for comments
#  * for fixed parameters
#  $ for variable parameters
#  > for reference to a parameter in the COMMON block
#    (> is forbidden in the COMMON block)
# common block parameters should not have a _ in their names
#-----------------------------------------------------------
#

COMMON:
$ commonwidth: 1, 0, 5
$ commonratio: .5, 0, 1

MODEL: LINE_1
shape: voigtmodel
    $ ampl:  1, 0.0, none
    $ pos:   -21.7, -22., -20
    > ratio: commonratio
    > width: commonwidth

MODEL: LINE_2
shape: voigtmodel
    $ ampl:  4, 0.0, none
    $ pos:   -24, -24.5, -23.5
    > ratio: commonratio
    > width: commonwidth

MODEL: LINE_3
shape: voigtmodel
    $ ampl:  4, 0.0, none
    $ pos:   -25.4, -25.8, -25
    > ratio: commonratio
    > width: commonwidth

MODEL: LINE_4
shape: voigtmodel
    $ ampl:  4, 0.0, none
    $ pos:   -27.8, -28.5, -27
    > ratio: commonratio
    > width: commonwidth

MODEL: LINE_5
shape: voigtmodel
    $ ampl:  4, 0.0, none
    $ pos:   -31.5, -32, -31
    > ratio: commonratio
    > width: commonwidth

"""

# %%
# We will work here on the processed 1D region of interest.
s4p = spectrum.snip()  # Baseline correction

# %%
# create an Optimize object
f1 = scp.Optimize(log_level="INFO")

# %%
# Set parameters
f1.script = script
f1.max_iter = 5000


# %%
# Fit the slice s4p
_ = f1.fit(s4p)

# %%
# Show the result
_ = s4p.plot()
ax = (f1.components[:]).plot(clear=False)
ax.autoscale(enable=True, axis="y")

# Plotmerit
som = f1.inverse_transform()
_ = f1.plot_merit(offset=2)

# %%
# This ends the example ! The following line can be removed or commented
# when the example is run as a notebook (`.ipynb`).

# scp.show()
