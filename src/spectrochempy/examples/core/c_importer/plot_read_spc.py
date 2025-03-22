# ruff: noqa
"""
Reading SPC format files
===========================
This example shows reading of 'Galactic Industries’ SPC data file format. (``.spc`` format).

"""

# %%
# First we need to import the spectrochempy API package
import spectrochempy as scp

DATADIR = scp.preferences.datadir
GALACTICDATA = DATADIR / "galacticdata"

# %%
# The SPC format is a proprietary format from Galactic Industries. It is used for
# storing spectroscopic data. The SPC format is a binary file format that can store
# multiple spectra in a single file. The SPC format is widely used in the field of
# spectroscopy and is supported by many spectroscopic software packages.

# %%
# Reading single file
ex1 = scp.read_spc(GALACTICDATA / "galacticdata/BENZENE.SPC")
ex1.plot()

# %%
# reading multiple files with same x coordinates (they are merged by default)
ex2 = scp.read_spc(GALACTICDATA / "galacticdata/CONTOUR.SPC")
ex2.plot()

# %%
# Reading multiple files with different x coordinates (they are not merged)
ex3 = scp.read_spc(GALACTICDATA / "galacticdata/DRUG_SAMPLE_PEAKS.SPC")
for nd in ex3:
    nd.plot_bar(width=0.1, clear=False)

# This ends the example ! The following line can be uncommented if no plot shows when
# running the .py script with python

# scp.show()

# %%
