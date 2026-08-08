# ruff: noqa
"""
Reading SPC format files
===========================
This example shows reading of 'Galactic Industries’ SPC data file format. (``.spc`` format).

"""

# %%
# Import the package
import spectrochempy as scp

# %%
# Read a single spectrum file
# ----------------------------
ex1 = scp.read_spc("galacticdata/BENZENE.SPC")
_ = ex1.plot()
ex1

# %%
# Read merged subfiles
# --------------------
# When subfiles share the same x-coordinates, they are merged automatically:
ex2 = scp.read_spc("galacticdata/CONTOUR.SPC")
_ = ex2.plot()
ex2

# %%
# Read subfiles with different x-coordinates
# -------------------------------------------
# Subfiles with distinct x-coordinates are returned as a list:
ex3 = scp.read_spc("galacticdata/DRUG_SAMPLE_PEAKS.SPC")
for nd in ex3:
    _ = nd.plot_bar(width=0.1, clear=False)
ex3

# This ends the example ! The following line can be uncommented if no plot shows when
# running the .py script with python

# scp.show()

# %%
