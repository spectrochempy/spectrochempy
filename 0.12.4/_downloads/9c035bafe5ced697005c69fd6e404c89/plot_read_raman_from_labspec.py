# ruff: noqa
"""
Loading RAMAN experimental file
===============================

Here we load experimental LABSPEC spectra and plot them.

"""

# %%
import spectrochempy as scp

# %%
# Read and plot a single file
# ----------------------------
datadir = scp.preferences.datadir
ramandir = datadir / "ramandata/labspec"
A = scp.read_labspec("Activation.txt", directory=ramandir)
A

# %%
_ = A.plot()

# %%
# Explore with different plot types
# ----------------------------------
# As a 2D dataset, we can display it as an image:
_ = A.plot_image()

# %%
# or a contour plot:
_ = A.plot_map()

# %%
# Read and merge multiple files
# ------------------------------
B = scp.read_labspec(ramandir / "subdir")
_ = B.plot()

# %%
# This ends the example ! The following line can be uncommented if no plot shows when
# running the .py script with python

# scp.show()
