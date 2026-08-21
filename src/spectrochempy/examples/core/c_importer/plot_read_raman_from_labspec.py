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
A = scp.read_labspec("ramandata/labspec/Activation.txt")
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
B = scp.read_labspec("ramandata/labspec/subdir")
_ = B.plot()

# %%
# Uncomment the following line to display all figures when running the script
# directly with Python.
#
# scp.show()
