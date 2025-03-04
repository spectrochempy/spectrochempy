# ruff: noqa
"""
Loading RAMAN experimental file
===============================

Here we load experimental LABSPEC spectra and plot them.

"""

# %%
import spectrochempy as scp

# %%
# Define the folder where are the spectra:
datadir = scp.preferences.datadir
ramandir = datadir / "ramandata/labspec"

# %%
# Read some data:
A = scp.read_labspec("Activation.txt", directory=ramandir)
A

# %%
# Now plot them:
A.plot()
# %%
# As it is a 2D dataset, we can plot it as an image:
A.plot_image()
# %%
# or a contour plot:
A.plot_map()
# %%
# We can also read the content of a folder, and merge all spectra:
B = scp.read_labspec(ramandir / "subdir")
B.plot()

# %%
# This ends the example ! The following line can be uncommented if no plot shows when
# running the .py script with python

# scp.show()
