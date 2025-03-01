# ruff: noqa
"""
Loading RAMAN experimental file
===============================

Here we load experimental LABSPEC spectra and plot them.

"""

# %%
import spectrochempy as scp

# %% [markdown]
# Define the folder where are the spectra:

# %%
datadir = scp.preferences.datadir
ramandir = datadir / "ramandata/labspec"

# %% [markdown]
# Read some data:

# %%
A = scp.read_labspec("Activation.txt", directory=ramandir)
A

# %% [markdown]
# Now plot them:

# %%
A.plot()

# %% [markdown]
# As it is a 2D dataset, we can plot it as an image:

# %%
A.plot_image()

# %% [markdown]
# or a contour plot:

# %%
A.plot_map()

# %% [markdown]
# We can also read the content of a folder, and merge all spectra:

# %%
B = scp.read_labspec(ramandir / "subdir")
B.plot()

# %%
# This ends the example ! The following line can be uncommented if no plot shows when
# running the .py script with python

# scp.show()
