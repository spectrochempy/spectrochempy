# %%
# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
"""
NDDataset creation and plotting example
=======================================
In this example, we create a 3D NDDataset from scratch,
and then we plot one section (a 2D plane)

.. sphinx_gallery_thumbnail_number = 2
"""

# %%
# Creation
# --------
# Now we will create a 3D NDDataset from scratch
import numpy as np

# %%
# As usual, we start by loading the spectrochempy library
import spectrochempy as scp


# %%
# Coordinates
# +++++++++++
# The `Coord` object allows creating an array of coordinates directly with
# ``Coord.linspace``, attaching metadata (units, labels, title) in a single
# step — no separate numpy array needed.
coord0 = scp.Coord.linspace(
    200.0,
    300.0,
    3,
    labels=["cold", "normal", "hot"],
    units="K",
    title="temperature",
)
coord1 = scp.Coord.linspace(0.0, 60.0, 100, units="minutes", title="time-on-stream")
coord2 = scp.Coord.linspace(4000.0, 1000.0, 100, units="cm^-1", title="wavenumber")

# %%
# Labels can be useful for instance for indexing
a = coord0["normal"]
print(a)


# %%
# Data and nd-Dataset
# +++++++++++++++++++
# ``scp.fromfunction`` builds an NDDataset directly from a Python function.
# The function receives the coordinate arrays and returns the intensity values.
def synth_func(temperature, time, wavenumber):
    return np.sin(2.0 * np.pi * wavenumber / 4000.0) * np.exp(-time / 60) * temperature


mydataset = scp.fromfunction(
    synth_func,
    coordset=[coord0, coord1, coord2],
    title="Absorbance",
    units="absorbance",
)
mydataset.description = """Dataset example created for this tutorial.
It's a 3-D dataset (with dimensionless intensity: absorbance )"""
mydataset.name = "An example from scratch"
mydataset.author = "Blake and Mortimer"
print(mydataset)

# %%
# In a Jupyter notebook, the NDDataset is displayed as follows (click on  the arrow on the left to expand the metadata):

mydataset

# %%
# We want to plot a section of this 3D NDDataset:
#
# NDDataset can be sliced like conventional numpy-array...
new = mydataset[..., 0]

# %%
# or maybe more conveniently in this case, using an axis labels:
new = mydataset["hot"]

# %%
# To plot a dataset, use the `plot` method (generic plot).
# As the section NDDataset is 2D, a lines plot is displayed by default. As you can see, the x-axis is in wavenumber
# and the ordinate axis is in absorbance. Note that in this case, the default `NDDataset.plot()` command is equivalent to
# `plot_lines()`.
_ = new.plot()

# %%
# Note also that a colormap ('viridis') has been automatically set for the lines. This is because the y-dimension of the
# dataset has float coordinates (they correspond to a time). In such a case it is easy to add a colorbar explaining the
# colors <-> time value correspondence:

_ = new.plot(colorbar=True)

# %%
# If the y-dimension had no coordinates or consecutive integer coordinates starting by `0`or `1`, a categorical
# color map would have been chosen. The default behavior can be overridden by explicitly passing a colormap. For instance,
# if we want to use a categorical colormap instead of a sequential one, we can do:

_ = new[:, 0:20].plot(cmap="tab20")

# %%
# But it is possible to display image plot instead (note that the x-axis is in wavenumber and
# the y-axis is in time-on-stream)
_ = new.plot_image()

# %%
# or contour plot  (note that
_ = new.plot_map()


# %%
# This ends the example ! The following line can be uncommented if no plot shows when
# running the .py script with python

# scp.show()
