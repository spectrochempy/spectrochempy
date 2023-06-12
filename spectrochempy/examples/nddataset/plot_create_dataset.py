# -*- coding: utf-8 -*-
# %%
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# flake8: noqa
"""
NDDataset creation and plotting example
=======================================
In this example, we create a 3D NDDataset from scratch,
and then we plot one section (a 2D plane)
"""
# %%
# Creation
# --------
# Now we will create a 3D NDDataset from scratch
#
# Data
# ++++++
# here we make use of numpy array functions to create the data for coordinates
# axis and the array of data
import numpy as np

# %%
# As usual, we start by loading the spectrochempy library
import spectrochempy as scp

# %%
# We create the data for the coordinates axis and the array of data
c0 = np.linspace(200.0, 300.0, 3)
c1 = np.linspace(0.0, 60.0, 100)
c2 = np.linspace(4000.0, 1000.0, 100)
nd_data = np.array(
    [
        np.array([np.sin(2.0 * np.pi * c2 / 4000.0) * np.exp(-y / 60) for y in c1]) * t
        for t in c0
    ]
)

# %%
# Coordinates
# +++++++++++
# The `Coord` object allow making an array of coordinates
# with additional metadata such as units, labels, title, etc
coord0 = scp.Coord(
    data=c0, labels=["cold", "normal", "hot"], units="K", title="temperature"
)
coord1 = scp.Coord(data=c1, labels=None, units="minutes", title="time-on-stream")
coord2 = scp.Coord(data=c2, labels=None, units="cm^-1", title="wavenumber")

# %%
# Labels can be useful for instance for indexing
a = coord0["normal"]
print(a)

# %%
# nd-Dataset
# +++++++++++
# The `NDDataset` object allow making the array of data with units, etc...
mydataset = scp.NDDataset(
    nd_data, coordset=[coord0, coord1, coord2], title="Absorbance", units="absorbance"
)
mydataset.description = """Dataset example created for this tutorial.
It's a 3-D dataset (with dimensionless intensity: absorbance )"""
mydataset.name = "An example from scratch"
mydataset.author = "Blake and Mortimer"
print(mydataset)

# %%
# We want to plot a section of this 3D NDDataset:
#
# NDDataset can be sliced like conventional numpy-array...
new = mydataset[..., 0]

# %%
# or maybe more conveniently in this case, using an axis labels:
new = mydataset["hot"]

# %%
# To plot a dataset, use the `plot` command (generic plot).
# As the section NDDataset is 2D, a contour plot is displayed by default.
new.plot()

# %%
# But it is possible to display image
#
# sphinx_gallery_thumbnail_number = 2
new.plot(method="image")

# %%
# or stacked plot
new.plot(method="stack")

# %%
# Note that the scp allows one to use this syntax too:
scp.plot_stack(new)

# %%
# This ends the example ! The following line can be uncommented if no plot shows when
# running the .py script with python

# scp.show()
