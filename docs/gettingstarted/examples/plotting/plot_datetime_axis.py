# -*- coding: utf-8 -*-
# %%
#  =====================================================================================
#  Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
#  See full LICENSE agreement in the root directory.
#  =====================================================================================
"""
Plotting with datetime axis
===========================


"""

# %%
import spectrochempy as scp

# %% [markdown]
# The location of the spectrochempy_data can be found in preferences

# %%
datadir = scp.preferences.datadir

# %% [markdown]
# Let's read on of the dataset (in `spg` Omnnic format)

# %%
dataset = scp.NDDataset.read_omnic(datadir / "irdata" / "nh4y-activation.spg")

# %% [markdown]
# We can have a pretty print of the dataset using the `print_`method. Note that in a
# jupyter notebook you just need to write
#  ```dataset```
# and an output in a html format will be issued.

# %%
scp.print_(dataset)

# %% [markdown]
# This dataset is actually to dimensional: one dimension (x) is the wavenumber axis (in $cm^1$ units),
# the second one is the acquisition datetime axis. The intensity is in a dimensionless
# units: absorbance.

# %% [markdown]
# By default it is plotted with the stack method

# %%
_ = dataset.plot(figsize=(6, 4))

# %% [markdown]
# Previously plotted as a stack plot, this dataset can also be plotted
# using a contour plot (method: map)

# %%
_ = dataset.plot_map(figsize=(6, 4))

# %% [markdown]
# As you see, spectrochempy does not display the datetimes on the vertical axis,
# because in this case the interval betwwen the date is relatively small. So it s displayed in seconds, with the starting reference noted on the aixs label.

# %% [markdown]
# Let's see another display of the datetime, that is in colorbar:

# %%
_ = dataset.plot(figsize=(6, 4), colorbar=True, colormap="magma")

# %% [markdown]
# **How to change the time units for instance from seconds to hours?**

# %% [markdown]
# The first solution, if we don't need to keep reference to the reference date, it to transform them in timedelta values:

# %%
dataset.y -= dataset.y[0]
scp.print_(dataset.y)

# %% [markdown]
# This way we have subtracted the first value of the y axis array to all other values. The values are now in units of seconds. To change this in hours for instance, just do this:

# %%
dataset.y.ito("hours")
scp.print_(dataset.y)

# %% [markdown]
# And now the result is:

# %%
_ = dataset.plot_map(figsize=(6, 4))

# %%
