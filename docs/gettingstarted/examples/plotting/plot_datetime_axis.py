# =====================================================================================
# Copyright (©) 2015-2022 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# =====================================================================================
"""
Plotting with datetime axis
===========================


"""

# %%
import spectrochempy as scp

# %%
# The location of the spectrochempy_data can be found in preferences

datadir = scp.preferences.datadir

# %%
# Let's read on of the dataset (in `spg` Omnic format)

dataset = scp.NDDataset.read_omnic(datadir / "irdata" / "nh4y-activation.spg")
dataset

# %%
# This dataset is actually to dimensional: one dimension (x) is the wavenumber axis
# (in $cm^1$ units),
# the second one is the acquisition datetime axis. The intensity is in a dimensionless
# units: absorbance.

# %%
# By default it is plotted with the stack method

_ = dataset.plot(figsize=(6, 4))

# %%
# Previously plotted as a stack plot, this dataset can also be plotted
# using a contour plot (method: map)

_ = dataset.plot_map(figsize=(6, 4))

# %%
# As you see, spectrochempy does not display the datetimes on the vertical axis,
# because in this case the interval betwwen the date is relatively small.
# So it s displayed in seconds, with the starting reference noted on the aixs label.

# %%
# Let's see another display of the datetime, that is in colorbar:

# %%
_ = dataset.plot(figsize=(6, 4), colorbar=True, colormap="magma")

# %%
# It is also displayed on 1D sections:

# %%
section = dataset[:, 3200.0]
_ = section.plot(figsize=(6, 3), c="red", lw=3)

# %%
# How to change the time units?
# -----------------------------

# %%
# It may be desirable to change the default units on the datetime axis,
# *e.g.,* from minutes to seconds in the above example.

# %%
# Solution 1
# ++++++++++
#
# The first solution, if we don't need to keep reference to the reference date,
# it to transform them in timedelta values:

# %%
newdataset = (
    dataset.copy()
)  # we copy it as we will need the original dataset for solution 2

newdataset.y -= newdataset.y[0]
newdataset.y

# %%
# This way we have subtracted the first value of the y axis array to all other values.
# The values are now in units of seconds. To change this in hours for instance,
# just do this:

newdataset.y.ito("seconds")
newdataset.y

# %%
# And now the result is:

_ = newdataset.plot_map(figsize=(6, 4))

# %%
# Solution 2
# ++++++++++
#
# Another solution is to pass the keword parameters `time_units` in the plot command.
# This time_units must be a string among : ["days", "hours", "minute", "second", "millisecond", "microsecond", "nanosecond"]

_ = dataset.plot_map(figsize=(6, 4), time_units="hours")

# %%

_ = section.plot(figsize=(6, 3), time_units="seconds", c="red", lw=3)

scp.show()
