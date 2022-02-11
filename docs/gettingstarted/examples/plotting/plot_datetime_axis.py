# %%
"""
Plotting with datetime axis
===========================

In this example, we show how the axis for the datetime dimension is plotted and how we can change its aspect.
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

# Let's set some plotting preferences
prefs = dataset.preferences  # we will use prefs instead of dataset.preference
prefs.figure.figsize = (6, 3)

# %%
# By default it is plotted with the stack method
_ = dataset.plot()

# %%
# Previously plotted as a stack plot, this dataset can also be plotted
# using a contour plot (method: map)
_ = dataset.plot_map()

# %%
# As you see, spectrochempy does not display the dates on the vertical axis,
# because in this case the interval between the dates is relatively small.
# So it is displayed in seconds in the vertical label, relative to the starting
# acquisition datetime.
#
# The acquisition date can be easily read in the `acquisition_date` attribute of the
# dataset (or of the y coordinate - both are identical).
dataset.acquisition_date

# %%
# Let's see another display of the datetime, that is in colorbar:
_ = dataset.plot(colorbar=True, colormap="magma")

# %%
# It is also displayed on 1D sections:
section = dataset[:, 3200.0]
_ = section.plot(c="red", lw=3)

# %%
# How to change the time units?
# -----------------------------
# It may be desirable to change the default units on the datetime axis,
# *e.g.,* from minutes to seconds in the above example.

# %%
# Solution 1
# ++++++++++
# The first solution, if we don't need to keep reference to the acquisition date,
# it to transform them in timedelta values:
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
newsection = newdataset[:, 3200.0]

# %%
# And now the result is:
_ = newdataset.plot_map()
_ = newsection.plot(c="green", lw=3)

# %%
# Solution 2
# ++++++++++
# Another solution is to pass the keyword parameters `time_units` in the plot command.
# This time_units must be a string among : ["days", "hours", "minute", "second", "millisecond", "microsecond", "nanosecond"]
_ = dataset.plot_map(time_units="hours")

# %%
_ = section.plot(time_units="seconds", c="red", lw=3)

# %%
# Final remark
# ++++++++++++
# It is important to note that axis in solution one is a timedelta axis, meaning that
# the information on the actual acquisition date is lost. However it is always
# possible to retrieve it using the acquisition_date attribute of the dataset.
# e.g., the first value
newdataset.y[0].value

# %%
# while
dataset.y[0].value

# %%
scp.show()
