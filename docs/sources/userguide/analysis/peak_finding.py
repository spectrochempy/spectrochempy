# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.10.11
# ---

# %% [markdown]
# # Peak Maxima Finding
#
# This tutorial shows how to find peaks and determine peak maxima with spectrochempy.
# As prerequisite, the user is
# expected to have read the [Import](../importexport/import.rst),
# [Import IR](../importexport/importIR.rst),
# [slicing](../processing/slicing.rst) tutorials.
#

# %% [markdown]
# Fist, as usual, we need to load the API.

# %%
import spectrochempy as scp

# %% [markdown]
# ## Loading an experimental dataset

# %% [markdown]
# A typical IR dataset (CO adsorption on supported CoMo catalyst
# in the 2300-1900 cm-1 region) will be used throughout.

# %% [markdown]
# We load the data using the generic API method  `read` (the type of data is inferred
# from the extension)

# %%
ds = scp.read("irdata/CO@Mo_Al2O3.SPG")

# %%
ds.y -= ds.y.data[0]  # start time a 0 for the  first spectrum
ds.y.title = "time"
ds.y = ds.y.to("minutes")

# %% [markdown]
# Let's set some preferences for plotting

# %%
prefs = ds.preferences
prefs.method_1D = "scatter+pen"
prefs.method_2D = "stack"
prefs.colorbar = True
prefs.colormap = "Dark2"

# %% [markdown]
# We select the desired region and plot it.

# %%
reg = ds[:, 2300.0:1900.0]
reg.plot()

# %% [markdown]
# ## Find maxima by manual inspection of the plot
# Once a given maximum has been approximately located manually with the mouse, it is
# possible to obtain [markdown]
# For instance, after zooming on the highest peak of the last spectrum,
# one finds that it is located at ~ 2115.5 cm$^{-1}$. The exact x-coordinate value can
# be obtained using the
# following code
# (see the [slicing tutorial](../processing/slicing.rst) for more info):

# %%
pos = reg.x[2115.5].values
pos

# %% [markdown]
# We can easily get the list of all individual maximas at this position

# %%
maximas = reg[:, pos].squeeze()
maximas.plot(marker="s", ls="--", color="blue")

# %%
ax = reg.plot()
x = pos.max()
y = maximas.max()
ax.set_ylim(-0.01, 0.3)
_ = ax.annotate(
    f"{x: ~0.2fP} {y: ~.3fP}",
    xy=(2115.5, maximas.max()),
    xytext=(30, -20),
    textcoords="offset points",
    bbox={"boxstyle": "round4,pad=.7", "fc": "0.9"},
    arrowprops={"arrowstyle": "->", "connectionstyle": "angle3"},
)

# %% [markdown]
# ## Find maxima with an automated method: `find_peaks()`
# Exploring the spectra manually is useful, but cannot be made systematically in large
# datasets with many - possibly
# shifting peaks. The maxima of a given spectrum can be found automatically by the
# find_peaks() method which is based
# on [scpy.signal.find_peaks()](
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html).
# It returns two outputs: `peaks` a NDDataset grouping the peak maxima (wavenumbers and
# absorbance) and `properties`
# a dictionary containing properties of the returned peaks (it is empty
# if no particular option is selected,
# `see below<#options>`_ for more information).

# %% [markdown]
# ###  Default behaviour

# %% [markdown]
# Applying this method on the last spectrum without any option will yield 7 peaks. But
# some peaks are very close so we use the option `distance` to avoid this.

# %%
last = reg[-1]
peaks, _ = last.find_peaks(distance=5.0)
# we do not catch the second output (properties) as it is void in this case

# %% [markdown]
# `peaks` is a NDDataset. Its `x` attribute gives the peak position:

# %%
peaks.x.values

# %% [markdown]
# The code below shows how the peaks found by this method can be marked on the plot:

# %%
ax = last.plot_pen()  # output the spectrum on ax. ax will receive next plot too
pks = peaks + 0.02  # add a small offset on the y position of the markers
_ = pks.plot_scatter(
    ax=ax,
    marker="v",
    color="black",
    clear=False,  # we need to keep the previous output on ax
    data_only=True,  # we don't need to redraw all things like labels, etc...
    ylim=(-0.01, 0.35),
)

for p in pks:
    x, y = p.x.values, p.values + 0.02
    _ = ax.annotate(
        f"{x.m: 0.0f}",
        xy=(x, y),
        xytext=(-5, 0),
        rotation=90,
        textcoords="offset points",
    )

# %% [markdown]
# Now we will do a peak-finding for the whole dataset:

# %%
peakslist = [s.find_peaks(distance=5)[0] for s in reg]

# %%
ax = reg.plot()
for peaks in peakslist:
    peaks.plot_scatter(
        ax=ax,
        marker="v",
        ms=3,
        color="red",
        clear=False,
        data_only=True,
        ylim=(-0.01, 0.30),
    )

# %% [markdown]
# It should be noted that this method finds only true maxima, not shoulders (!).
# For the detection of such underlying
# peaks, the use of methods based on derivatives or advanced detection methods -
# which will be treated in separate
# tutorial - are required. Once their maxima of a given peak have been found,
# it is possible, for instance,
# to plot its evolution with, e.g. the time. For instance for the peaks located
# at 2220-2180 cm$^{-1}$:

# %%
# Find peak's position
positions = [s.find_peaks(distance=5)[0].x.values for s in reg[:, 2220.0:2180.0]]

# Make a NDDataset
evol = scp.NDDataset(positions, title="wavenumber at the maximum")
evol.x = scp.Coord(
    reg.y, title="acquisition time"
)  # the x coordinate is st to the acquisition time for each spectra
evol.preferences.method_1D = "scatter+pen"

# plot it
evol.plot(ls=":")

# %% [markdown]
# ###  Options of `find_peaks()` <a id='options'></a>
# The default behaviour of find_peaks() will return *all* the detected maxima.
# The user can choose various options to
# select among these peaks:
#
# **Parameters relative to "peak intensity":**
# - `height` : minimal required height of the peaks (single number) or minimal
# and maximal heights
# (sequence of two numbers)
# - `prominence` : minimal prominence of the peak to be detected
# (single number) or minimal and maximal prominence (sequence of 2 numbers). In brief
# the "prominence" of a peak
# measures how much a peak stands out from its surrounding and is the vertical distance
# between the peak and its
# lowest "contour line". It should not be confused with the height as a peak can have
# an important height but a small
# prominence when surrounded by other peaks (see below for an illustration).
#     - in addition to the prominence, the user can define `wlen` , the width (in points)
#     of the window used to look
#     at neighboring minima, the peak maximum being is at the center of the window.
# - `threshold` : a single number (the minimal required threshold) or a sequence of two
# numbers (minimal and maximal).
# The thresholds are the difference of height of the
# maximum with its two neighboring points (useful to detect spikes for instance)
#
# **Parameters relative to "peak spacing":**
# - `distance` : the required minimal horizontal distance between neighbouring peaks.
# Smaller peaks are removed first.
# - `width` : Required minimal width of peaks in samples (single number) or minimal and
# maximal width. The width is
# assessed from the peak height,
# prominence and neighboring signal. - In addition the user can define `rel_height`
# (a float between 0. and 1.) used
# to compute the width - see the [scipy documentation](
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.peak_widths.html)
# for further details.
# - Finally, we mention en passant a last parameter, `plateau_size()` used for
# selecting peaks having truly flat tops (
# as in e.g. square-boxed signals).
#
# The use of some of these options for the last spectrum of the dataset is
# exemplified in the following:

# %%
s = reg[-1].squeeze()

# %% [markdown]
# we use squeeze it because one of the dimensions for this dataset of shape (1, N)
# is useless

# %%
# default settings

peaks, properties = s.find_peaks()
ax = s.plot_pen(color="black")
peaks.plot_scatter(
    ax=ax,
    label="default",
    marker="v",
    ms=4,
    color="black",
    clear=False,
    data_only=True,
    ylim=(-0.01, 0.35),
)

# find peaks higher than 0.05  (NB: the spectra are shifted for the display.
# Refer to the 1st spectrum for true heights)

peaks, properties = s.find_peaks(height=0.05)
color = "blue"
label = "0.05<height"
offset = 0.05
(s + offset).plot_pen(color=color, clear=False)
(peaks + offset).plot_scatter(
    ax=ax, label=label, m="v", mfc=color, mec=color, ms=5, clear=False, data_only=True
)

# find peaks heights between 0.05 and 0.2 (the highest peak won't be detected)

peaks, properties = s.find_peaks(height=(0.05, 0.2))
color = "green"
label = "0.05<height<0.2"
offset = 0.1
(s + offset).plot_pen(color=color, clear=False)
(peaks + offset).plot_scatter(
    ax=ax, label=label, m="v", mfc=color, mec=color, ms=5, clear=False, data_only=True
)

# find peaks with prominence >= 0.05 (only the two most prominent peaks are detected)

peaks, properties = s.find_peaks(prominence=0.05)
color = "purple"
label = "prominence=0.05"
offset = 0.15
(s + offset).plot_pen(color=color, clear=False)
(peaks + offset).plot_scatter(
    ax=ax, label=label, m="v", mfc=color, mec=color, ms=5, clear=False, data_only=True
)

# find peaks with distance >= 10 (only the highest of the two maxima at ~ 2075 is
# detected)

peaks, properties = s.find_peaks(distance=10)
color = "red"
label = "distance>10"
offset = 0.20
(s + offset).plot_pen(color=color, clear=False)
(peaks + offset).plot_scatter(
    ax=ax, label=label, m="v", mfc=color, mec=color, ms=5, clear=False, data_only=True
)

# find peaks with width >= 10 (none of the two maxima at ~ 2075 is detected)

peaks, properties = s.find_peaks(width=10)
color = "grey"
label = "width>10"
offset = 0.25
(s + offset).plot_pen(color=color, clear=False)
(peaks + offset).plot_scatter(
    ax=ax, label=label, m="v", mfc=color, mec=color, ms=5, clear=False, data_only=True
)

_ = ax.legend(fontsize=6)

# %% [markdown]
# ### More on peak properties
#
# If the concept of "peak height" is pretty clear, it is worth examining further some
# peak properties as defined and
# used in `find_peaks()` . They can be obtained (and used) by passing the parameters
# `height` , `prominence` ,
# `threshold`  and `width` . Then `find_peaks()` will return the corresponding properties
# of the detected peaks in the
# `properties` dictionary.
#
# #### Prominence
#
# The prominence of a peak can be defined as the vertical distance from the peak’s
# maximum to *the lowest horizontal
# line passing through a minimum but not containing any higher peak*. This is
# illustrated below for the three most
# prominent peaks of the above spectra:
#
# <img src="images/prominence.png" alt="prominence_def" width="350" align="center" />
#
# Let's illustrate this for the second-highest peak which height is comprised between
# ~ 0.15 and 0.22 and see which
# properties are returned when, on top of `height` , we pass `prominence=0`: this will
# return the properties
# associated to the prominence and warrant that this peak will not be rejected on the
# prominence criterion.

# %%
peaks, properties = s.find_peaks(height=(0.15, 0.22), prominence=0)
properties

# %% [markdown]
# The actual prominence of the peak is this 0.0689, a value significantly lower
# that is peak height ( [markdown]
# The peak prominence is 0.0689, a much lower value than the height (0.1995), as
# could be expected by the illustration
# above.
#
# The algorithm used to determine the left and right 'bases' is illustrated below:
# - (1) extend a line to the left and right of the maximum until it reaches the window
# border (here on
# the left) or the signal (here on the right).
# - (2) find the minimum value within the intervals defined above. These
# points are the peak's bases.
# - (3) use the higher base (here the right base) and peak maximum to calculate the
# prominence.
#
# <img src="images/prominence_algo.png" alt="prominence_algo" width="350"
# align="center" />
#
# The following code shows how to plot the maximum and the two "base points" from the
# previous output of `find_peaks()`:

# %%
ax = s.plot_pen()

# plots the  maximum
_ = peaks.plot_scatter(
    ax=ax, marker="v", mfc="green", mec="green", data_only=True, clear=False
)

wl, wr = properties["left_bases"][0], properties["right_bases"][0]

# wavenumbers of left and right bases
for w in (wl, wr):
    ax.axvline(w, linestyle="--")  # add vertical line at the bases
    ax.plot(w, s[w].data, "v", color="red")
    # and a red mark  #TODO: add function to plot this easily

ax = ax.set_xlim(2310.0, 1900.0)  # change x limits to better see the 'left_base'

# %% [markdown]
# It leads to base marks at their expected locations. We can further check that the
# prominence of the [markdown]
# We can check that the correct value of the peak prominence is obtained by the
# difference between its height
# and the highest base, here the 'right_base':

# %%
prominence = peaks[0].values - s[wr].values
print(f"calc. prominence = {prominence: 0.4f}")

# %% [markdown]
# Finally, we illustrate how the use of the `wlen` parameter - which limits the
# search of the "base [markdown]
# Finally, the figure below shows how the prominence can be affected by `wlen` , the
# size of the window used to determine the peaks' bases.
#
# <img src="images/prominence_wlen.png" alt="prominence_def"
# width="700" align="center" />
#
# As illustrated above a reduction of the window should reduce the prominence of the
# peak. This impact can be checked
# with the code below:

# %%
peak, properties = s.find_peaks(height=0.2, prominence=0)
print(f"prominence with full spectrum: {properties['prominences'][0]: 0.4f}")

peak, properties = s.find_peaks(
    height=0.2, prominence=0, wlen=50.0
)  # a float should be explicitly passed, else will be considered as points
print(f"prominence with reduced window: {properties['prominences'][0]: 0.4f}")

# %% [markdown]
# #### Width

# %% [markdown]
# The peak widths, as returned by `find_peaks()` can be *very approximate* and for
# precise assessment,
# The find_peaks() method also returns the peak widths. As we will see below, the
# method is **very approximate** and
# more advanced methods (such as peak fitting), also implemented in spectrochempy
# should be used (see e.g.,
# [this example](../../gettingstarted/examples/gallery/auto_examples_analysis/c_curvefitting/plot_fit.html)).
# On the other hand, **the magnitude of the width is generally fine**.
#
# This estimate is based on an algorithm similar to that used for the "bases" above,
# except that the horizontal
# line starts from a `width_height` computed from the peak height subtracted by a
# *fraction* of the peak prominence
# defined bay `rel_height` (default = 0.5). The algorithm is illustrated below for the
# two most prominent peaks:
#
# <img src="images/width_algo.jpg" alt="width_algo" width="900" align="center" />
#
# When the `width` keyword is used, `properties` dictionary returns the prominence
# parameters (as it is used for
# the calculation of the width), the width and the left and right interpolated positions
# ("ips") of the intersection
# of the horizontal line with the spectrum:
# %%
peaks, properties = s.find_peaks(height=0.2, width=0)
properties

# %% [markdown]
# The code below shows how these heights and widths can be extracted from the
# dictionary and plotted [markdown]
# The code below shows how these data can be extracted and then plotted:

# %%
# extraction of data (for better readability of the code below)

height = properties["peak_heights"][0]
width_height = properties["width_heights"][0]
wl = properties["left_ips"][0]
wr = properties["right_ips"][0]

ax = s.plot_pen()
_ = peaks.plot_scatter(
    ax=ax, marker="v", mfc="green", mec="green", data_only=True, clear=False
)
_ = ax.axhline(height, linestyle="--", color="blue")
_ = ax.axhline(width_height, linestyle="--", color="red")
_ = ax.axvline(wl, linestyle="--", color="green")
_ = ax.axvline(wr, linestyle="--", color="green")

# %% [markdown]
# As stressed above, we see here that the peak width is very approximate and probably
# exaggerated in
# It is obvious here that the peak width is overestimated in the present case due to the
# presence of the second peak on
# the left. Here a better estimate would be obtained by considering the right
# half-width, or reducing the `rel_height`
# parameter as shown below.

# %% [markdown]
# ### A code snippet to display properties
#
# The self-contained code snippet below can be used to display in a matplotlib plot
# and print the various
# peak properties of a single peak as returned by `find_peaks()`:

# %%
# user defined parameters ------------------------------

s = reg[-1]  # define a single-row NDDataset
s.preferences.method_1D = "pen"

# peak selection parameters; should be set to return a single peak

height = 0.08  # minimal height or min and max heights)
prominence = 0.0  # minimal prominence or min and max prominences
width = 0.0  # minimal width or min and max widths
threshold = None  # minimal threshold or min and max threshold)

# prominence and width parameter

wlen = None  # the length of the window used to compute the prominence
rel_height = 0.47  # the fraction of the prominence used to compute the width

# code: find peaks, plot and print properties -------------------

peaks, properties = s.find_peaks(
    distance=10,
    height=height,
    prominence=prominence,
    wlen=wlen,
    threshold=threshold,
    width=width,
    rel_height=rel_height,
)

table_pos = "  ".join([f"{peaks[i].x.value.m: >10.3f}" for i in range(len(peaks))])
print(f"{'peak_position (cm⁻¹)': >26}: {table_pos}")
for key in properties:
    table_property = "  ".join(
        [f"{properties[key][i].m: >10.3f}" for i in range(len(peaks))]
    )
    title = f"{key: >.16} ({properties[key][0].u: ~P})"
    print(f"{title: >26}: {table_property}")

ax = s.plot()
peaks.plot_scatter(
    ax=ax, marker="v", mfc="green", mec="green", data_only=True, clear=False
)

for i in range(len(peaks)):
    for w in (properties["left_bases"][i], properties["right_bases"][i]):
        ax.plot(w, s[0, w].data.T, "v", color="red")
    for w in (properties["left_ips"][i], properties["right_ips"][i]):
        ax.axvline(w, linestyle="--", color="green")

# %% [markdown]
# -- this is the end of this tutorial --
