# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # TUTORIAL: Peak finding, part 1: maxima
#
# This tutorial shows how to find peaks and determine peak maxima with spectrochempy. As prerequisite, the user is
# expected to have read the [Import](../IO/import.ipynb), [Import IR](../IO/importIR.ipynb),
# [slicing](../processing/slicing.ipynb) tutorials. First lets import the modules that will be used in this tutorial:
#

# %% [markdown] execution={"iopub.execute_input": "2020-06-22T12:21:23.143Z", "iopub.status.busy": "2020-06-22T12:21:23.097Z",
# ####  `%matplotlib qt`  or `%matplotlib widget`

# %% [markdown]
# In `Jupyter Lab` (or `Jupyter Notebook`) the use of the **"magic command "** `%matplotlib widget` triggers the interactive drawing. This process is integrated in the Notebook with basic tools to navigate inside the figure. As shown below, from top to bottom of the
# sidebar :
#
# - hide/show tools,
# - reset the view ("home"),
# - previous view ( left arrow ),
# - next view ( right arrow ),
# - move ("arrow cross"),
# - zoom ( rectangle ),
# - save the image ("disk").
#
# <img src="figures/widgetsmode.png" alt="widgets mode" width="700" align="center" />
#
# In recent versions of **SpectroChemPy** (i.e. from versions > 0.1.21) this is the default setting and you do not need to run this command . 
# Another possibility is to use the magic command `%matplotlib qt` instead. In this case the plot is generated in an external window, with some interactive capability. 
#
# <div class = "alert alert alert warning">
# <WARNING: </b> This selection must be made before loading the API, otherwise it will not be taken into account (and the widget <em>%matplotlib</em> will be used by default)
# </div>
#
# In both these interactive mode, the current `x` and `y` coordinate values are shown when the mouse pointer is moved inside  the plot limits.
#
# If you prefer to generate static plots (for a laboratory notebook for example), just enter `%matplotlib inline`.

# %% [markdown]
# Well, this said, let's us start the tutorial by importaning some data

# %% [markdown]
# ## Loading an expeirmental dataset

# %% [markdown]
# A typical IR dataset (CO adsorption on supported CoMo catalyst in the 2300-1900 cm-1 region)  will be used throughout.
#
# Fist we need to loas the API.

# %% execution={"iopub.execute_input": "2020-06-22T12:21:23.143Z", "iopub.status.busy": "2020-06-22T12:21:23.097Z",
import spectrochempy as scp

# %% [markdown]
# Then we load the data using the genric API method  `read` (the type of data is inferred from the extension) 

# %% execution={"iopub.execute_input": "2020-06-22T12:21:31.950Z", "iopub.status.busy": "2020-06-22T12:21:31.942Z",
ds = scp.read('irdata/CO@Mo_Al2O3.SPG')

# %% [markdown]
# Let's set some preferences for plotting

# %%
ds.plotmeta.method_1D = 'scatter+pen'
ds.plotmeta.method_2D = 'stack'
ds.plotmeta.colormap_stack='Dark2'

# %% [markdown] execution={"iopub.execute_input": "2020-06-22T12:21:31.950Z", "iopub.status.busy": "2020-06-22T12:21:31.942Z",
# We select the desired region and plot it.

# %% execution={"iopub.execute_input": "2020-06-22T12:21:31.950Z", "iopub.status.busy": "2020-06-22T12:21:31.942Z",
reg = ds[:, 2300.:1900.]
_ = reg.plot()

# %% [markdown]
# ## Find maxima by manual inspection of the plot
# %% Once a given maximum has been approximately located manually with the mouse, it is possible to obtain [markdown]
# For instance, after zooming on the highest peak of the last spectrum,
# one finds that it is locate at ~ 2115.5 cm$^{-1}$. The exact x-coordinate value can the be obtained using the following code
# (see the [slicing tutorial](../processing/slicing.ipynb) for more info):

# %%
pos = reg.x[2115.5].values
pos

# %% [markdown]
# We can easily get the list of all individual maximas at this position

# %%
maximas = reg[:, pos.m].squeeze()
_ = maximas.plot(marker='s', ls='--', color='blue')

# %% execution={"iopub.execute_input": "2020-06-22T12:21:38.304Z", "iopub.status.busy": "2020-06-22T12:21:38.285Z",
ax = reg.plot()
x = pos.max()
y = maximas.max()
_ = ax.annotate(f'{x:~0.2fP} {y:~.3fP}', xy=(2115.5, maximas.max()), xytext=(30,-20), textcoords='offset points',
            bbox=dict(boxstyle="round4,pad=.7", fc="0.9"), arrowprops=dict(arrowstyle="->", 
                                                                           connectionstyle="angle3"))

# %% [markdown]
# ## Find maxima with an automated method: `find_peaks()`
#
# Exploring the spectra manually is useful, but cannot be made systematically in large datasets with many - possibly
# shifting peaks. The maxima of a given spectrum can be found automatically by the find_peaks() method which is based
# on [scpy.signal.find_peaks()](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html).
# It returns two outputs: `peaks` a NDDataset grouping the peak maxima (wavenumbers and absorbances) and `properties`
# a dictionary containing properties of the returned peaks (it is empty if no particular option is selected,
# see section 2.3. for more information).
#
# ###  Default behaviour
# Applying this method on the last spectrum without any option will yield 7 peaks :

# %%
peaks, properties = X[-1].find_peaks()  # apply find peaks
peaks.x  # "peaks" is a NDDataset. Its x Coord gives the peak position

# %% [markdown]
# The code below shows how the peaks found by this method can be marked:

# %%
ax = X.plot(colormap='Dark2')
(peaks+0.01).plot_scatter(ax=ax, marker='v', color='black', clear=False, data_only=True, ylim=(-0.01, 0.30))

# %% [markdown]
# and for the whole dataset:

# %%
ax = X.plot(colormap='magma')
for s in X:  # loop over rows (= spectra)
    peaks, prop = s.find_peaks()  # find peaks
    (peaks+0.01).plot_scatter(ax=ax, marker='v', color='red', clear=False, data_only=True, ylim=(-0.01, 0.30))

# %% [markdown]
# It should be noted that this method finds only true maxima, not shoulders (!). For the detection of such underlying
# peaks, the use of methods based on derivatives or advanced detection methods - which will be treated in separate
# tutorial - are required. Once their maxima of a given peak have been found, it is possible, for instance,
# to plot its evolution with, e.g. the time. For instance for the peaks located at 2220-2180 cm$^{-1}$:

# %%
maxwn = []  # empty list, will contain wavenumbers at the maximum
for s in X:  # loop over  spectra
    peak, prop = s[:, 2220.:2180.].find_peaks()  # find peak
    maxwn.append(peak.x.data[0])  # append the wavenumber
time = (X.y - X.y[0]).to("minute") # return a NDDataset of time in minutes, relative to the 1st spectrum

coordm = scp.Coord(maxwn, title="wavenumber at maximum", units="1 /cm")
nd = scp.NDDataset(time.data, coordset=[coordm], title="acquisition time", units=time.units)
nd.plot_scatter(rcolor='blue', marker ='s')
scp.show()

# %% [markdown]
# ###  Options of `find_peaks()`
#
# The default behaviour of find_peaks() will return *all* the detected maxima. The user can choose various options to
# select among these peaks:
#
# **Parameters relative to "peak intensity":**
# - `height`: minimal required height of the peaks (single number) or minimal and maximal heights
# (sequence of two numbers)
# - `prominence`: minimal prominence of the peak to be detected
# (single number) or minimal and maximal prominence (sequence of 2 numbers). In brief the "prominence" of a peak
# measures how much a peak stands out from its surrounding and is the vertical distance between the peak and its
# lowest "contour line". It should not be confused with the height as a peak can have an important height but a small
# prominence when surronded by other peaks (see below for an illustration).
#     - in addition to the prominence, the user can define `wlen`, the width (in points) of the window used to look
#     at neighboring minima, the peak maximum being is at the center of the window.
# - `threshold`: a single number (the minimal required threshold) or a sequence of two numbers (minimal and maximal).
# The thresholds are the difference of height of the
# maximum with its two neighboring points (useful to detect spikes for instance)
#
# **Parameters relative to "peak spacing":**
# - `distance`: the required minimal horizontal distance between neighbouring peaks. Smaller peaks are removed first.
# - `width`: Required minimal width of peaks in samples (single number) or minimal and maximal width. The width is
# assessed from the peak height,
# prominence and neighboring signal. - In addition the user can define `rel_height` (a float between 0. and 1. used
# to compute the width - see the [scipy documentation](
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.peak_widths.html)  for further details.
# - Finally we mention en passant a last parameter, `plateau_size()` used for selecting peaks having truly flat tops (
# as in e.g. square-boxed signals).
#
# The use of some of these options for the last spectrum of the dataset is examplified in the following:

# %%
s = X[-1]

# default settings
peaks, properties = s.find_peaks()
s.plot()
for peak in peaks:
    plt.plot(peaks.x, peaks.data.T, 'v', color='black')

# find peaks heights than 0.05  (NB: the spectra are shifted for the display.
# Refer to the 1st spectrum for true heights)
peaks, properties = s.find_peaks(height=0.05)
(s + 0.05).plot(clear=False)
for peak in peaks:
    plt.plot(peaks.x, peaks.data.T + 0.05, 'v', color='blue')

# find peaks heights between 0.05 and 0.2 (the highest peak won't be detected)
peaks, properties = s.find_peaks(height=(0.05, 0.2))
(s + 0.1).plot(clear=False)
for peak in peaks:
    plt.plot(peaks.x, peaks.data.T + 0.1, 'v', color='red')

# find peaks with prominence >= 0.05 (only the two most prominent peaks are detected)
peaks, properties = s.find_peaks(prominence=0.05)
(s + 0.15).plot(clear=False)
for peak in peaks:
    plt.plot(peaks.x, peaks.data.T + 0.15, 'v', color='purple')

# find peaks with distance >= 10 (only tht highest of the two maxima at ~ 2075 is detected)
peaks, properties = s.find_peaks(distance=10)
(s + 0.20).plot(clear=False)
for peak in peaks:
    plt.plot(peaks.x, peaks.data.T + 0.20, 'v', color='green')

# find peaks with width >= 10 (none of the two maxima at ~ 2075 is detected)
peaks, properties = s.find_peaks(width=10)
(s + 0.25).plot(clear=False)
for peak in peaks:
    plt.plot(peaks.x, peaks.data.T + 0.25, 'v', color='grey')

# %% [markdown]
# ### More on peak properties
#
# If the concept of "peak height" is pretty clear, it is worth examining further some peak properties as defined and
# used in `find_peaks()`. They can be obtained (and used) by passing the parameters `height`, `prominence`,
# `threshold`  and `width`. Then `find_peaks()` will return the corresponding properties of the detected peaks in the
# `properties` dictionary.
#
# #### Prominence
#
# The prominence of a peak can be defined as the vertical distance from the peak’s maximum to the *lowest horizontal
# line passing through a minimum but not containing any higher peak*. This is illustrated below for the three most
# prominent peaks of the above spectra:
#
# <img src="figures/prominence.png" alt="prominence_def" width="350" align="center" />
#
# Let's illustrate this for the second highest peak which height is comprised between ~ 0.15 and 0.22 and see which
# properties are returned when, on top of `height`, we pass `prominence=0`: this will return the properties
# associated to the prominence and warrant that this peak will not be rejected on the prominence criterion.

# %%
peaks, properties = s.find_peaks(height=(0.15, 0.22), prominence=0)
properties

# %% The actual prominence of the peak is this 0.0689, a value significantly lower that is peak height ( [markdown]
# The peak prominence is 0.0689, a much lower value than the height (0.1995), as could be expected by the illustration
# above.
#
# The algorithm used to determine the left and right 'bases' is illustrated below:
# - (1) extend a line to the left and right of the maximum until until it reaches the window border (here on
# the left) or the signal (here on the right.
# - (2) find the minimum value within the intervals defined above. These
# points are the peak's bases.
# - (3) use the higher base (here the right base) and peak maximum to calculate the
# prominence.
#
# <img src="figures/prominence_algo.png" alt="prominence_algo" width="350" align="center" />
#
# The following code shows how to plot the maximum and the two "base points" from the previous output of `find_peaks()`:

# %%
ax = s.plot()
plt.plot(peaks.x, peaks.data[0, 0], 'v', color='green')  # plots the  maximum
wl, wr = properties['left_bases'][0], properties['right_bases'][0]  # wavenumbres of of left and right bases
for w in (wl, wr):
    plt.axvline(w, linestyle='--')  # add vertical line at the bases
    plt.plot(w, s[0, w].T, 'v', color='red')  # and a red mark
ax = ax.set_xlim(2310.0, 1900.0)  # change x limits to better see the 'left_base'

# %% It leads to base marks at their expected locations. We can further check that the prominence of the [markdown]
# We can check that the correct value of the peak prominence is obtained by the difference between its height
# and the highest base, here the 'right_base':

# %%
print("calc. prominence={:f}".format((peaks - s[:, wr]).data[0, 0]))

# %% Finally, we illustrate how the use of the `wlen` parameter - which limits the search of the "base [markdown]
# Finally, the figure below shows how the prominence can be affected by `wlen`, the size of the window used to
#  determine the peaks' bases.
#
# <img src="figures/prominence_wlen.png" alt="prominence_def" width="700" align="center" />
#
# As illustrated above a reduction of the window should reduce the prominence of the peak. This impact can be checked
# with the code below:

# %%
peak, prop = s.find_peaks(height=0.2, prominence=0)
print("prominence with full spectrum: {:f}".format(prop['prominences'][0]))

peak, prop = s.find_peaks(height=0.2, prominence=0,
                          wlen=50.0)  # a float should be explicitely passed, else will be considered as points
print("prominence with reduced window: {:f}".format(prop['prominences'][0]))

# %% [markdown]
# #### Width

# %% The peak widths, as returned by `find_peaks()` can be *very approximate* and for precise assessment, [markdown]
# The find_peaks() method also returns the peak widths. As we will see below, the method is **very approximate** and
# more advanced methods (such as peak fitting, also implemented in spectrochempy - see [this example]
# (https://www.spectrochempy.fr/dev/gallery/auto_examples/fitting/plot_fit.html#sphx-glr-gallery-auto-examples-fitting
# -plot-fit-py) should be used. On the other hand, **the magnitude of the width is generally fine**.
#
# This estimate is based on an algorithm similar to that used for the "bases" above, except that the horizontal
# line starts from a `width_height` computed from the peak height subtracted by a *fraction* of the peak prominence
# defined bay `rel_height` (default = 0.5). The algorithm is illustrated below for the two most prominent peaks:
#
# <img src="figures/width_algo.jpg" alt="width_algo" width="900" align="center" />
#
# When the `width` keyword is used, `properties` disctionarry returns the prominence parameters (as it is used for
# the calculation of the width), the width and the left and right interpolated positions ("ips") of the intersection
# of the horizontal line with the spectrum:

# %%
peak, prop = s.find_peaks(height=0.2, width=0)
prop

# %% The code below shows how these heights and widths can be extracted from the dictionary and plotted [markdown]
# The code below shows howe these data can be extracted and then plotted using the matplotlib library:

# %%
# extraction of data (for better readbility pof the code below)
height = prop['peak_heights'][0]
width_height = prop['width_heights'][0]
wl = prop['left_ips'][0]
wr = prop['right_ips'][0]

s.plot()
plt.axhline(height, linestyle='--', color='blue')
plt.axhline(width_height, linestyle='--', color='red')
plt.axvline(wl, linestyle='--', color='green')
plt.axvline(wr, linestyle='--', color='green')

# %% As stressed above, we see here that the peak width is very approximate and probably exaggerated in [markdown]
# It is obvious here that the peak width is overestimated in the present case due to the presence of the second peak on
# the left. Here a better estimate would be obtained by considering the right half-width, or reducing the `rel_height`
# parameter as shown below.

# %% [markdown]
# ### A code snippet to display properties
#
# The self-contained code snippet below can be used to display in a matplolib plot and print the various
# peak properties of a single peak as returned by `find_peaks()`:

# %%
# user defined parameters ------------------------------

s = X[-1]  # define a single-row NDDataset
# peak selection parameters; should be set to return a single peak
height = 0.2  # minimal height or min and max heights)
prominence = 0.0  # minimal prominence or min and max prominences
width = 0.0  # minimal width or min and max widths
threshold = None  # minimal threshold or min and max threshold)

# prominence and width parameter
wlen = None  # the length of the window used to compute the prominence
rel_height = 0.47  # the fraction of the prominence used to compute the width

# code: find peaks, plot and print properties -------------------
peak, prop = s.find_peaks(height=height, prominence=prominence, wlen=wlen,
                          threshold=threshold, width=width, rel_height=rel_height)
s.plot()
plt.plot(peak.x, peak.data[0, 0], 'v', color='blue')
for w in (prop['left_bases'][0], prop['right_bases'][0]):
    plt.plot(w, s[0, w].data.T, 'v', color='red')
for w in (prop['left_ips'][0], prop['right_ips'][0]):
    plt.axvline(w, linestyle='--', color='green')

print('{:>16}: {:<8.4f}'.format("peak_maximum", peak.x.data[0]))
for key in prop:
    print('{:>16}: {:<8.4f}'.format(key[:-1], prop[key][0]))

# %% [markdown]
# -- this is the end of this tutorial --
