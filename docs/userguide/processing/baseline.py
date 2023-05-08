# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
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
#     version: 3.10.10
# ---

# %% [markdown]
# # Baseline corrections
#
# This tutorial shows how to make baseline corrections with **SpectroChemPy** using the `Baseline`class processor or equivalent API or dataset methods.
#
# As prerequisite,
# the user is expected to have read the [Import](../importexport/import.ipynb)
# and [Import IR](../importexport/importIR.ipynb) tutorials.

# %% [markdown]
# Spectrochempy offers two approaches to perform baseline corrections. The first one is to use methods of the NDDataset object directly or via the SpectroChemPy API. The second one is the use of a Baseline object which allows to perform all the correction operations with a maximum of flexibility and settings.

# %% [markdown]
# ## 1. Baseline correction using NDDataset/API methods

# %% [markdown]
# Let's focus on the first approach first.
#
# As usual we start by importing the SpectroChemPy API

# %%
import spectrochempy as scp

# %% [markdown]
# As an example, we can use a dataset consisting of 80 samples of corn measured on a NIR spectrometers. This dataset (and others) can be loaded from http://www.eigenvector.com.

# %%
A = scp.read("http://www.eigenvector.com/data/Corn/corn.mat")[4]

# %% [markdown]
# Add some label for a better reading of the data axis

# %%
A.title = "absorbance"
A.units = "a.u."
A.x.title = "Wavelength"
A.x.units = "nm"

# %% [markdown]
# Now plot the original dataset A:

# %%
prefs = A.preferences
prefs.figure.figsize = (7, 3)
prefs.colormap = "magma_r"
_ = A.plot()

# %%
A[0, 1856.0].data, A[0, 378].data

# %% [markdown]
# ### Detrending
#
# It is quite clear that this spectrum series has an increasing trend with both a vertical shift and a drift.
#
# The `detrend` method can help to remove such trends.

# %% [markdown]
# #### Constant trend
#
# When the trend is simply a shift one can subtract the mean absorbance to each spectrum.

# %%
Ac = A.detrend(order="constant")
_ = Ac.plot()

# %% [markdown]
# #### Linear trend
# But here the trend is clearly closer to a linear trend. So we can use a linear correction
# with `A.detrend(order="linear")` or simply `A.detrend()` as "linear" is the default.

# %%
Al = A.detrend()
_ = Al.plot()

# %% [markdown]
# #### Polynomial trend
#
# If a higher degree of polynomial is necessary, it is possible to use a nonnegative integer scalar to define order (degree).
# Note that for degree 2 and 3, the "quadratic" and "cubic" keywords are also available to define 2 and 3-degree of polynomial.

# %%
Ap = A.detrend(order=2)
_ = Ap.plot()

# %% [markdown]
# #### Detrend independently on several data segment
#
# For this we must define a vector (`bp`) which contains the location of the break-points, which determine the limits of each segments.
#
# For example, let's try on a single spectrum for clarity:

# %%
# without bp
A1 = A[0]
Ab = A1.detrend()
_ = Ab.plot()
A1.plot(clear=False)
ax = (A1 - Ab).plot(clear=False, cmap=None, color="red", ls=":")
_ = ax.set_ylim([-0.3, 0.8])

# %% [markdown]
# <div class="alert alert-info">
# <b>Note</b>
#
#  we use float number to define breakpoint as coordinate.
#  Integer number would mean that we use indice starting at 0 (not the same thing!).
#  in this case, indice 1856 does not exist as the size of the x axis is 700.</div>

# %%
# with bp
bp = [1856.0]
Ab = A1.detrend(bp=bp)
_ = Ab.plot()
A1.plot(clear=False)
ax = (A1 - Ab).plot(clear=False, cmap=None, color="red", ls=":")
_ = ax.set_ylim([-0.3, 0.8])

# %% [markdown]
# ### Automatic linear baseline correction `abc`
#
# When the baseline to remove is a simple linear correction, one can use `abc`.
# This performs an automatic baseline correction. This is very close to detrend(), but use the spectra limit to fit the baseline.

# %%
Aa = A.abc()
_ = Aa.plot()

# %% [markdown]
# Now let's import and plot a typical IR dataset which was recorded during the
# removal of ammonia from a NH4-Y
# zeolite:

# %%
X = scp.read_omnic("irdata/nh4y-activation.spg")[::5]
X[:, 1290.0:890.0] = scp.MASKED

# %% [markdown]
# After setting some plotting preferences and plot it

# %%
prefs = X.preferences
prefs.figure.figsize = (7, 3)
prefs.colormap = "magma"
_ = X.plot()

# %% [markdown]
# ## Background subtraction
#
# Often, particularly for surface species, the baseline is first corrected
# by subtracting a reference spectrum. In this
# example, it could be, for instance, the last spectrum (index -1). Hence:

# %%
Xdiff = X - X[-1]
_ = Xdiff.plot()

# %% [markdown]
# ## Detrend  (# TODO replace with data where the effect is more interesting)
#
# Other simple baseline corrections - often use in preprocessing prior chemometric
# analysis - consist in shifting the spectra or removing a trend. This is done using the detrend() method.

# %% [markdown]
# ### Linear trend
# Subtract the linear trend of each spectrum (This is the default behavior: equivalent to ``X.detrend(order='linear'``))

# %%
_ = X.detrend().plot()

# %% [markdown]
# ### Constant trend
# Subtract the average absorbance to each spectrum

# %%
_ = X.detrend(order="constant").plot()

# %% [markdown]
# ### Polynomial trend

# %%
_ = X.detrend(order=2).plot()

# %% [markdown]
# When the baseline to remove is a simple linear correction, one can use `abc` .
# This performs an automatic baseline correction.

# %%
_ = scp.abc(X).plot()

# %% [markdown]
# ## Advanced baseline correction using the Baseline class processor
#
# The `Baseline` class processor props
#
# 'Advanced' baseline correction basically consists for the user to choose:
#
# - spectral ranges which s/he considers as belonging to the baseline - the type of
# polynomial(s) used to model the
# baseline in and between these regions (keyword: `interpolation` ) - the method used
# to apply the correction to
# spectra: sequentially to each spectrum, or using a multivariate approach
# (keyword: `method` ).
#
# ### Range selection
#
# Each spectral range is defined by a list of two values indicating the limits of the
# spectral ranges, e.g. `[4500.,
# 3500.]` to
# select the 4500-3500 cm$^{-1}$ range. Note that the ordering has no importance and
# using `[3500.0, 4500.]` would
# lead to exactly the same result. It is also possible to formally pick a single
# wavenumber `3750.` .
#
# The first step is then to select the various regions that we expect to belong to
# the baseline

# %%
ranges = (
    [5900.0, 5400.0],
    4550.0,
    [4230.0, 4330.0],
    3780,
    [2100.0, 2000.0],
    [1550.0, 1555.0],
    1305.0,
    840.0,
)

# %% [markdown]
# After selection of the baseline ranges, the baseline correction can be made using a
# sequence of simple commands:
#
# 1. Initialize an instance of Baseline

# %%
blc = scp.Baseline()

# %% [markdown]
# 2. Set the range parameter

# %%
blc.ranges = ranges

# %% [markdown]
# 3. Fit baseline on the dataset

# %%
_ = blc.fit(X)

# %% [markdown]
# 4. Plot the baseline corrected (using the transform method)

# %%
_ = blc.transform().plot()

# %% [markdown]
# <div class='alert alert-info'>
# <b>Note</b>
#
# One can also use the property `corrected` instead of the method `transform()`,
# both giving equivalent results.
#
# ```python
# _ = blc.corrected.plot()
# ```

# %% [markdown]
# Obviously, one can restrict the analysis to one of the spectra only. Below we fit a
# baseline and display the results on the same plot:

# %%
X1 = X[0]
_ = blc.fit(X1)

prefs.figure.figsize = (7, 4)
_ = X1.plot(label="X")
_ = blc.baseline.plot(label="Baseline", clear=False, cmap=None, color="red")
ax = blc.corrected.plot(label="X Corrected", clear=False, color="green", legend="best")

blc.show_regions(ax)

# %% [markdown]
# To examine which ranges has been used, use the `used_ranges` attribute. Note, the
# extrema have been automatically added.

# %%
blc.used_ranges

# %% [markdown]
# To avoid this, set the `include_limits` parameter to False (note it is necessary to
# use a new instance of Baseline, else limits will not be removed from those previously
# set):

# %%
blc.include_limits = False

_ = blc.fit(X1)

prefs.figure.figsize = (7, 4)
_ = X1.plot(label="X")
_ = blc.baseline.plot(label="Baseline", clear=False, cmap=None, color="red")
_ = blc.corrected.plot(
    label="X Corrected", clear=False, color="green", legend="best", ylim=(-0.1, 6)
)

blc.used_ranges

# %% [markdown]
# Clearly in this case it is not a very good idea. See the divergent part around 6000 cm$^{-1}$.

# %% [markdown]
# ### Interpolation method
#
#
# The previous baseline fitting was made using the default parameters for the interpolation, *i.e.*, an interpolation using cubic Hermite spline interpolation:
# `interpolation='pchip'` (`pchip` stands for
# **P**iecewise **C**ubic **H**ermite
# **I**nterpolating **P**olynomial).

# %%
blc.interpolation

# %% [markdown]
# This option triggers the use of
# [scipy.interpolate.PchipInterpolator()](
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html)
# to which we refer the interested readers. The other interpolation method is the
# classical polynomial interpolation (interpolation=`'polynomial'` ) in which case the
# order can also be set (*e.g.*, `order=3` , the default value being 6).
# In this case, the base methods used for the interpolation are those of the
# [polynomial module](
# https://numpy.org/doc/stable/reference/routines.polynomials.polynomial.html)
# of spectrochempy, in particular the
# [numpy.polynomial.polynomial.polyfit()](
# https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.polyfit.html#numpy.polynomial.polynomial.polyfit)
# method.
#
# For the demonstration, we use the same `ranges` definition as above.

# %%
blc = scp.Baseline()
blc.ranges = ranges
blc.include_limits = True
blc.interpolation = "polynomial"
blc.order = 6

# %% [markdown]
# Now as before we fit the baseline on X

# %%
_ = blc.fit(X)

# %% [markdown]
# and display the corrected NDDataset.

# %%
_ = blc.corrected.plot()

# %% [markdown]
# As before, we display the baseline for one of the spectra:
#

# %%
X1 = X[0]
_ = blc.fit(X1)

prefs.figure.figsize = (7, 4)
_ = X1.plot(label="X")
_ = blc.baseline.plot(label="Baseline", clear=False, cmap=None, color="red")
_ = blc.corrected.plot(
    label="X Corrected", clear=False, color="green", legend="best", ylim=(-0.5, 4)
)

blc.used_ranges

# %% [markdown]
# ### Multivariate method
#
# The `method` option defines whether the baseline regions selected in the spectra should be taken "as is" (this is the default method `"sequential"`) or modeled using a multivariate model (method=`'multivariate'`).
#
# The `'multivariate'` option is useful when the signal to noise ratio is low
# and/or when baseline changes in different regions of the spectrum are
# different regions of the spectrum are correlated. It consists of (i) modeling the baseline regions
# by a principal component analysis (PCA), (ii) interpolating the loadings of the first
# principal components over the whole spectrum
# and (iii) model the baselines of the spectra from the product of the PCA scores and the interpolated loadings.
# (For details: see [Vilmin et al. Analytica Chimica Acta 891
# (2015)](http://dx.doi.org/10.1016/j.aca.2015.06.006)).
#
# If this option is selected, the user must also set the `n_components` parameter,
# i.e. the number of principal components used to model the baseline.
# In a sense, this parameter has the same role as the `order` parameter, except that it affects
# how the baseline fits the selected regions on *both dimensions: wavelength and acquisition time*.
# In particular, a large value of `n_components` will lead to an overfitting
# of the baseline variation with time and lead to the same result as the
# while a value that is too small may fail to detect a main component underlying the baseline variation over time.
# Typical optimal values are `n_components=2` or `n_components=3` (see exercises below).
#
# Let's fit the baseline using the `multivariate` method:

# %%
blc.interpolation = "pchip"
blc.method = "multivariate"
blc.n_components = 3

blc.fit(X)

_ = blc.corrected.plot()

# %% [markdown]
# ### Code snippet for 'advanced' baseline correction
# The following code in which the user can change any of the parameters and look at
# the changes after re-running
# the cell:

# %%
# Create a baseline instance and give it a name (here basc)
# ---------------------------------------------------------
basc = scp.Baseline()

# user defined parameters
# -----------------------
basc.ranges = (  # ranges can be pair or single values
    [5900.0, 5400.0],
    [4000.0, 4500.0],
    4550.0,
    [2100.0, 2000.0],
    [1550.0, 1555.0],
    [1250.0, 1300.0],
    [800.0, 850.0],
)
basc.interpolation = "pchip"  # choose 'polynomial' or 'pchip'
basc.order = 5  # only used for 'polynomial'
basc.method = "sequential"  # choose 'sequential' or 'multivariate'
basc.n_components = 3  # only used for 'multivariate'

# fit baseline, plot original and corrected NDDatasets and ranges
# ---------------------------------------------------------------
_ = basc.fit(X)
Xc = basc.corrected

axs = scp.multiplot(
    [X, Xc],
    labels=["Original", "Baseline corrected"],
    sharex=True,
    nrow=2,
    ncol=1,
    figsize=(7, 6),
    dpi=96,
)
basc.show_regions(axs["axe21"])

# %% [markdown]
# ### Widget for "advanced" baseline corrections

# %% [markdown]
# The `BaselineCorrector` widget can be used in either Jupyter notebook or Jupyter
# lab.
#
# The buttons are the following:
# * `upload`: allows to upload new data.
# * `process` : baseline correct and plot original dataset + baseline and corrected
# datasets
# * `save as`: save the baseline corrected dataset.
#
# The `x slice` and `y slice` text boxes can be used to slice the original dataset
# with the usual `[start:stop:step]` format. In both dimensions, coordinates or indexes
# can be used (for example, [3000.0::2] or [:100:5] are valid entries).
#
# The `Method` and `Interpolation` dropdown fields are self-explanatory,
# see above for details.
#
# `Ranges` must be entered as a tuple of digits or wave numbers,
# e.g. `([5900.0, 5400.0], 2000.0, [1550.0, 1555.0],)` .

# %%
X = scp.read_omnic("irdata/nh4y-activation.spg")
out = scp.BaselineCorrector(X)

# %% [markdown]
# After processing, one can get the original (sliced) dataset, corrected dataset
# and baselines
# through the following attributes:

# %%
out.original, out.corrected, out.baseline

# %% [markdown]
# <div class='alert alert-info'>
#     <b>Exercises</b>
#
# **basic:**
# - write commands to subtract (i) the first spectrum from a dataset and (ii)
# the mean spectrum from a dataset
# - write a code to correct the baseline of the last 10 spectra of the above dataset
# in the 4000-3500 cm$^{-1}$ range
#
# **intermediate:**
# - what would be the parameters to use in 'advanced' baseline correction to mimic
# 'detrend' ? Write a code to check
# your answer.
#
# **advanced:**
# - simulate noisy spectra with baseline drifts and compare the performances of
# `multivariate` vs `sequential` methods
# </div>

# %%
