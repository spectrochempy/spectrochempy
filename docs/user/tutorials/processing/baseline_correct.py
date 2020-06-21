# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Baseline corrections
#
# This tutorial shows how to make baseline corrections with spectrochempy.
# As prerequisite, the user is expected to have read the [Import](../IO/import.ipynb)
# and [Import IR](../IO/importIR.ipynb) tutorials.

# %%
import spectrochempy as scp
import matplotlib.pyplot as plt  # will be used for some plots

# %% [markdown]
# Now let's import and plot a typical IR dataset which wase recorded during the removal of ammonia from a NH4-Y zeolite:

# %%
X = scp.read_omnic("irdata/nh4y-activation.spg")
subplot1 = X.plot()

# %% [markdown]
# ## 1. background subtraction
#
# Often, particularly for surface species, the baseline is first corrected by subtracting a reference spectrum. In this
# example, it could be, for instance, the last spectrum (index -1). Hence:

# %%
Xdiff = X - X[-1]
subplot2 = Xdiff.plot()

# %% [markdown]
# ## 2. detrend
#
# Other simple baseline corrections - often use in preprocessing prior chemometric analysis - constist in shifting
# the spectra or removing a linear trend. This is done using the detrend() method, which is a wrapper of the [
# detrend() method](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.detrend.html) from the [
# scipy.signal](https://docs.scipy.org/doc/scipy/reference/signal.html) module to which we refer the interested reader.

# %%
subplot3 = X.detrend().plot()  # subtract the linear trend of each spectrum (type='linear', default)
subplot4 = X.detrend(type='constant').plot()  # subtract the average absorbance to each spectrum

# %% [markdown]
# ## 3. "Advanced" baseline correction
#
# 'Advanced' baseline correction basically consists for the user to choose:
#
# - spectral ranges which s/he considers as belonging to the base line - the type of polynomial(s) used to model the
# baseline in and between these regions (keyword: `interpolation`) - the method used to apply the correction to
# spectra: sequentially to each spectrum, or using a multivariate approach (keyword: `method`).
#
# ### 3.1. Range selection
#
# Each spectral range is defined by an array indicating the limits of the spectral ranges, e.g. `[4500., 3500.]` to
# select the 4500-3500 cm$^{-1}$ range. Note that the ordering has no importance and using `[3500.0, 4500.]` would
# lead to exactly the same result. At present it is not possible to formally pick a single wavenumber. For instance
# using `[3750.]` would lead to an error. A workaround is to repeat the wavenumber, for instance: `[3750.,3750.]`.
#
# After selection of the baseline ranges, the baseline correction can be made using this sequence of 2 commands (the
# 3rd one will plot the result):
#

# %%
blc = scp.BaselineCorrection(X)  # initialize an instance of BaselineCorrection
Xcorr = blc.compute([5900.0, 5400.0], [4500., 4000.], [2100., 2000.0], [1550., 1555.])  # compute baseline
subplot5 = Xcorr.plot()  # blc.corrected.plot() would lead to the same result

# %% [markdown]
# ### 3.2. Interpolation method
#
#
# The previous correction was made using the default parameters for the interpolation ,i.e. an interpolation using
# cubic Hermite spline interpolation: `interpolation='pchip'` (`pchip` stands for **P**iecewise **C**ubic **H**ermite
# **I**nterpolating **P**olynomial). This option triggers the use of [scipy.interpolate.PchipInterpolator()]
# (https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html) to which we refer
# the interested readers. The other interpolation method is the classical polynomial interpolation
# (`interpolation='polynomial'`) in which case the order can also be set (e.g. `order=3`, the default value being 6).
# In this case, the base methods used for the interpolation are those of the [polynomial module](
# https://numpy.org/doc/stable/reference/routines.polynomials.polynomial.html) of spectrochempy, in particular the [
# polyfit()](https://numpy.org/doc/stable/reference/generated/numpy.polynomial.polynomial.polyfit.html#numpy
# .polynomial.polynomial.polyfit) method.
#
# For instance:

# %%
ranges = [[5900.0, 5400.0], [4000., 4500.], [2100., 2000.0], [1550., 1555.]]  # here we put the ranges in a list
blc.compute(*ranges, interpolation='polynomial', order=6)  # the use of * will unpack the elements of the list
subplot6 = blc.corrected.plot()  # the 'corrected' attribute contains the corrected NDDataset.

# %% [markdown]
# ### 3.3 `multivariate` method
#
# The `method` option defines whether the selected baseline regions of the spectra should be taken 'as is'
# this is the default `method='sequential'`), or modeled using a multivariate approach (`method='multivariate'`).
#
# The `'multivariate'` option is useful when the signal‐to‐noise ratio is low and/or when the baseline changes in
# various regions of the spectrum are correlated. It constist in (i) modeling the baseline regions by a principal
# component analysis (PCA), (ii) interpolate the loadings of the first principal components over the whole spectral
# and (iii) modeling the spectra baselines from the product of the PCA scores and the interpolated loadings.
# (for detail: see [Vilmin et al. Analytica Chimica Acta 891 (2015)](dx.doi.org/10.1016/j.aca.2015.06.006).
#
# If this option is selected, the user should also choose `npc`, the number of principal components used to model the
# baseline. In a sense, this parameter has the same role as the `order` parameter, except tha it will affect how well
# the baseline fits the selected regions, but on *both dimensions: wavelength and acquision time*. In particular a
# large value of `npc` will lead to an overfit of baseline variation with time and will lead to the same result as the
# `sequential` method while a too small `value` would miss important pricipal component underlying the baseline change
# over time. Typical optimum values are `npc=2` or `npc=3` (see Exercises below).

# %%
blc.compute(*ranges, interpolation='pchip', method='multivariate', npc=2)  # * unpacks the elements of the list
subplot8 = blc.corrected.plot()  # the 'corrected' attribute contains the corrected dataset

# %% [markdown]
# ### 3.4 Code snippet for 'advanced' baseline correction
# It can be useful in some instances to visualize the selected ranges to correct the baseline 'almost-interactively'
# (a truly interactive widget is under development !). This can be done using the [axvspan()]
# (https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.axvspan.html) method of the standard matplotlib library
# and the following code in which the user can change any of the parameters and look at the changes after re-running
# the cell:

# %%
# user defined parameters ------------------------
NDDataset = X  # uncorrected dataset
ranges = [[5900.0, 5400.0], [4000., 4500.], [2100., 2000.0], [1550., 1555.], [1250.0, 1300.], [800., 850.]]
interpolation = 'pchip'  # choose 'polynomial' or 'pchip'
order = 5  # only used for 'polynomial'
method = 'sequential'  # choose 'sequential' or 'multivariate'
npc = 3  # only used for 'multivariate'
# --------------------------------------------------
# code: compute baseline, plot original and corrected NDDatasets and ranges
blc = scp.BaselineCorrection(NDDataset)
blc.compute(*ranges, interpolation=interpolation, order=order, method=method, npc=npc)
NDDataset.plot()  # plot original NDDataset
plt.gca().get_xaxis().set_visible(False)  # and remove xaxis info
blc.corrected.plot()
for r in ranges:  # loop over the ranges
    plt.axvspan(r[0], r[1], facecolor='lightgreen')  # plot a rectangular area in each range

# %% [markdown]
# ## 3.5. Exercises
#
# basic:
# - write commands to subtract (i) the first spectrum from a dataset and (ii) the mean spectrum from a dataset
# - write a code to correct the baseline of the last 10 spectra of the above dataset in the 4000-3500 cm$^{-1}$ range
#
# intermediate:
# - what would be the parameters to use in 'advanced' baseline correction to mimic 'detrend' ? Write a code to check
# your answer.
#
# advanced:
# - simulate noisy spectra with baseline drifts and compare the performances of `multivariate` vs `sequential` methods

# %% [markdown]
# -- end of this tutorial -- Now, you should be ready to correct the baseline of your own data :-)
