# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: title,-all
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
# # Filtering and Smoothing
#
# In this tutorial, we show how to filter/smooth spectra
# and we gives information on the algorithms used in Spectrochempy.
#
# We first import spectrochempy, the other libraries used in this tutorial, and a sample raman dataset:

# %%
import spectrochempy as scp

# %%
# import spectra
X = scp.read("ramandata/SMC1-Initial_RT.txt")

# plot it
_ = X.plot()

# %% [markdown]
# ## The `smooth()` method

# %% [markdown]
# Two methods implemented in spectrochempy can be used to smooth spectra along either one dimension (
# The `smooth()` method is adapted from the ["Smoothing of a 1D signal" code](
# https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html) of the [Scipy cookbook](
# https://scipy-cookbook.readthedocs.io/). It is a (weighted)-moving average method and consist in the convolution of
# a window of a given length with the spectrum.
#
# In its simplest form - *i.e.* unweighted moving average - each absorbance at a given wavenumber of the smoothed
# spectrum is the average of the absorbance at the absorbance at the considered wavenumber and the N neighboring
# wavenumbers (*i.e.* N/2 before and N/2 after), hence the conventional use of an odd number of N+1 points to define
# the window length. For the points located at both end of the spectra, the extremities of the spectrum are mirrored
# beyond the initial limits to minimize boundary effects.
#
# When passed as is, i.e. `X.smooth()` , the method uses a moving average of 5 points:

# %%
Xsm = X.smooth()

# %% [markdown]
# Note that it is also possible to use the API function `scp.smooth(X)` instead of the dataset method `X.smooth()`. The result
# is the same.

# %% [markdown]
# Now we plot the original and smoothed spectra on the same figure, and select a small region to better evaluate
# the quality of the smoothing.

# %%

ax = X.plot(color="b", label="original")
ax = Xsm.plot(
    clear=False, color="r", ls="-", label="smoothed ma(5)"
)  # moving average 5 point
_ = ax.legend(loc="best", fontsize=10)

# %% [markdown]
# ### Window length
#
# The following code compares the influence of the window size on the smoothing of the
# first spectrum of the NDDataset `X[0]` .

# %% [markdown]
# Loop over window lengths.
# `i` index will run from 0 to 6. Difference between original and smothed spectra are also displayed, and the standard deviation beteen both spectra is indicated in the legend.

# %%

for i, length in enumerate([3, 5, 11]):
    Xsm = X.smooth(window_length=length)  # smooth

    ax = X.plot(color="b", label="original", figsize=(8, 4))
    ax = Xsm.plot(clear=False, ls="-", lw=2, label=f"smoothed ma({length})")
    diff = X - Xsm
    s = round(diff.std(dim="x").values, 2)
    ax = (diff + 1000.0).plot(clear=False, ls="-", lw=1, label=f"diff ($\sigma$={s})")

    _ = ax.legend(loc="best", fontsize=10)

    # for clarity with zoom on a restricted region
    _ = ax.set_xlim(60, 350)
    _ = ax.set_ylim(500, 3500)


# %%
# The above spectra clearly show that as that the width of the window increases, the peaks belonging to [markdown]
# For large value of the window length, the spectrum is flattened out and distorted. When determining the optimum window length, one should thus consider
# the balance between noise removal and signal integrity: the larger the window length, the stronger the smoothing,
# but also the greater the chance to distort the spectrum.
#
# ### Window function
#
# Besides the window `length` (default=11 points), the user can also choose the type of
# window (`window` ) from `flat` , `hanning` (i.e. Hann window), `hamming` , `bartlett` or
# `blackman` .  The `flat`
# window - which is the default shown above - should be fine for the vast majority of
# cases.
#
# The code below compares the effect of the type of window:

# %%
for i, window in enumerate(["flat", "bartlett", "hanning", "hamming", "blackman"]):
    Xsm = X.smooth(window_length=7, window=window)

    ax = X.plot(color="b", label="original", figsize=(8, 4))
    ax = Xsm.plot(clear=False, ls="-", lw=2, label=f"{window} window")
    diff = X - Xsm
    s = round(diff.std(dim="x").values, 2)
    ax = (diff + 1000.0).plot(clear=False, ls="-", lw=1, label=f"diff ($\sigma$={s})")
    _ = ax.legend(loc="upper left", fontsize=10)
    _ = ax.set_xlim(60, 250)
    _ = ax.set_ylim(500, 3500)

# %%
# Close examination of the spectra shows that the flat window leads to the stronger smoothing. This is [markdown]
# because the other window functions (also known as *apodization functions*) are used as weighting functions for the
# N+1 points, with the largest weight on the central point and smaller weights for external points.
#
# The window functions as used in SpectroChemPy are derived from the numpy library. These builtin functions are such
# that the value of the central point is 1. Hence, as shown below,  they are normalized to the sum of weights. The
# code below displays the corresponding normalized functions for 27 points:

# %%
import numpy as np

functions = []
labels = []
for i, f in enumerate([np.bartlett, np.hanning, np.hamming, np.blackman]):
    coord = scp.NDDataset.linspace(-13, 13, 27)
    s = scp.NDDataset(
        f(27) / np.sum(27) + i * 0.01, coordset=[coord]
    )  # normalized window function, y shifted : +0.1 for each function
    functions.append(s)
    labels.append(f"function: {f.__name__}")

ax = scp.plot_multiple(
    figsize=(7, 5),
    ylim=(0, 0.1),
    method="pen",
    datasets=functions,
    labels=labels,
    ls="-",
    lw=2,
)
_ = ax.legend(labels, loc="upper left", fontsize=10)

# %%  [markdown]
# As shown above, the "bartlett" function is equivalent to a triangular apodization,
# while other
# functions (`hanning` , `hamming` , `blackman` ) are bell-shaped. More information on
# window functions can be found [
# here](https://en.wikipedia.org/wiki/Window_function).

# %% [markdown]
# ## Savitzky-Golay algorithm:`sgs()`
#
# The second algorithm implemented in spectrochempy is the Savitzky-Golay filter which uses a polynomial
# interpolation in the moving window. A demonstrative illustration of the method can be found on the [Savitzky-Golay
# filter](https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter) entry of Wikipedia.
#
# The function implemented in spectrochempy is a wrapper of the [savgol_filter() method](
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html) from the [scipy.signal](
# https://docs.scipy.org/doc/scipy/reference/signal.html) module to which we refer the interested reader. It not only
# used to smooth spectra but also to compute their successive derivatives. The latter are treated in [the
# peak-finding tutorial](../analysis/peak_finding.ipynb) and we will focus here on the smoothing which is the default
# of the filter (default parameter: `deriv=0` ).
#
# As for the `smooth()` method, it is a moving-window based method. Hence, the window length (`window_length`
# parameter) plays an equivalent role, except that it *must* be odd. Moreover, instead of choosing a window function,
# the user can choose the order of the polynomial used to fit the window data points (`polyorder` , default value: 0).
# The latter must be strictly smaller than the window size (so that the polynomial coefficients can be fully
# determined).
#
# The use of this method is illustrated below, we leave to the reader to assess the impact of the window length and
# polynomial order (see Exercises below)

# %%
_ = X.savgol_filter(window_length=5, polyorder=0, mode="mirror").plot()

# %% [markdown]
# ## Whittaker-eilers algorithm : `whittaker_smooth`
#

# %% [markdown]
# <div class='alert alert-info'>
# <b>Exercises</b>
#
# <em>intermediate</em>: - what would be the parameters to use in the `savogol_filter()` method to mimic `smooth()`?
# Write a
# code to check your answer - examine the impacts of `window_length` and `polyorder` on the extent of smoothing with
# a Svitzky-Golay filter.
# </div>
