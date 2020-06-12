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
# # Smoothing
#
# This tutorial shows how to smooth spectra along one dimension (another tutorial will be devoted to 2-D smoothing)
# and gives informations on the algorithms used in Spectrochempy.
#
# We first import spectrochempy, the other libraries used in this tutorial, and a sample dataset (
# nh4y-activation.spg) from which we extract a noisy part:

# %%
import spectrochempy as scp
import numpy as np
import matplotlib.pyplot as plt

X = scp.read_omnic('irdata//nh4y-activation.spg')  # import spectra
X = X[0:5, 3600.0:2800.0]  # select a noisy part (the first 5 spectra in the 3700-2800 cm-1 range)
ax = X.plot()  # plot

# %% Two methods implemented in spectrochempy can be used to smooth spectra along either one dimension ( [markdown]
# i.e. in this example: wavenumbers or acquision time). In this tutorial we will how to use them for smoothing the
# spectra along the wavlength dimension. Both methods are based on window functions, which ptototype is the *moving
# average*.
#
# ## 1. The `smooth()` method The `smooth()` method is adapted from the ["Smoothing of a 1D signal" code](
# https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html) of the [Scipy cookbook](
# https://scipy-cookbook.readthedocs.io/). It is a (weighted)-moving average method and consist in the convolution of
# a window of a given length with the spectrum.
#
# In its simplest form - i.e. unweighted moving average - each absorbance at a given wavenumber of the smoothed
# spectrum is the average of the aborbances at the the absorbance at the considered wavenumber and the N neighboring
# wavenumbers (i.e. N/2 before and N/2 after), hence the conventional use of an odd number of N+1 points to define
# the window length. For the points located at both end of the spectra, the extremities of the spectrum are mirrored
# beyond the initial limits so as to minimize boundary effects.
#
# When pass as is (i.e. `X.smooth()`), the method uses a moving average of 5 points:

# %%
ax = X.smooth().plot()

# %% [markdown]
# ### 2.1. Window length
#
# The following code compares the influence of the window size on the smoothing of the first spectrum of the NDDataset:

# %%
Xs = X[0]  # Xs will store the noisy spectrum (X[0]); smoothed spectra will be concatenated
for i, length in enumerate([5, 11, 27, 51, 101, 201, 501]):  # loop over window lengths. i index will run from 0 to 6.
    s = X[0].smooth(window_length=length)  # smooth
    s += 0.1 * (1 + i)  # shift the absorbance by +0.1 a.u. with respect to previous iteration
    Xs = scp.concatenate(Xs, s, axis='y')  # concatenate to Xs along the `y` (rows) dimension
ax = Xs.plot()

# %% The above spectra clearly show that as that the width of the window increases, the peaks belonging to [markdown]
# the spectrum are flattened out and distorted. When determining the optimum window length, one should thus consier
# the balance between noise removal and signal integrity: the larger the window length, the stronger the smoothing,
# but also the greater the chance to distort the spectrum.
#
# ### 2.2. Window function Besides the window `length` (default=11 points), the user can also choose the type of
# window (`window`) from 'flat', 'hanning' (i.e. Hann window), 'hamming', 'bartlett' or 'blackman'.  The `flat`
# window - which is the default shown above - should be fine for the vast majority of cases.
#
# The code below compares the effect of the type of window:

# %%
Xs = X[0]
for i, window in enumerate(['flat', 'bartlett', 'hanning', 'hamming', 'blackman']):  # loop over 5 window types
    s = X[0].smooth(window_length=27, window=window) + 0.1 * (1 + i)  # smooth and shift
    Xs = scp.concatenate(Xs, s, axis='y')
ax = Xs.plot()

# %% Close examination of the spectra shows that the flat window leads to the stronger smoothing. This is [markdown]
# because the other window functions (also known as *apodization functions*) are used as weighting functions for the
# N+1 points, with the largest weight on the central point and smaller weights for external points.
#
# The window functions as used in Spectrochempy are derived from the numpy library. These builtin functions are such
# that the value of the central point is 1. Hence, as shown below,  they are normalised to the sum of weights. The
# code below displays the corresponding normalized fucntions for 27 points:

# %%
for i, f in enumerate([np.bartlett, np.hanning, np.hamming, np.blackman]):
    winfunc = f(27) / np.sum(27)  # normalized window function
    plt.plot(np.arange(27) - 13, winfunc + i * 0.01)  # shift along x (centered on 0) and y (+0.1 for each function)

# %% As shown above, the "bartlett" function is equivalent to a triangular apodization, while other [markdown]
# fonctions ("hanning", "hamming", "blackman") are bell-shaped. More information on window funcntions can be found [
# here](https://en.wikipedia.org/wiki/Window_function).
#
# Overall, the impact of the window function on the final spectrum is moderate, as can be shown by comparing the
# differences (noisy spetcrum *minus* smoothde spectra:

# %%
Xs_diff = Xs[1:] - Xs[0]
ax = Xs_diff.plot()  # from bottom to top: flat, bartlett/triangular, hann, hamming, blackman:

# %% [markdown]
# and the standard deviations (the largest the value, the stronger the smoothing):

# %%
Xs_diff.std(dim='x').data  # from left to right: flat, bartlett/triangular, hanning, hamming, blackman:

# %% [markdown]
# ## 3. Savitzky-Golay algorithm:`savgol_filter()`
#
# The second algorithm implemented in spectrochempy is the Savitzky-Golay filter which uses a polynomial
# interpolation in the moving window. A demonstrative illustration of the method can be found on the [Savitzky-Golay
# filter](https://en.wikipedia.org/wiki/Savitzky%E2%80%93Golay_filter) entry of Wikipedia.
#
# The function implemented in spectrochempy is a wrapper of the [savgol_filert() method](
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html) from the [scipy.signal](
# https://docs.scipy.org/doc/scipy/reference/signal.html) module to which we refer the interested reader. It not only
# used to smooth spectra but also to compute their successive derivatives. The latter are treated in [the
# peak-finding tutorial](../analysis/peak_finding.ipynb) and we will focus here on the smoothing which is the default
# of the filter (default parameter: `deriv=0`).
#
# As for the `smooth()` method, it is a moving-window based method. Hence, the window length (`window_length`
# parameter) plays an equivalent role, except that it *must* be odd. Moreover, instead of choosing a window function,
# the user can choose the order of the polynomial used to fit the window data points (`polyorder`, default value: 0).
# The latter must be strictly smaller than the window size (so that the polynomial coefficients can be fully
# determined).
#
# The use of this method is illustrated below, we leave to the reader to assess the impact of the window length and
# polynomial order (see Exercises below)

# %%
_ = X.savgol_filter(window_length=5, polyorder=0).plot()

# %% [markdown]
# ## 3. Exercises
#
# intermediate: - what would be the parameters to use in the 'savogol_filter()' methof to mimic 'smooth()' ? Write a
# code to check your answer - examine the impacts of `window_length` and `polyorder` on the extent of smoothing with
# a Svitzky-Golay filter.

# %%
