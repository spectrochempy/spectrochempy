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
#     version: 3.13.2
# ---

# %% [markdown]
# # Basic transformations
#
# Let's show some SpectroChemPy features on a group of IR spectra

# %%
import spectrochempy as scp
from spectrochempy import MASKED
from spectrochempy import DimensionalityError
from spectrochempy import error_

# %%
dataset = scp.read_omnic("irdata/nh4y-activation.spg")
dataset.y -= dataset.y[0]
dataset.y.title = "time"
dataset

# %%
prefs = scp.preferences
prefs.figure.figsize = (6, 3)
prefs.colormap = "Dark2"
prefs.colorbar = True
ax = dataset.plot()

# %% [markdown]
# ## Masking data

# %% [markdown]
# if we try to get for example the maximum of this dataset, we face a problem due to the saturation around
# 1100 cm$^{-1}$.

# %%
dataset.max()

# %% [markdown]
# One way is to apply the max function to only a part of the spectrum (using [slicing](slicing.rst)). Another way is to
# mask the undesired data.
#
# Masking values in this case is straightforward. Just set a value `masked` or True for those data you want to mask.

# %%
dataset[:, 1290.0:890.0] = MASKED
# note that we specify floating values in order to sect wavenumbers, not index.

# %% [markdown]
# Here is a display the figure with the new mask

# %%
_ = dataset.plot_stack()

# %% [markdown]
# Now the max function return the  maximum in the unmasked region, which is exactly what we wanted.

# %%
dataset.max()

# %% [markdown]
# To clear this mask, we can simply do:

# %%
dataset.remove_masks()
_ = dataset.plot()

# %% [markdown]
# ## Transposition

# %% [markdown]
# Dataset can be transposed

# %%
dataset[:, 1290.0:890.0] = MASKED  # we mask the unwanted columns
t_dataset = dataset.T
t_dataset

# %% [markdown]
# As it can be observed the dimension `x`and `y`have been exchanged, *e.g.* the original shape was **(x: 5549, y: 55)**,
# and after transposition it is **(y:55, x:5549)**.
# (the dimension names stay the same, but the index of the corresponding axis are exchanged).

# %% [markdown]
# Let's visualize the result:

# %%
_ = t_dataset.plot()

# %% [markdown]
# ## Changing units

# %% [markdown]
# Units of the data and coordinates can be changed, but only towards compatible units. For instance, data are in
# absorbance units, which are dimensionless (**a.u**). So a dimensionless units such as **radian** is allowed,
# even if in this case it makes very little sense.

# %%
dataset.units = "radian"

# %%
_ = dataset.plot()

# %% [markdown]
# Trying to change it in 'meter' for instance, will generate an error!

# %%
try:
    dataset.to("meter")
except DimensionalityError as e:
    error_(DimensionalityError, e)

# %% [markdown]
# If this is for some reasons something you want to do, you must for the change:

# %%
d = dataset.to("meter", force=True)
d.units

# %% [markdown]
# When units are compatible there is no problem to modify it. For instance, we can
# change the `y` dimension units (
# Time) to hours. Her we use the inplace transformation `ito` .

# %%
dataset.y.ito("hours")
_ = dataset.plot()

# %% [markdown]
# See [Units](../objects/dataset/dataset.ipynb#Units) for more details on these units operations

# %% [markdown]
# # Chemometric preprocessing
#
# SpectroChemPy provides standard preprocessing operations commonly used in
# chemometrics and spectroscopic data analysis.  They operate along a chosen
# dimension and respect masks, units, coordinates, and metadata.

# %%
# Load a dataset and focus on a small region for clarity
ds = scp.read_omnic("irdata/nh4y-activation.spg")
ds = ds[:, 4000.0:2000.0]

# %% [markdown]
# ## Normalization
#
# With `dim="x"`, `normalize` scales each spectrum independently along
# its spectral axis. The default `method="max"` divides by the largest
# absolute value in the selected region. Choose the region before
# normalizing, since a saturated band or a spike can dominate the scale.
# Compare relative band intensities after normalization and use the
# original spectra when absolute absorbance or changes in overall
# intensity are part of the analysis.

# %%
nd = ds.normalize(method="max", dim="x")
_ = nd.plot(title="Max-normalized spectra")

# %% [markdown]
# Other methods include ``'sum'``, ``'vector'`` (L2 norm), and ``'minmax'``.

# %%
nd = ds.normalize(method="minmax", dim="x")
_ = nd.plot(title="Min-max scaled to [0, 1]")

# %% [markdown]
# ## Mean-centering and autoscaling
#
# For an IR series with acquisitions along `y` and wavenumbers along
# `x`, `center(dim="y")` subtracts the mean spectrum of the series.
# Positive and negative residuals show where each acquisition lies
# above or below the series mean. This is the usual axis for centering
# spectral variables before PCA or PLS.
#
# With `center(dim="x")`, each acquisition is shifted by its own mean
# absorbance in the selected window. The shift is constant across that
# spectrum, so the differences between its bands are preserved.
#
# %%
window = ds[:, 4000.0:2000.0]
by_wavenumber = window.center(dim="y")
by_spectrum = window.center(dim="x")
_ = scp.multiplot(
    [by_wavenumber[::10], by_spectrum[::10]],
    method="lines",
    ncol=2,
    labels=["Subtract the mean spectrum", "Subtract each spectrum's mean"],
    x_reverse=True,
    figsize=(10, 4),
)

# %% [markdown]
# Both panels use the same subset of acquisitions for display. The means
# are computed before that subset is selected. Centering keeps the
# absorbance scale. Autoscaling also divides the centered values by their
# standard deviation along the chosen axis. Scaling along `y` gives
# low-variance wavenumbers more weight relative to high-variance ones,
# which can also amplify the influence of noisy regions.
#
# %%
nd = window.autoscale(dim="y")
_ = nd[::10].plot(title="Autoscaled per wavenumber")

# %% [markdown]
# ## Standard Normal Variate (SNV)
#
# SNV subtracts each spectrum's mean and divides by its standard deviation
# across the spectral axis. In SpectroChemPy, `spectra.snv()` gives the same
# values as `spectra.autoscale(dim="x")`. SNV removes a constant offset and
# a positive scale factor applied to a whole spectrum. This is useful for
# comparing band shapes when those differences arise from scattering.
#
# Select the spectral region first and mask saturated or invalid samples.
# The mean and standard deviation use only the unmasked values in that
# region. For a nonconstant spectrum, the result has mean zero and
# population standard deviation one. A flat spectrum gives zeros at
# its valid positions.
#
# %%
standardized = ds.snv()
print("Means:", standardized.mean(dim="x").data[:3].round(6))
print("Standard deviations:", standardized.std(dim="x").data[:3].round(6))

# %% [markdown]
# Compare the same spectrum standardized over the full working window and
# over a narrower window. Both curves cover the same displayed wavenumbers,
# but their means and standard deviations were computed from different
# regions.
#
# %%
spectrum = ds[20:21, :]
narrow = spectrum[:, 3800.0:3500.0]
full_window_snv = spectrum.snv()[:, 3800.0:3500.0]
narrow_window_snv = narrow.snv()
for scaled in (full_window_snv, narrow_window_snv):
    scaled.title = "standardized intensity"
    scaled.units = None

_ = scp.plot_multiple(
    [full_window_snv.squeeze(), narrow_window_snv.squeeze()],
    method="pen",
    labels=["SNV over 4000–2000 cm⁻¹", "SNV over 3800–3500 cm⁻¹"],
    x_reverse=True,
    legend="best",
)

# %% [markdown]
# The vertical scale expresses deviations in standard deviations.
# Keep the original absorbance for analyses that rely on absolute band
# intensities. Within each spectrum SNV applies the same offset and scale
# at every wavenumber, so a sloped baseline or a change in band shape can
# persist.

# %% [markdown]
# ## Multiplicative Scatter Correction (MSC)
#
# MSC fits each measured spectrum as an offset plus a slope times a
# reference spectrum. It subtracts the fitted offset and divides by the
# fitted slope. Differences in band shape can remain after this correction.
#
# The default reference is the mean spectrum of the input series. Pass
# `reference=` to use a representative spectrum on the same spectral grid.
# Check that bands used for interpretation retain their relative intensities.
# A reference from another sample population can change the fitted correction.
#
# In transformer workflows, `MSCTransformer.fit()` learns only the reference
# spectrum.  The intercept and slope remain local to each transformed spectrum,
# so `inverse_transform()` is intentionally unsupported.

# %%
nd = ds.msc()
_ = nd.plot(title="MSC corrected")

# %% [markdown]
# ## Using transformers for machine-learning workflows
#
# Split the spectra before fitting any transformer that learns statistics
# across acquisitions. With `dim="y"`, autoscaling learns one mean and
# standard deviation per wavenumber from the training spectra. Apply those
# same statistics to the held-out spectra. In cross-validation, fit a fresh
# transformer on the training portion of each fold.
#
# This split shows how to reuse statistics in acquisition order.

# %%
X_train, X_test = ds[:40], ds[40:]
scaler = scp.AutoscaleTransformer(dim="y")
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %% [markdown]
# `scaler.mean_` and `scaler.std_` hold the training statistics.
# `scaler.inverse_transform(X_test_scaled)` restores the original scale
# using the same fitted statistics. A scikit-learn clone copies the
# configuration and must be fitted before use.

# %% [markdown]
# Sample-local operations such as `SNVTransformer`,
# `NormalizeTransformer(dim="x")`, and the per-spectrum regression step in
# `MSCTransformer` still compute each observation's statistics during
# `transform()`, because those statistics belong to the spectrum being
# transformed rather than to the training set.  Their inverse transform is not
# supported unless a future API carries the required per-observation state on
# the transformed result.
#
# All nine operations have a matching transformer (e.g.
# `CenterTransformer`, `NormalizeTransformer`, `MSCTransformer`, …).
# They implement the familiar `fit()` / `transform()` / `fit_transform()` /
# `inverse_transform()` lifecycle where meaningful and expose `get_params()` /
# `set_params()` for scikit-learn-compatible cloning.

# %% [markdown]
# All operations support `inplace=True` and can be called as either top-level
# functions (``scp.normalize(...)``) or dataset methods (``dataset.normalize(...)``).
