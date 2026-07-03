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
ds = ds[:, 2200.0:1800.0]

# %% [markdown]
# ## Normalization
#
# `normalize` scales data along a dimension.  The default method is ``'max'``.

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
# `center` subtracts the mean.  `autoscale` mean-centres and divides by the
# standard deviation (z-score).  By default these operate per variable
# (``dim='y'``), which is the convention before PCA or PLS.

# %%
nd = ds.center(dim="y")
_ = nd.plot(title="Mean-centered (per wavenumber)")

# %%
nd = ds.autoscale(dim="y")
_ = nd.plot(title="Autoscaled (z-score per wavenumber)")

# %% [markdown]
# ## Standard Normal Variate (SNV)
#
# SNV is equivalent to autoscaling each spectrum individually (``dim='x'``).
# It is a classic NIR preprocessing step.

# %%
nd = ds.snv()
_ = nd.plot(title="SNV corrected")

# %% [markdown]
# ## Multiplicative Scatter Correction (MSC)
#
# MSC corrects for multiplicative and additive scattering effects by linearly
# regressing each spectrum against a reference (the mean spectrum by default).

# %%
nd = ds.msc()
_ = nd.plot(title="MSC corrected")

# %% [markdown]
# ## Using transformers for machine-learning workflows
#
# For train/test splits or cross-validation, the procedural API above
# recalculates statistics on every call.  To reuse statistics learned from
# a training set, use the transformer classes instead:
#
# ```python
# scaler = scp.AutoscaleTransformer(dim="y")
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled  = scaler.transform(X_test)   # uses train mean/std
# ```
#
# All nine operations have a matching transformer (e.g.
# `CenterTransformer`, `NormalizeTransformer`, `MSCTransformer`, …).
# They implement the familiar `fit()` / `transform()` / `fit_transform()` /
# `inverse_transform()` lifecycle and expose `get_params()` / `set_params()`
# for scikit-learn-compatible cloning.

# %% [markdown]
# All operations support `inplace=True` and can be called as either top-level
# functions (``scp.normalize(...)``) or dataset methods (``dataset.normalize(...)``).
