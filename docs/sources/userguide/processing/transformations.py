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
prefs = dataset.preferences
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
dataset.plot_stack()

# %% [markdown]
# Now the max function return the  maximum in the unmasked region, which is exactly what we wanted.

# %%
dataset.max()

# %% [markdown]
# To clear this mask, we can simply do:

# %%
dataset.remove_masks()
dataset.plot()

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
t_dataset.plot()

# %% [markdown]
# ## Changing units

# %% [markdown]
# Units of the data and coordinates can be changed, but only towards compatible units. For instance, data are in
# absorbance units, which are dimensionless (**a.u**). So a dimensionless units such as **radian** is allowed,
# even if in this case it makes very little sense.

# %%
dataset.units = "radian"

# %%
dataset.plot()

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
dataset.plot()

# %% [markdown]
# See [Units](../objects/dataset/dataset.ipynb#Units) for more details on these units operations
