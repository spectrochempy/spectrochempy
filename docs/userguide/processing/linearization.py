# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python 3
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
#     version: 3.9.0
#   widgets:
#     application/vnd.jupyter.widget-state+json:
#       state: {}
#       version_major: 2
#       version_minor: 0
# ---

# %% [markdown]
# # Linearization of coordinates

# %%
import spectrochempy as scp

# %% [markdown]
# To allow some mathematical operations or dataset processing,
# it is often necessary that the datasets have linear
# coordinates.
#
# Here we will show how to do such linearization in various situations.
# Let's take an Omnic IR 2D spectra for sake of
# demonstration

# %%
dataset = scp.NDDataset.read_omnic('irdata/nh4y-activation.spg')
dataset.y -= dataset.y[0]  # remove offset in the time
dataset.y.title = 'Time'
prefs = dataset.preferences
prefs.figure.figsize = (7, 3)
_ = dataset.plot_stack(colormap='viridis', colorbar=True)

# %% [markdown]
# As we will be modify the dataset using different method,
# we will works on copies keeping the original dataset
# unchanged.

# %% [markdown]
# ## Changing the `linear` property

# %%
nd = dataset.copy()

# %% [markdown]
# If we look at the spacing of the x coordinate (wavenumbers),
# it is clear that the spacing is not perfecty linear:

# %%
spacing = nd.x.spacing
spacing

# %% [markdown]
# We get two different but really very close spacings. It might be justified
# to correct this
# assuming there is only one spacing (the mean value of all the spacings).
# In this case the fastest way to linearize
# is to use the `linear` properties of coordinates.
#
# Indeed if we read this property, we find that the coordinate is not linear as shown above (two different but close
# spacings)

# %%
nd.x.linear

# %% [markdown]
# We just need to set it to True and the necessary transformation will be done automatically:

# %%
nd.x.linear = True
spacing = nd.x.spacing
spacing

# %% [markdown]
# Note that the spacing here is not the average of the two `0.965` and `0.964`spacing which have been rounded,
# it is the average of **all** spacings in the coordinate array.

# %% [markdown]
# ## Using the LinearCoord constructor
#
# Another easy way is to transform the `Coord` objects (the `x`coordinate!) into `LinearCoord`. In this case the
# Linear property will be set to True, automatically.

# %%
nd = dataset.copy()  # reset to the original data
nd.x = scp.LinearCoord(nd.x)
spacing = nd.x.spacing
spacing

# %% [markdown]
# ## When changing the `linear` property or using `LinearCoord` constructor do not work?

# %% [markdown]
# Now let's look at the second dimension `y`.

# %%
spacing = nd.y.spacing
spacing

# %% [markdown]
# Now we see that we have three relatively close spacing but one completely different: actually it looks like being
# twice larger, indicating that probable one row might be missing.
# In this case if we use the previous method, we will get a wrong spacing, due to the anormal values `1201`.
# Spectrochempy is aware of this problem and then will not try to change the coordinates:

# %%
nd.y.linear = True
spacing = nd.y.spacing
spacing

# %% [markdown]
# The `linear` properties remains unchanged!

# %%
nd.y.linear

# %% [markdown]
# ## Interpolation

# %% [markdown]
# In the previous case we were not able to linearize the data due to some missing data. In this case,the only way
# would be to interpolate data so that the variation is linear.

# %%
nd = dataset.copy()  # reset to the original data  # nd.interpolate(linear=True)  #TODO
