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
#     version: 3.9.1
#   widgets:
#     application/vnd.jupyter.widget-state+json:
#       state: {}
#       version_major: 2
#       version_minor: 0
# ---

# %% [markdown]
# # Alignment of datasets

# %%
import spectrochempy as scp

# %% [markdown]
# ## Choose how to align coordinates
#
# Use `method="inner"` to keep only shared coordinates or
# `method="outer"` to keep the union. The outer join masks values
# where a dataset has no matching coordinate.
#
# Use `method="interpolate"` when the datasets sample the same physical
# axis at different positions. This evaluates the other datasets on the
# first dataset's grid. Interpolation uses the supplied coordinates.
# Correct any calibration shift before aligning the grids.

# %% [markdown]
# ### Resample onto a grid you choose
#
# Pass `coord=target` to align several datasets on a custom grid along
# one dimension. The example uses two small spectra with overlapping
# ranges. Read the output through `.masked_data` to see which coordinates
# have a valid intensity in each spectrum.
#
# %%
first = scp.NDDataset(
    [0.2, 0.9, 0.3],
    coordset=[scp.Coord([1180.0, 1200.0, 1220.0], units="cm^-1")],
)
second = scp.NDDataset(
    [0.6, 0.7, 0.1],
    coordset=[scp.Coord([1190.0, 1210.0, 1230.0], units="cm^-1")],
)
target = scp.Coord([1180.0, 1200.0, 1220.0, 1230.0], units="cm^-1")
left, right = scp.align(first, second, dim="x", method="interpolate", coord=target)
print(left.x.data)
print(left.masked_data)
print(right.masked_data)

# %% [markdown]
# The first target lies outside the second spectrum, and the last target
# lies outside the first. Both are masked in the corresponding result.
# A target that coincides with an original sample keeps its intensity and
# mask. Between samples, the result is masked if the interpolator needs
# a masked measurement. Subtracting the aligned spectra uses their common
# valid range.
#
# Use `interpolate_method="pchip"` to select shape-preserving cubic
# interpolation when aligning several datasets.

# %% [markdown]
# ### Interpolate a single spectrum
#
# Call `dataset.interpolate()` to evaluate one spectrum on a new grid.
# Linear interpolation connects adjacent samples with straight lines.
# PCHIP uses a smooth curve that preserves monotonic intervals. A denser
# grid gives more estimates between measurements at the original spectral
# resolution.
#
# Here the target includes a point between measurements, a measured point,
# and a point beyond the end of the spectrum. Inspect the estimated
# intensities for both methods.
#
# %%
target = scp.Coord([1190.0, 1200.0, 1230.0], units="cm^-1")
linear = first.interpolate(dim="x", coord=target)
pchip = first.interpolate(dim="x", coord=target, method="pchip")
print("Wavenumber:", target.data)
print("Linear:   ", linear.data)
print("PCHIP:    ", pchip.data)

# %% [markdown]
# Both methods keep the intensity at 1200 cm⁻¹ and return NaN at
# 1230 cm⁻¹ with the default fill value. To use a constant outside the
# measured range, pass it as `fill_value`. Pass `fill_value="extrapolate"`
# only when extending the fitted curve beyond the measurements is useful
# for your analysis.

# %% [markdown]
# ## Example

# %% [markdown]
# To allow some mathematical operations or dataset processing, it is often necessary that the datasets are aligned,
# *i.e.,* that they have compatible coordinate for the dimensions.
#
# For sake of demonstration, let's take an experimental dataset that will be arbitrary split into four unaligned
# datasets. Then will realign them and apply some binary mathematical operation such as addition or subtraction that
# required aligned coordinates.

# %%
dataset = scp.read_omnic("irdata/nh4y-activation.spg")
dataset.y = dataset.y - dataset.y[0]  # remove offset in the time
dataset.y.title = "time"
prefs = scp.preferences
prefs.reset()
prefs.figure.figsize = (7, 3)
prefs.figure.dpi = 100
_ = dataset.plot(method="contour", colormap="viridis", colorbar=True)
print("shape:", dataset.shape)

# %%
nd1 = dataset[0:30, 0:4000]
nd2 = dataset[0:30, 2000:5549]
nd3 = dataset[10:55, 0:4000]
nd4 = dataset[10:55, 2000:5549]
_ = scp.multiplot_contour(
    datasets=[nd1, nd2, nd3, nd4],
    colormap="viridis",
    nrow=2,
    ncol=2,
    sharex=True,
    sharey=True,
    dpi=100,
)

# %% [markdown]
# The four datasets `nd1` to `nd4` have some overlapping in both dimensions. But if we want for example to add `nd2`
# with `nd4` . This will fail because the dimension are not aligned.

# %%
try:
    nd2 + nd4
except Exception as e:
    scp.error_(Exception, str(e) + " Cannot add unaligned datasets.")

# %% [markdown]
# Let try to align them, in the `y` dimension (*i.e.* the first) as this the one which differ in size.
# (NOTE: to find the actual names of the dimensions, just get the `dims` attribute of the datasets).

# %%
nd2.dims, nd4.dims

# %% [markdown]
# To align we can use different methods, depending on the expected results (missing values in the aligned datasets
# will be masked)

# %%
# `outer` method => union of the coordinates
nd2a, nd4a = scp.align(nd2, nd4, dim="y", method="outer")

# %% [markdown]
# Now we can perform an addition without any problem

# %%
ndadd = nd2a + nd4a
ndadd.shape

# %% [markdown]
# Let's plot both individual aligned arrays, and their sum. Note, that only the common region appears in the result
# array, as the mathematical operation are aware of the masks.

# %%
_ = scp.multiplot_contour(
    datasets=[nd2a, nd4a, ndadd],
    colormap="viridis",
    sharey=True,
    nrow=1,
    ncol=3,
    figsize=(8, 3),
    dpi=100,
)

# %% [markdown]
# Now, assume we want to align in the other dimension, or both

# %%
try:
    nd1 + nd2
except Exception as e:
    scp.error_(Exception, str(e) + " Cannot add unaligned datasets.")

# %%
nd1a, nd2a = scp.align(nd1, nd2, dim="x", method="outer")
ndadd = nd1a + nd2a
_ = scp.multiplot_contour(
    datasets=[nd1a, nd2a, ndadd],
    colormap="viridis",
    sharey=True,
    nrow=1,
    ncol=3,
    figsize=(8, 3),
    dpi=100,
)
ndadd.shape

# %% [markdown]
# ## Methods for alignments
# Method for alignments are:
#
# * **outer** which means that a union of the different coordinates is achieved (missing values are masked)
# * **inner** which means that the intersection of the coordinates is used
# * **first** which means that the first dataset is used as reference
# * **last** which means that the last dataset is used as reference
# * **interpolate** uses the first dataset's grid by default and masks
#   positions outside each source range and positions that depend on masked
#   samples. For one dimension, pass
#   `scp.align(first, second, dim="x", method="interpolate", coord=target)`
#   to choose a different grid. `interpolate_sampling="auto"` keeps the
#   chosen grid unchanged.

# %% [markdown]
# ### `inner` method
#
# An inner join keeps the intersection of the two coordinate grids.
# The alignment carries over the intensities and their masks at the shared
# positions. For two groups of acquisitions with different spectral windows,
# align along `x` before comparing their mean spectra. Each group keeps its
# own acquisition axis `y`.
#
# Two windows can overlap physically and still have no shared grid points.
# In that case, `inner` raises a `ValueError`. Use interpolation when
# you need to estimate values on a common grid. An inner join also raises
# a `ValueError` if rounding would make two coordinates within one input
# match the same position.

# %%
# `inner` method => intersection of the coordinates
nd2a, nd4a = scp.align(nd2, nd4, dim="y", method="inner")
ndadd = nd2a + nd4a
ndadd.shape  # note the difference with the outer method above (the shape correspond to the intersection)

# %%
_ = scp.multiplot_contour(
    datasets=[nd2a, nd4a, ndadd],
    colormap="viridis",
    sharey=True,
    nrow=1,
    ncol=3,
    figsize=(8, 3),
    dpi=100,
)

# %% [markdown]
# ### `first` method
#
# Use `first` to keep the first dataset's grid. In the other dataset,
# values are masked where there is no matching coordinate. Coordinates
# outside the reference grid are dropped.

# %%
# `first` method => align on the first dataset
nd2a, nd4a = scp.align(nd2, nd4, dim="y", method="first")
ndadd = nd2a + nd4a
ndadd.shape  # note the difference with the outer method above

# %%
_ = scp.multiplot_contour(
    datasets=[nd2a, nd4a, ndadd],
    colormap="viridis",
    sharey=True,
    nrow=1,
    ncol=3,
    figsize=(8, 3),
    dpi=100,
)

# %% [markdown]
# ### `last` method
#
# Use `last` to keep the last dataset's grid. In the other dataset,
# values are masked where there is no matching coordinate. Coordinates
# outside the reference grid are dropped.

# %%
# `last` method => align on the last dataset
nd2a, nd4a = scp.align(nd2, nd4, dim="y", method="last")
ndadd = nd2a + nd4a
ndadd.shape  # note the difference with the outer method above

# %%
_ = scp.multiplot_contour(
    datasets=[nd2a, nd4a, ndadd],
    colormap="viridis",
    sharey=True,
    nrow=1,
    ncol=3,
    figsize=(8, 3),
    dpi=100,
)

# %% [markdown]
# ## Alignment along several dimensions

# %%
nd1a, nd4a = scp.align(nd1, nd4, dims=["y", "x"])  # by default the outer method is used
ndadd = nd1a + nd4a
# Comparison of the result array with the original (only the common region is visible, due to the masks)
_ = scp.multiplot_contour(
    datasets=[nd1a, nd4a, ndadd],
    colormap="viridis",
    sharex=0,
    sharey=True,
    nrow=1,
    ncol=3,
    figsize=(8, 3),
    dpi=100,
)
nd1a.shape, nd4a.shape, ndadd.shape

# %%
nd1a, nd4a = scp.align(nd1, nd4, dims=["y", "x"], method="inner")
ndadd = nd1a + nd4a
# Comparison of the result array with the original (only the common region is visible, due to the masks)
_ = scp.multiplot_contour(
    datasets=[nd1a, nd4a, ndadd],
    colormap="viridis",
    sharex=0,
    sharey=True,
    nrow=1,
    ncol=3,
    figsize=(8, 3),
    dpi=100,
)

# %%
nd1a, nd4a = scp.align(nd1, nd4, dims=["y", "x"], method="first")
ndadd = nd1a + nd4a
# Comparison of the result array with the original (only the common region is visible, due to the masks)
_ = scp.multiplot_contour(
    datasets=[nd1a, nd4a, ndadd],
    colormap="viridis",
    sharex=0,
    sharey=True,
    nrow=1,
    ncol=3,
    figsize=(8, 3),
    dpi=100,
)

# %%
nd1a, nd4a = scp.align(nd1, nd4, dims=["y", "x"], method="last")
ndadd = nd1a + nd4a
# Comparison of the result array with the original (only the common region is visible, due to the masks)
_ = scp.multiplot_contour(
    datasets=[nd1a, nd4a, ndadd],
    colormap="viridis",
    sharex=0,
    sharey=True,
    nrow=1,
    ncol=3,
    figsize=(8, 3),
    dpi=100,
)
