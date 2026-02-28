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
# ---

# %% [markdown]
# # Plot Types
#
# SpectroChemPy provides several plotting methods. `ds.plot()` chooses
# automatically, but explicit methods give you control.

# %% [markdown]
# ## Line Plot
#
# For spectra and time-series:

# %%
import spectrochempy as scp

ds = scp.read("irdata/nh4y-activation.spg")
ds.y -= ds.y[0]  # Set y coordinates as relative time  for better visualization
ds.y.ito('hour')
ds.y.title = "Time on stream"  # Update y-axis title accordingly
ds[:, 1290.0:920.0] = scp.MASKED  # We also mask a region that we do not want to display
ds1 = ds[0]  # Single spectrum

# %%
_ = ds1.plot()

# %% [markdown]
# Or using `plot_lines()` explicitly (canonical form):

# %%
_ = ds.plot_lines()

# %% [markdown]
# ## Image Plot
#
# For 2D data where both axes are numerical:

# %%
_ = ds.plot_image()

# %% [markdown]
# Image plots automatically include a colorbar:

# %%
_ = ds.plot_image(colorbar=True)

# %% [markdown]
# Hide the colorbar if needed:

# %%
_ = ds.plot_image(colorbar=False)

# %% [markdown]
# ## Contour Plot
#
# For continuous data with smooth transitions:

# %%
_ = ds.plot_contour()

# %% [markdown]
# Contour plots also support colorbars:

# %%
_ = ds.plot_contour(colorbar=True)

# %% [markdown]
# ## Decision Guide
#
# | Method | Use When |
# |--------|----------|
# | `plot()` / `plot_lines()` | Showing spectra, time series, or stacked traces |
# | `plot_image()` | 2D field with spatial x/y axes |
# | `plot_contour()` | Smooth visualization of continuous 2D data |
# | `plot_surface()` | 3D perspective view of 2D data |
# | `plot_waterfall()` | 3D-style waterfall representation |

# %% [markdown]
# ## Surface Plot
#
# For a 3D perspective view of 2D data:

# %%
_ = ds.plot_surface(y_reverse=True, linewidth=0)

# %% [markdown]
# ## Waterfall Plot
#
# For a waterfall-style representation:

# %%
ds = scp.read("irdata/nh4y-activation.spg")
ds.y -= ds.y[0]  # Set y coordinates as relative time  for better visualization
ds.y.ito('hour')
ds.y.title = "Time on stream"  # Update y-axis title accordingly
ds[:, 1290.0:920.0] = scp.MASKED  # We also mask a region that we do not want to display
_ = ds.plot_waterfall(y_reverse=True)

# %% [markdown]
# ## Combining with Options
#
# All plot methods accept the same customization options:

# %%
_ = ds.plot_image(
    cmap="plasma",
    xlim=(2000, 1300),
    ylim=(1, 5),
)

# %% [markdown]
# The appropriate method is chosen automatically when you call `ds.plot()`,
# but explicit methods make your intent clear and provide specific functionality.

# %% [markdown]
# ## Deprecated Method Names
#
# The following method names are deprecated but still work:
#
# | Deprecated | Current (Canonical) |
# |------------|---------------------|
# | `plot_stack()` | `plot_lines()` |
# | `plot_map()` | `plot_contour()` |
# | `plot(method="image")` | `plot_image()` or `plot_contourf()` |
#
# Using the canonical names is recommended for new code.
