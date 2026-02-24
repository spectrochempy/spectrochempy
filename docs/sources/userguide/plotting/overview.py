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
# # Plotting Overview
#
# SpectroChemPy plotting is designed to be:
#
# - **Automatic** – sensible defaults are chosen for you.
# - **Consistent** – similar data produces similar visuals.
# - **Customizable** – override anything locally or globally.
# - **Style-driven** – visual appearance can be changed without rewriting code.
#
# In most cases, plotting your data requires only one line.

# %% [markdown]
# ## Your First Plot
#
# Once a dataset is loaded, simply call the `plot()` method:

# %%
import spectrochempy as scp

ds = scp.read("irdata/nh4y-activation.spg")
_ = ds.plot()

# %% [markdown]
# That’s it.
#
# Here SpectroChemPy automatically:
#
# - Detects the dataset is 2D and chooses an appropriate plot type (stacked lines in this case).
# - Selects a suitable colormap (sequential because the y-dimension is a time axis)
# - Adds axis labels from dataset metadata.
# - Adjusts layout and scaling.
#
# In most situations, no additional configuration is needed.

# %% [markdown]
# ## Automatic Plot Type Selection
#
# The `plot()` method adapts to the dimensionality of your dataset:
#
# | Dataset type | Default plot |
# |--------------|--------------|
# | 1D / 2D      | Line(s) plot |
# | 2D field     | Image / contour (depending on method) |
#
# But you can also explicitly choose another plotting method:

# %%
_ = ds.plot_contour()


# %%
_ = ds.plot_image()

# %% [markdown]
# ## Automatic Color Selection
#
# SpectroChemPy chooses colors intelligently:
#
# - for lines:
#   - Sequential data → sequential colormap for lines
#   - Categorical stacks → distinct categorical colors
# - for contours and images:
#   - Data with positive and negative values → diverging colormap
#   - Data with only positive or only negative values → sequential colormap
#
# This behavior works automatically — no configuration required.

# %% [markdown]
# For line plots, however, you can disable continuous colormaps and use categorical colors:

# %%
_ = ds.plot_lines(cmap=None)

# %% [markdown]
# ## Colorbars
#
# By default SpectroChemPy never print a colorbar.  But using the option `colorbar=auto` will make it print  whenever
# a sequential or diverging colormap is used/

# %%
_ = ds.plot(colorbar="auto")  # shows colorbar whenever applicable

# %% [markdown]
# ## Changing the Colormap
#
# You can change colors per plot:

# %%
_ = ds.plot(cmap="plasma")

# %% [markdown]
# ## Colormap Precedence
#
# When choosing a colormap, SpectroChemPy follows this priority order:
#
# 1. **Explicit `cmap` kwarg** (i.e. `cmap=XXX` passed in plot()— highest priority
# 2. **Preferences colormap** — if set and not "auto" (see preferences documentation)
# 3. **Matplotlib style** — if the style sets `image.cmap` and prefs.colormap is "auto" (see styles documentation)
# 4. **Default preferences** — `prefs.colormap_sequential` or `prefs.colormap_diverging` (see preferences documentation)
#
# This means:
# - `ds.plot(cmap="inferno")` always uses inferno
# - `prefs.colormap = "cividis"` uses cividis (unless overridden by explicit cmap)
# - `ds.plot(style="grayscale")` with `prefs.colormap="auto"` uses grayscale
# - Default behavior uses preferences-defined sequential/diverging colormaps

# %% [markdown]
# ## Changing the Overall Style
#
# You can change the visual appearance using styles:

# %%
_ = ds.plot(style="grayscale")

# %% [markdown]
# Styles affect fonts, grid appearance, backgrounds, and (in auto mode) colormap defaults.

# %% [markdown]
# ## The Mental Model
#
# In practice:
#
# - `ds.plot()` just works.
# - `cmap=` changes colors.
# - `colorbar=` controls the colorbar.
# - `style=` changes the overall appearance.
# - `scp.preferences` changes defaults persistently.
#
# Everything else is optional.
#
# In the following sections, we will explore how to customize plots in more detail.
