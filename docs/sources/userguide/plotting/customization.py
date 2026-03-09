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
# # Customizing a Plot
#
# The default plot is designed to look good with minimal effort.
# However, every visual aspect can be adjusted using keyword arguments
# passed to `ds.plot()`. This page covers the most common customizations.

# %% [markdown]
# ## Line Appearance
#
# For 1D and 2D datasets, you can adjust how lines appear:

# %%
import spectrochempy as scp

ds = scp.read("irdata/nh4y-activation.spg")

# %% [markdown]
# ### Color

# %%
_ = ds.plot(color="red")

# %% [markdown]
# ### Line Width

# %%
_ = ds.plot(linewidth=2)

# %% [markdown]
# ### Line Style

# %%
_ = ds.plot(linestyle="--")

# %% [markdown]
# ### Marker

# %%
_ = ds.plot(marker="o")


# %% [markdown]
# ## Axis Limits
#
# Restrict the displayed range:
# %%
_ = ds.plot(
    xlim=(2000, 1000)
)  # NB: slicing is also possible here, ds[:,2000.:1000.].plot() would give the same result

# %%
_ = ds.plot(ylim=(0, 0.5))

# %% [markdown]
# ## Figure Size
#
# Control the figure dimensions (width, height in inches):

# %%
_ = ds.plot(figsize=(10, 4))

# %% [markdown]
# ## Titles and Labels
#
# `ds.plot()` returns a Matplotlib Axes object. Use it to set titles and labels:

# %%
ax = ds.plot()
ax.set_title("NH4Y Activation Spectrum")
ax.set_xlabel("Wavenumber (cm⁻¹)")
ax.set_ylabel("Absorbance (a.u.)")

# %% [markdown]
# ## Grid
#
# Add a grid for easier reading:

# %%
_ = ds.plot(grid=True)

# %% [markdown]
# ## Colormap
#
# As shown in the overview, change colors using the `cmap` argument:

# %%
_ = ds.plot(cmap="viridis")

# %% [markdown]
# ### Categorical Colors (cmap=None)
#
# For line plots, passing `cmap=None` uses categorical colors instead of a continuous colormap:

# %%
_ = ds.plot_lines(cmap=None)

# %% [markdown]
# ## Colormap Normalization
#
# You can customize how colors map to values using Matplotlib's normalization classes:

# %%
import matplotlib as mpl

# Centered norm - useful for data with a natural center (e.g., deviations from mean)
norm = mpl.colors.CenteredNorm()
_ = ds.plot_image(cmap="RdBu_r", norm=norm)

# %% [markdown]
# Log norm - useful for data spanning several orders of magnitude
# norm = mpl.colors.LogNorm(vmin=0.01, vmax=1.0)
# _ = ds.plot_image(cmap="viridis", norm=norm)

# %% [markdown]
# ## Combining Customizations
#
# Most arguments can be combined for a tailored plot:

# %%
_ = ds.plot(
    xlim=(1500, 4000),
    ylim=(0, 0.8),
    linewidth=1.5,
    linestyle="-",
    grid=True,
    figsize=(10, 5),
)

# %% [markdown]
# ## The Mental Model
#
# To summarize:
#
# - `ds.plot()` gives you a clean default.
# - Keyword arguments customize a single plot.
# - The returned Axes object provides full Matplotlib control.
# - Persistent changes across sessions are handled via `scp.preferences` (covered elsewhere).
#
# These tools cover most day-to-day plotting needs.
