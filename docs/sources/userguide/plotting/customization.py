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
import os
from pathlib import Path

import spectrochempy as scp

TEST_FILE = Path(os.environ.get("TEST_FILE", "irdata/nh4y-activation.spg"))

ds = scp.read(TEST_FILE)

# %% [markdown]
# ### Color

# %%
_ = ds.plot(color="red")

# %% [markdown]
# ### Line Width

# %%
_ = ds.plot(linewidth=2)

# %% [markdown]
# The short alias `lw=...` is accepted as well:

# %%
_ = ds.plot(lw=2)

# %% [markdown]
# ### Line Style

# %%
_ = ds.plot(linestyle="--")

# %% [markdown]
# The corresponding short alias is `ls=...`:

# %%
_ = ds.plot(ls="--")

# %% [markdown]
# ### Marker

# %%
_ = ds.plot(marker="o")

# %% [markdown]
# Marker-related aliases such as `ms` (marker size) and `mew`
# (marker edge width) are normalized automatically:

# %%
_ = ds.plot(marker="o", ms=5, mew=1.5)


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
# The aliases `c=...` for line color and `colormap=...` for `cmap=...` are also
# accepted:

# %%
_ = ds[0].plot(c="darkred")

# %%
_ = ds.plot_image(colormap="plasma")

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
# ## Composite Plot Customization
#
# Analysis and comparison plots such as `plot_score`, `plot_compare`,
# `plot_merit`, `plot_baseline`, and `plot_parity` accept keyword arguments
# for fine-grained control over visual appearance.
#
# ### Score Plot: Marker and Transparency
#
# The `plot_score` function (accessible via `PCA.plot_score` or standalone
# `scp.plot_score`) supports `marker`, `s` (marker size), and `alpha`
# (transparency):
#
# %%
# Load example data and fit PCA
# (requires an NDDataset - replace with your own data)
import spectrochempy as scp

# %% [markdown]
# For a fitted PCA model:
#
# ```python
# pca = scp.PCA(n_components=5).fit(dataset)
# ax = pca.plot_score(marker="s", s=30, alpha=0.7)
# ```
#
# These parameters are forwarded to `matplotlib.axes.Axes.scatter`.
#
# ### Compare / Merit Plot: Line Style Parameters
#
# The `plot_compare` function accepts domain-specific keyword aliases
# for style control:
#
# | Parameter | Effect | Default |
# |---|---|---|
# | `exp_linestyle`, `calc_linestyle`, `resid_linestyle` | Line style per category | `"-"`, `"--"`, `"-"` |
# | `exp_linewidth`, `calc_linewidth`, `resid_linewidth` | Line width per category | `1.0`, `1.6`, `1.0` |
# | `exp_c`, `calc_c`, `resid_c` | Color per category | Semantic defaults |
# | `offset` | Vertical separation between groups | `None` |
# | `nb_traces` | Number of traces to display | `"all"` |
#
# Example:
#
# ```python
# scp.plot_compare(original, reconstructed,
#     exp_linestyle="-", calc_linestyle=":",
#     exp_linewidth=1.5, calc_linewidth=2.0,
#     offset=0.1)
# ```
#
# ### Parity Plot: Scatter Parameters
#
# The `plot_parity` function (accessible via `pls.plot_parity` or standalone
# `scp.plot_parity`) passes keyword arguments directly to
# `matplotlib.axes.Axes.scatter`:
#
# ```python
# plot_parity(Y, Y_hat, s=20, marker="o", alpha=0.5, c="tab:blue")
# ```
#
# ### Lifecycle Control for Composites
#
# All composite functions accept the same lifecycle parameters:
#
# - `ax`: Existing axes to plot on (creates new figure if `None`).
# - `clear`: Whether to clear the axes before plotting (`True` by default;
#   set to `False` to overlay on existing axes).
# - `show`: Whether to display the figure (`True` by default).
#
# ```python
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
# plot_score(scores, ax=ax1, show=False)
# plot_parity(Y, Y_hat, ax=ax2, show=True)
# ```
#
# ## The Mental Model
#
# To summarize:
#
# - `ds.plot()` gives you a clean default.
# - Keyword arguments customize a single plot.
# - The returned Axes object provides full Matplotlib control.
# - Composite plots (`plot_score`, `plot_compare`, etc.) have their own
#   domain-specific parameters but follow the same `ax`/`clear`/`show` contract.
# - Persistent changes across sessions are handled via `scp.preferences` (covered elsewhere).
#
# These tools cover most day-to-day plotting needs.
