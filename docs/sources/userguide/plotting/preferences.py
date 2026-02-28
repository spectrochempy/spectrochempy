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
# # Plotting Preferences
#
# While keyword arguments customize individual plots, preferences control
# persistent defaults that affect all future plots in the current session and the following ones.

# %% [markdown]
# ## Accessing Preferences

# %%
import spectrochempy as scp

prefs = scp.preferences

# %% [markdown]
# Preferences are organized by category. The most commonly used are:

# %%
print("Figure size:", prefs.figure.figsize)
print("Default colormap:", prefs.colormap)
print("Font family:", prefs.font.family)
print("Current style:", prefs.style)

# %% [markdown]
# ## Changing Defaults
#
# Set a preference once; it affects all subsequent plots:

# %%
ds = scp.read("irdata/nh4y-activation.spg")

# %%
# Save original values for later restoration
original_cmap = prefs.colormap
original_figsize = prefs.figure.figsize

# %%
# Change defaults
prefs.colormap = "magma"
prefs.figure.figsize = (8, 4)

# %% [markdown]
# Now all plots use these new defaults:

# %%
_ = ds.plot()

# %% [markdown]
# ## Resetting to Defaults

# %%
# Restore original values
prefs.colormap = original_cmap
prefs.figure.figsize = original_figsize

# %% [markdown]
# Alternatively, reset all plotting preferences to defaults:

# %%
prefs.reset()

# %% [markdown]
# ## Common Preferences
#
# | Preference | Description |
# |------------|-------------|
# | `prefs.colormap` | Default colormap for plots ("auto" for automatic selection) |
# | `prefs.colormap_sequential` | Default sequential colormap (e.g., "viridis") |
# | `prefs.colormap_diverging` | Default diverging colormap (e.g., "RdBu_r") |
# | `prefs.figure.figsize` | Default figure (width, height) |
# | `prefs.font.family` | Font family (sans-serif, serif, monospace) |
# | `prefs.font.size` | Default font size |
# | `prefs.axes.grid` | Show grid by default |
# | `prefs.style` | Default visual style |

# %% [markdown]
# Example: setting font and colormap:

# %%
prefs.font.family = "serif"
prefs.colormap_sequential = "plasma"
_ = ds.plot()

# %%
# Reset for other examples
prefs.reset()

# %% [markdown]
# ## Group Access
#
# Preferences can be accessed by group. For example, all line-related preferences:

# %%
prefs.lines

# %% [markdown]
# Or all font-related preferences:

# %%
prefs.font

# %% [markdown]
# ## Getting Help
#
# Get help on a specific preference:

# %%
prefs.help("colormap")

# %% [markdown]
# ## The Mental Model
#
# - **Keyword arguments** (`ds.plot(cmap="red")`) → single plot
# - **Preferences** (`prefs.colormap = "red"`) → all future plots
# - **Style** (`ds.plot(style="grayscale")`) → appearance theme
#
# Use preferences for session-long defaults; kwargs for one-off changes.

# %% [markdown]
# Below ios the list of all available preferences:

# %%
prefs.list_all()  # Uncomment to see all preferences