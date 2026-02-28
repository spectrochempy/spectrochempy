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
# # Advanced Matplotlib Integration
#
# SpectroChemPy plots return Matplotlib Axes objects, giving you full
# access to Matplotlib's capabilities.

# %% [markdown]
# ## Modifying the Axes
#
# After plotting, customize using Matplotlib methods:

# %%
import spectrochempy as scp

ds = scp.read("irdata/nh4y-activation.spg")
ds1 = ds[0]

# %%
ax = ds1.plot()
ax.set_title(r"NH$_4$Y Activation - $\nu_{NH}$ Region")
ax.set_xlabel(r"Wavenumber (cm$^{-1}$)")
ax.set_ylabel("Absorbance (a.u.)")
ax.set_xlim(3500, 2800)
ax.annotate(
    "NH stretch",
    xy=(3250, 0.6),
    xytext=(3400, 0.7),
    arrowprops=dict(arrowstyle="->", color="gray"),
)

# %% [markdown]
# ## Multiple Plots
#
# Create separate plots with different settings:

# %%
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))

# Plot 1: full spectrum
_ = ds.plot(ax=ax1)
ax1.set_title("Full Spectrum")

# Plot 2: subset
_ = ds[:,1800.:1500.].plot(ax=ax2)
ax2.set_title("Water bending Region")

plt.tight_layout()


# %% [markdown]
# ## Colormap Normalization
#
# Advanced colormap normalization for special data scenarios:

# %%
import matplotlib as mpl

# CenteredNorm - centers the colormap around a specific value
norm = mpl.colors.CenteredNorm(vcenter=1.0)
_ = ds.plot_image(cmap="RdBu_r", norm=norm, colorbar=True)

# %% [markdown]
# ## LaTeX-like Math in Labels
#
# SpectroChemPy supports LaTeX math notation in labels:

# %%
ax = ds1.plot()
ax.set_xlabel(r"$ \tilde{\nu}$ (cm$^{-1}$)")
ax.set_ylabel(r"$ \epsilon$ (mol$^{-1}$·L·cm$^{-1}$)")
ax.set_title(r"Beer-Lambert: $A = \epsilon c l$")

# %% [markdown]
# ## Reproducibility
#
# Avoid modifying global Matplotlib state. Instead:
#
# - Use **kwargs** for per-plot settings
# - Use **preferences** for session defaults
# - Use **styles** for theme changes

# %% [markdown]
# Example of clean, reproducible plotting:


# %%
def plot_spectrum(dataset, title=None, output_path=None):
    """Plot a spectrum with consistent styling."""
    ax = dataset.plot(
        linewidth=1.5,
        color="navy",
        grid=True,
    )
    if title:
        ax.set_title(title)
    ax.set_xlabel(r"Wavenumber (cm$^{-1}$)")
    ax.set_ylabel("Absorbance")
    return ax


# Each call produces consistent results
ax1 = plot_spectrum(ds1, title="Sample 1")
ax2 = plot_spectrum(ds1 * 1.5, title="Sample 2 (amplified)")

# %% [markdown]
# ## Where to Go Further
#
# SpectroChemPy is built on Matplotlib. For advanced customization:
#
# - [Matplotlib Axes documentation](https://matplotlib.org/stable/api/axes_api.html)
# - [Matplotlib customization guide](https://matplotlib.org/stable/tutorials/introductory/customizing.html)
# - SpectroChemPy API reference for plot method options
#
# The combination of SpectroChemPy's convenience with Matplotlib's power
# gives you full control over your visualizations.
