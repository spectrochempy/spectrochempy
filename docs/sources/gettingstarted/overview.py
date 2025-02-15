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
#     version: 3.11.11
#   widgets:
#     application/vnd.jupyter.widget-state+json:
#       state: {}
#       version_major: 2
#       version_minor: 0
# ---

# %% [markdown]
# # SpectroChemPy Overview
#
# Welcome to SpectroChemPy! This tutorial will give you a quick overview of the main features
# and capabilities of the library. We'll cover:
#
# 1. Loading and displaying spectroscopic data
# 2. Basic data manipulation and plotting
# 3. Processing techniques (smoothing, baseline correction)
# 4. Advanced analysis methods
#
# For more detailed information, check out:
# * [Gallery of Examples](examples/index.rst)
# * [User's Guide](../userguide/introduction/introduction.rst)
# * [API Reference](../reference/index.rst)

# %% [markdown]
# ## Getting Started
#
# First, let's import SpectroChemPy. By convention, we use the alias `scp`:

# %%
import spectrochempy as scp

# %% [markdown]
# ## Working with NDDataset Objects
#
# The `NDDataset` is the core data structure in SpectroChemPy. It's designed specifically
# for spectroscopic data and provides:
#
# * Multi-dimensional data support
# * Coordinates and units handling
# * Built-in visualization
# * Processing methods
#
# Let's load some example FTIR (Fourier Transform Infrared) data:

# %%
ds = scp.read("irdata/nh4y-activation.spg")

# %% [markdown]
# ### Exploring Your Data
#
# SpectroChemPy provides multiple ways to inspect your data. Let's look at:
#
# 1. Basic information summary
# 2. Detailed metadata

# %%
# Quick overview
scp.set_loglevel("INFO")  # to see information
scp.info_(ds)

# %%
# Detailed information
ds

# %% [markdown]
# ### Data Visualization
#
# SpectroChemPy's plotting capabilities are built on matplotlib but provide
# spectroscopy-specific features:

# %%
_ = ds.plot()

# %% [markdown]
# ### Data Selection and Manipulation
#
# You can easily select specific regions of your spectra using intuitive slicing.
# Here we select wavenumbers between 4000 and 2000 cm⁻¹:

# %%
region = ds[:, 4000.0:2000.0]
_ = region.plot()

# %% [markdown]
# ### Mathematical Operations
#
# NDDataset supports various mathematical operations. Here's an example of baseline correction:

# %%
# Make y coordinate relative to the first point
region.y -= region.y[0]
region.y.title = "Dehydration time"

# Subtract the last spectrum from all spectra
region -= region[-1]

# Plot with colorbar to show intensity changes
_ = region.plot(colorbar=True)

# %% [markdown]
# ## Data Processing Techniques
#
# SpectroChemPy includes numerous processing methods. Here are some common examples:

# %% [markdown]
# ### 1. Spectral Smoothing
#
# Reduce noise while preserving spectral features:

# %%
smoothed = region.smooth(size=51, window="hanning")
_ = smoothed.plot(colormap="magma")

# %% [markdown]
# ### 2. Baseline Correction
#
# Remove baseline artifacts using various algorithms:

# %%
# Prepare data
region = ds[:, 4000.0:2000.0]
smoothed = region.smooth(size=51, window="hanning")

# Configure baseline correction
blc = scp.Baseline()
blc.ranges = [[2000.0, 2300.0], [3800.0, 3900.0]]  # Baseline regions
blc.multivariate = True
blc.model = "polynomial"
blc.order = "pchip"
blc.n_components = 5

# Apply correction
_ = blc.fit(smoothed)
_ = blc.corrected.plot()

# %% [markdown]
# ## Advanced Analysis as for instance IRIS Processing
#
# IRIS (Iterative Regularized Inverse Solver) is an advanced technique for analyzing
# spectroscopic data. Here's an example with CO adsorption data:

# %%
# Load and prepare CO adsorption data
ds = scp.read_omnic("irdata/CO@Mo_Al2O3.SPG")[:, 2250.0:1950.0]

# Define pressure coordinates
pressure = [
    0.00300,
    0.00400,
    0.00900,
    0.01400,
    0.02100,
    0.02600,
    0.03600,
    0.05100,
    0.09300,
    0.15000,
    0.20300,
    0.30000,
    0.40400,
    0.50300,
    0.60200,
    0.70200,
    0.80100,
    0.90500,
    1.00400,
]
ds.y = scp.Coord(pressure, title="Pressure", units="torr")

# Plot the dataset
_ = ds.plot(colormap="magma")

# %%
# Perform IRIS analysis
iris = scp.IRIS(reg_par=[-10, 1, 12])
K = scp.IrisKernel(ds, "langmuir", q=[-8, -1, 50])
iris.fit(ds, K)
_ = iris.plotdistribution(-7, colormap="magma")
