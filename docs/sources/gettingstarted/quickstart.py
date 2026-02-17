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
#   widgets:
#     application/vnd.jupyter.widget-state+json:
#       state: {}
#       version_major: 2
#       version_minor: 0
# ---

# %%
# ruff: noqa: T201
# %% [markdown]
# # Quickstart Tutorial 🚀
#
# **Contents**
# - [Installing SpectroChemPy](#installing-spectrochempy)
# - [Getting Started](#getting-started)
# - [Working with Spectroscopic Data](#working-with-spectroscopic-data)
# - [Data Processing Techniques](#data-processing-techniques)
# - [Advanced Analysis](#advanced-analysis)
#
# > **What you'll learn**
# > * Load and visualize spectroscopic data
# > * Perform basic data manipulation and plotting
# > * Apply common processing techniques
# > * Use advanced analysis methods

# %% [markdown]
# ## Installing SpectroChemPy
#
# > **Prerequisites** ✅
# > * Python 3.11 or later
# > * Basic knowledge of Python
# > * Jupyter notebook environment
#
# You can install SpectroChemPy using either pip or mamba:
#
# **Using pip:**
# ```bash
# pip install spectrochempy
# ```
#
# **Using mamba:**
# ```bash
# mamba install -c spectrocat spectrochempy
# ```
#
# See the [Installation Guide](install/index.rst) for detailed instructions.

# %% [markdown]
# In the following, we assume that we are running `spectrochempy` in a Jupyter notebook.
# See [here](../userguide/introduction/interface.ipynb) for details on how to start a Jupyter notebook.

# %% [markdown]
# ## Getting Started
#
# Let's start by importing SpectroChemPy and checking its version:

# %%
import spectrochempy as scp

# %% [markdown]
# ## Working with Spectroscopic Data
#
# ### Loading Data
#
# SpectroChemPy supports many file formats including:
# * OMNIC (.spa, .spg)
# * JCAMP-DX (.dx, .jdx)
# * CSV files
# * And [many more](../userguide/importexport/import.ipynb)
#
# Let's load an example FTIR dataset:

# %%
# Load FTIR data of NH4Y zeolite activation
ds = scp.read("irdata/nh4y-activation.spg")
print(f"Dataset shape: {ds.shape}")  # Show dimensions
print(f"x-axis unit: {ds.x.units}")  # Show wavenumber units
print(f"y-axis unit: {ds.y.units}")  # Show time units

# %% [markdown]
# The `read` function is a powerful method that can load various file formats. In this case,
# we loaded an OMNIC file. For a full list of supported formats, see the
# [Import tutorial](../userguide/importexport/import.ipynb) section.
#
# The `read` function returns an `NDDataset` object.

# %% [markdown]
# ## Exploring the Data

# %% [markdown]
# ### Understanding the NDDataset object
# The `NDDataset` is the core data structure in SpectroChemPy. It's designed specifically
# for spectroscopic data and provides:
#
# * Multi-dimensional data support
# * Coordinates and units handling
# * Built-in visualization
# * Processing methods

# %% [markdown]
# You can display the loaded NDDataset in a Jupyter notebook as follows:

# %%
ds

# %% [markdown]
# Basic information about the data are given in the summary: data type, units, shape, and name of the dataset.
#
# Clicking on the arrow on the left side of the summary will expand the metadata section,
# which contains additional information about the dataset.
#
# The data itself is contained in the `data` attribute, which is a `numpy` array of shape (55,5549).

# %%
ds.data

# %% [markdown]
# The `x` and `y` attributes contain the coordinates of the dataset.
# In this case, the x-axis represents the wavenumber:

# %%
ds.x

# %% [markdown]
# The y-axis represents the sample acquisition time:

# %%
ds.y

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
# NDDataset supports various mathematical operations. Here's an example of basic operations on coordinates and data:

# %% [markdown]
# Make y coordinate (time) relative to the first spectrum, convert to hours (default are seconds), and update the title
# to reflect the new meaning of the y-axis.

# %%
region.y -= region.y[0]
region.y.ito("hour")
region.y.title = "Dehydration time"

# %% [markdown]
# Subtract the last spectrum from all spectra

# %%
region -= region[-1]

# %% [markdown]
# Plot with colorbar to show intensity changes

# %%
_ = region.plot(colorbar=True)

# %% [markdown]
# ### Other Operations
#
# NDDataset supports many other operations, such as:
# - Arithmetic operations
# - Statistical analysis
# - Data transformation
# - And more
#
# For more information, see:
#
# - [More insight on the NDDataset objects](../userguide/objects/dataset/dataset.ipynb) section.
# - [API Reference](../reference/index.rst) for a full list of available operations.
# - [Plotting tutorial](../userguide/plotting/plotting.ipynb) for more information on advanced plotting.

# %% [markdown]
# ## Data Processing Techniques
#
# SpectroChemPy includes numerous processing methods. Here are some common examples:

# %% [markdown]
# ### Spectral Smoothing
#
# Reduce noise while preserving spectral features:

# %%
smoothed = region.smooth(size=51, window="hanning")
_ = smoothed.plot(colormap="magma")

# %% [markdown]
# ### Baseline Correction
#
# Various algorithms are available for baseline correction, including polynomial fitting, rubberband, and more. Here an
# example of multivariate polynomial baseline correction using PCHIP interpolation is shown:

# %% [markdown]
# Configure baseline correction

# %%
blc = scp.Baseline()
blc.ranges = [[2000.0, 2300.0], [3800.0, 3900.0]]  # Baseline regions
blc.multivariate = True
blc.model = "polynomial"
blc.order = "pchip"
blc.n_components = 5

# %% [markdown]
# Apply correction

# %%
blc.fit(smoothed)
_ = blc.corrected.plot()

# %% [markdown]
# SpectroChemPy provides many other processing techniques, such as:
# * Normalization
# * Derivatives
# * Peak detection
# * And more
#
# For more information, see the [Processing tutorial](../userguide/processing/index.rst) section.

# %% [markdown]
# ## Advanced Analysis
#
# ### 2D-IRIS Processing example
#
# Two-Dimensional Integral Regularized Inversion for Spectroscopy (2D-IRIS) is an advanced technique for analyzing
# spectroscopic data. In the following example, spectra of CO adsorption on a sulfide catalyst at various pressures
# are used to extract the distribution of adsorption energies using 2D-IRIS analysis.

# %% [markdown]
# Load and prepare CO adsorption data

# %%
ds = scp.read_omnic("irdata/CO@Mo_Al2O3.SPG")[:, 2250.0:1950.0]

# %% [markdown]
# Define pressure coordinates

# %%
pressure = [
    0.003,
    0.004,
    0.009,
    0.014,
    0.021,
    0.026,
    0.036,
    0.051,
    0.093,
    0.150,
    0.203,
    0.300,
    0.404,
    0.503,
    0.602,
    0.702,
    0.801,
    0.905,
    1.004,
]
ds.y = scp.Coord(pressure, title="Pressure", units="torr")

# %% [markdown]
# Plot the dataset

# %%
_ = ds.plot()

# %% [markdown]
# Perform IRIS analysis assuming a local Langmuir isotherms and plot the distribution of adsorption energies at a
# specific regularization parameter.

# %%
iris = scp.IRIS(reg_par=[-10, 1, 12])
K = scp.IrisKernel(ds, "langmuir", q=[-7, -1, 50])
iris.fit(ds, K)
_ = iris.f[-7].plot(method='map', colorbar=True)

# %%
# %% [markdown]
# ### Other Advanced Analysis Techniques
# SpectroChemPy includes many other advanced analysis techniques, such as:
# * Multivariate analysis
# * Peak fitting
# * Kinetic modeling
# * And more
#
# For more information, see the [Advanced Analysis tutorial](../userguide/analysis/index.rst) section.

# %% [markdown]
# ## Next Steps 🎯
#
# Now that you've got a taste of SpectroChemPy's capabilities, here are some suggestions for diving deeper:
#
# - [Examples Gallery](examples/index) 📈: Browse through practical examples and use cases
# - [User Guide](../userguide/index) 📖: Learn about specific features in detail
# - [API Reference](../reference/index) 🔍: Explore the complete API documentation
# - [Get Help](https://github.com/spectrochempy/spectrochempy/discussions) 💬: Join our community discussions
