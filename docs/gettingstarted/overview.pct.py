# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent,md:myst
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.7
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
#     version: 3.9.9
#   widgets:
#     application/vnd.jupyter.widget-state+json:
#       state: {}
#       version_major: 2
#       version_minor: 0
# ---

# %% [markdown]
# # Overview

# %% [markdown]
# The purpose of this page is to give you some quick examples of what can be done with SpectroChemPy.
#
# See the [gallery of examples](gallery/auto_examples/index.rst) and consult the [user's guide](
# ../userguide/introduction/introduction.ipynb) for more information on using SpectroChemPy
#
# Before using the package, we must load the **API (Application Programming Interface)**

# %%
import spectrochempy as scp

# %% [markdown]
# ## NDDataset, the main object

# %% [markdown]
# NDDataset is a python object, actually a container, which can represent most of your multidimensional spectroscopic
# data.
#
# For instance, in the following we read data from a series of FTIR experiments, provided  by the OMNIC software:

# %%
ds = scp.read("irdata/nh4y-activation.spg")

# %% [markdown]
# ### Display dataset information

# %% [markdown]
# Short information:

# %%
print(ds)

# %% [markdown]
# Detailed information on the main metadata:

# %%
ds

# %% [markdown]
# ### Plotting a dataset

# %%
_ = ds.plot()

# %% [markdown]
# ### Slicing a dataset

# %%
region = ds[:, 4000.0:2000.0]
_ = region.plot()

# %% [markdown]
# ### Maths on datasets

# %% tags=[]
region.y -= region.y[0]  # make y coordinate relative to the first point
region.y.title = "time of dehydratation"
region -= region[-1]  # suppress the last spectra to all
_ = region.plot(colorbar=True)

# %% [markdown]
# ### Processing a dataset

# %% [markdown]
# We just give here few examples

# %% [markdown]
# #### Smoothing

# %% tags=[]
smoothed = region.smooth(window_length=51, window="hanning")
_ = smoothed.plot(colormap="magma")

# %% [markdown]
# #### Baseline correction

# %% tags=[]
region = ds[:, 4000.0:2000.0]
smoothed = region.smooth(window_length=51, window="hanning")
blc = scp.BaselineCorrection(smoothed)
basc = blc.compute(
    [2000.0, 2300.0],
    [3800.0, 3900.0],
    method="multivariate",
    interpolation="pchip",
    npc=5,
)

# %% tags=[]
_ = basc.plot()

# %% [markdown]
# ### Analysis
#
# #### IRIS processing

# %% tags=[]
ds = scp.NDDataset.read_omnic("irdata/CO@Mo_Al2O3.SPG")[:, 2250.0:1950.0]
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
_ = ds.plot(colormap="magma")

# %% jupyter={"source_hidden": true} pycharm={"name": "#%%\n"}

iris = scp.IRIS(ds, "langmuir", q=[-8, -1, 50], reg_par=[-10, 1, 12])
_ = iris.plotdistribution(-7, colormap="magma")
