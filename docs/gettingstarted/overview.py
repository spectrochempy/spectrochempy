# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.10.2
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
#     version: 3.8.0
#   widgets:
#     application/vnd.jupyter.widget-state+json:
#       state: {}
#       version_major: 2
#       version_minor: 0
# ---

# %% [markdown] slideshow={"slide_type": "slide"}
# # Overview

# %% [markdown]
# The purpose of this page is to give you some quick examples of what can be done with SpectroChemPy.
#
# See the [gallery of examples](generated/auto_examples/index.html) and consult the [user's guide](
# ../user/index.html) for more information on using SpectroChemPy
#
# Before using the package, we must load the **API
# (Application Programming Interface)**
#
# The simplest way is to import all objects and methods at once into your
# python namespace. The loading step may take several seconds due to the
# large number of methods to be imported into the API namespace.

# %%
from spectrochempy import *

# %% [markdown]
# ## NDDataset, the main object

# %% [markdown]
# NDDataset is a python object, actually a container, which can represent most of your multidimensional spectroscopic
# data.
#
# For instance, in the following we read data from a series of FTIR experiments, provided  by the OMNIC software:

# %%
ds = read('irdata/nh4y-activation.spg')

# %% [markdown]
# ### Display dataset information

# %% [markdown]
# Short information

# %%
ds

# %% [markdown]
# Detailled information:

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

# %% jupyter={"source_hidden": true}
region.y -= region.y[0]  # make y coordinate reative to the first point
region.y.title = 'time of dehydratation'
region -= region[-1]  # suppress the last spectra to all
_ = region.plot(colorbar=True)

# %% [markdown]
# ### Processing a dataset

# %% [markdown]
# We just give here few examples

# %% [markdown]
# #### Smoothing

# %% jupyter={"source_hidden": true}
smoothed = region.smooth(window_length=51, window='hanning')
_ = smoothed.plot(colormap='magma')

# %% [markdown]
# #### Baseline correction

# %% jupyter={"source_hidden": true}
region = ds[:, 4000.0:2000.0]
smoothed = region.smooth(window_length=51, window='hanning')
blc = BaselineCorrection(smoothed, method='multivariate', interpolation='pchip', npc=5)
basc = blc.compute([2000., 2300.], [3800., 3900.])

# %% jupyter={"source_hidden": true}
_ = basc.plot()

# %% [markdown]
# ### Analyis
#
# #### IRIS processing

# %% jupyter={"source_hidden": true}
ds = NDDataset.read_omnic('irdata/CO@Mo_Al2O3.SPG')[:, 2250.:1950.]
pressure = [0.00300, 0.00400, 0.00900, 0.01400, 0.02100, 0.02600, 0.03600, 0.05100, 0.09300, 0.15000, 0.20300, 0.30000,
            0.40400, 0.50300, 0.60200, 0.70200, 0.80100, 0.90500, 1.00400]
ds.y = Coord(pressure, title='Pressure', units='torr')
_ = ds.plot(colormap='magma')

# %% jupyter={"source_hidden": true}
param = {'epsRange': [-8, -1, 50], 'lambdaRange': [-10, 1, 12], 'kernel': 'langmuir'}

iris = IRIS(ds, param, verbose=False)
_ = iris.plotdistribution(-7, colormap='magma')

# %%
