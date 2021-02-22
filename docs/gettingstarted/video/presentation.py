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
#     version: 3.9.1
#   widgets:
#     application/vnd.jupyter.widget-state+json:
#       state: {}
#       version_major: 2
#       version_minor: 0
# ---

# %% [markdown] slideshow={"slide_type": "slide"}
# # Overview

# %% [markdown]
# In this presentation, we will shortly present some of the **SpectroChemPy** features.

# %% [markdown] slideshow={"slide_type": "notes"}
# ## Using the Jupyter Lab environment

# %% [markdown]
# PRobably the most easiet way to start experiencing SpectroChemPy
#
# <img src='images/enter_md.png' />

# %% [markdown] slideshow={"slide_type": "subslide"}
# We first need to load the API

# %% slideshow={"slide_type": "fragment"} tags=["hide-output"]
import spectrochempy as scp

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## NDDataset, the main object

# %% [markdown] slideshow={"slide_type": "subslide"}
# NDDataset is a python object, actually a container, which can represent most of your multidimensional spectroscopic
# data.

# %% [markdown] slideshow={"slide_type": "subslide"}
# For instance, in the following we read data from a series of FTIR experiments,
# provided  by the OMNIC software, and create a **NDDataset** from these data

# %%
ds = scp.read('irdata/nh4y-activation.spg')
print(ds)

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Display dataset information

# %%
ds

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Plotting a dataset

# %% slideshow={"slide_type": "fragment"}
_ = ds.plot()

# %% [markdown] slideshow={"slide_type": "subslide"}
# #### Modifying the figure

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Slicing a dataset

# %% slideshow={"slide_type": "fragment"}
region = ds[:, 4000.0:2000.0]
_ = region.plot()

# %% [markdown] slideshow={"slide_type": "subslide"}
# ### Maths on datasets

# %% slideshow={"slide_type": "fragment"}
region.y -= region.y[0]  # make y coordinate reative to the first point
region.y.title = 'time of dehydratatioin'
region -= region[-1]  # suppress the last spectra to all
_ = region.plot(colorbar=True)

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## Processing a dataset

# %% [markdown] slideshow={"slide_type": "subslide"}
# We just give here few examples

# %% [markdown] slideshow={"slide_type": "subslide"}
# #### Smoothing

# %% slideshow={"slide_type": "fragment"}
smoothed = region.smooth(window_length=51, window='hanning')
_ = smoothed.plot(colormap='magma')

# %% [markdown] slideshow={"slide_type": "subslide"}
# #### Baseline correction

# %% slideshow={"slide_type": "fragment"}
region = ds[:, 4000.0:2000.0]
smoothed = region.smooth(window_length=51, window='hanning')
blc = scp.BaselineCorrection(smoothed, method='multivariate', interpolation='pchip', npc=5)
basc = blc.compute([2000., 2300.], [3800., 3900.])
_ = basc.plot()

# %% [markdown] slideshow={"slide_type": "subslide"}
# ## IRIS processing

# %% slideshow={"slide_type": "subslide"}
ds = scp.read('irdata/CO@Mo_Al2O3.SPG')[:, 2250.:1950.]
pressure = [0.00300, 0.00400, 0.00900, 0.01400, 0.02100, 0.02600, 0.03600,
            0.05100, 0.09300, 0.15000, 0.20300, 0.30000, 0.40400, 0.50300,
            0.60200, 0.70200, 0.80100, 0.90500, 1.00400]
ds.y = scp.Coord(pressure, title='Pressure', units='torr')
_ = ds.plot(colormap='magma')

# %% slideshow={"slide_type": "subslide"}
param = {'epsRange': [-8, -1, 50], 'lambdaRange': [-10, 1, 12], 'kernel': 'langmuir'}

iris = scp.IRIS(ds, param, verbose=False)
_ = iris.plotdistribution(-7, colormap='magma')
