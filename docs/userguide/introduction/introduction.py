# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
#     formats: ipynb,py:percent
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.7
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

# %% [markdown] {"slideshow": {"slide_type": "slide"}}
# # Introduction
#
# The **SpectroChemPy** project was developed to provide advanced tools for processing and
# analyzing spectroscopic data, initially for internal purposes in the
# [LCS (https://www.lcs.ensicaen.fr)](https://www.lcs.ensicaen.fr).
#
# **SpectroChemPy** is essentially a library written in python language and which proposes objects (`NDDataset`, and
# `Project`) to contain data, equipped with methods to analyze, transform or display
# this data in a simple way by the user.
#
#  The processed data are mainly spectroscopic data from techniques such as IR, Raman or NMR, but they are not limited
# to this type of application, as any type of numerical data arranged in tabular form can generally serve as the main
# input.

# %% [markdown] {"slideshow": {"slide_type": "slide"}}
# ## How to get started
#
# <div class='alert alert-info'>
# <b>Note</b>
#
# We assume that the SpectroChemPy package has been correctly
# installed. if is not the case, please go to [SpectroChemPy installation
# procedure](../../gettingstarted/install/index.rst).
# </div>

# %% [markdown] {"nbsphinx-toctree": {"maxdepth": 3}, "slideshow": {"slide_type": "subslide"}}
# [interface](interface.ipynb)

# %% [markdown] {"slideshow": {"slide_type": "slide"}}
# ## Loading the API
#
# Before using SpectroChemPy, we need to load the **API (Application Programming Interface)**: it exposes many
# objects and functions.
#
# To load the API, you must import it using one of the following syntax.
#
# In the first syntax we load the library into a namespace called `scp`
# (we recommend this name, but you can choose whatever
# you want - except something already in use):

# %%
import spectrochempy as scp  # SYNTAX 1

nd = scp.NDDataset()


# %% [markdown]
# or in the second syntax, with a wild `*` import.

# %%
from spectrochempy import *  # SYNTAX 2

nd = NDDataset()

# %% [markdown]
# With the second syntax, as often in python, the access to objects/functions can be greatly simplified. For example,
# we can use directly `NDDataset` without a prefix instead of `scp.NDDataset`
# which is the first syntax but there is always a
# risk of overwriting some variables or functions already present in the namespace.
# Therefore, the first syntax is generally highly recommended.
#
# Alternatively, you can also load only the objects and function required by your application:
#

# %%
from spectrochempy import NDDataset  # SYNTAX 3

nd = NDDataset()


# %% [markdown]
# ## NDDataset, the main object
#
# NDDataset is a python object, actually a container, which can represent most of your multidimensional spectroscopic
# data.
#
# For instance, in the following we read data from a series of FTIR experiments, provided  by the OMNIC software:

# %%
import os

nd = NDDataset.read_omnic(os.path.join("irdata", "nh4y-activation.spg"))

# %% [markdown]
# Note that for this example, we use data stored in a ``test`` directory. For your own usage, you probably have to
# give the full pathname (see ... for the way to overcome this using `preferences` setting)

# %% [markdown]
# ### Display dataset information

# %% [markdown]
# Several ways are available to display the data we have just read and that are now stored in the dataset
#
# * **Printing** them, using the print function of python to get a short text version of the dataset information.

# %%
print(nd)

# %% [markdown]
# A much Longer (and colored) information text can be obtained using the spectrochempy provided ``print_`` function.

# %%
scp.print_(nd)

# %% [markdown]
# * **Displaying html**, inside a jupyter notebook, by just typing the name of the dataset (must be the last
# instruction of a cell, however!)

# %%
nd

# %% [markdown]
# ### Plotting a dataset
#
# First, we can set some general plotting preferences for this dataset

# %%
prefs = nd.preferences
prefs.reset()
prefs.figure.figsize = (6, 3)

# %% [markdown]
# Let's plot first a 1D spectrum (for instance one row of nd)

# %%
row = nd[-1]
_ = row.plot()

# %% [markdown]
# or a column ...

# %%
col = nd[:, 3500.0]  # note the indexing using wavenumber!
_ = col.plot_scatter()

# %% [markdown]
# 2D plots can be also generated as stacked plot

# %%
_ = nd.plot(method="stack")  # or nd.plot_stack()

# %% [markdown]
# or as an image plot:

# %%
prefs.colormap = "magma"
_ = nd.plot(method="image")  # or nd.plot_image()

# %% [markdown]
# Note that as we plot wavenumbers as abscissa, by convention the coordinates' direction is reversed.
#
# This can be changed by using the keyword argument `reversed = False`.

# %% [markdown]
# ### Processing a dataset

# %% [markdown]
# Some arithmetic can be performed on such dataset. Here is an example where we subtract one reference spectrum to
# the whole nddataset that we have read above (`nd`).

# %% [markdown]
# Let's take, e.g., the last row as reference

# %%
ref = nd[0]
_ = ref.plot()

# %% [markdown]
# Now suppress this ref spectrum to all other spectra of the whole dataset (additionally we mask the region of
# saturation

# %%
prefs.colormap = "jet"
prefs.colorbar = True
nds = nd - ref
nds[:, 1290.0:890.0] = scp.MASKED
_ = nds.plot_stack()

# %% [markdown]
# More details on available processing and analysis function will be given later in this user guide.

# %% [markdown]
# This was a short overview of the possibilities. To go further you can **Continue with ...**

# %% [markdown] {"nbsphinx-toctree": {"maxdepth": 3}}
# [Data structures](../objects.rst)
