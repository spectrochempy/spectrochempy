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
#       jupytext_version: 1.14.5
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
#     version: 3.9.16
#   widgets:
#     application/vnd.jupyter.widget-state+json:
#       state: {}
#       version_major: 2
#       version_minor: 0
# ---

# %% [markdown]
# # Introduction
#
# The `SpectroChemPy` project was developed to provide advanced tools for processing and
# analyzing spectroscopic data, initially for internal purposes within the
# [LCS (https://www.lcs.ensicaen.fr)] (https://www.lcs.ensicaen.fr).
#
# The `SpectroChemPy` project is essentially a library written in
# [python](https://www.python.org) language and provides objects,
# [NDDataset](../../reference/generated/spectrochempy.NDDataset.rst)
# and [Project](../../reference/generated/spectrochempy.Project.rst),
# to hold data, equipped with methods to analyze, transform or display these data in a
# simple way through a python type interface.
#
# The processed data are mainly spectroscopic data from techniques such as IR, Raman or
# NMR, but they are not limited to this type of application, as any type of data can be
# used.

# %% [markdown]
# ## How to get started
#
# We assume that the SpectroChemPy package has been correctly
# installed. if is not the case, please go to [SpectroChemPy installation
# procedure](../../gettingstarted/install/index.rst).

# %% [markdown]
# ### Writing and executing SpectroChemPy scripts
#
# If you are already an experienced `python` user, you can certainly use your favorite
# IDE to run your scripts, debug them and display the results. But if you want an easier
# way, especially if you are a beginner, we recommend you to use `Jupyter Lab` to do it.
# To get started, you can follow links below:

# %% [markdown]
# [Jupyter Lab interface](interface.ipynb).

# %% [markdown]
# ### Loading the API
#
# Before using SpectroChemPy, we need to load the **API
# (Application Programming Interface)**: it exposes many
# objects and functions.
#
# To load the API, you must import it using one of the following syntax.
#
# **Syntax 1** (recommended)
#
# In the first syntax we load the library into a namespace called `scp`
# (we recommend this name, but you can choose whatever
# you want - except something already in use):

# %%
import spectrochempy as scp

nd = scp.NDDataset()

# %% [markdown]
# **Syntax 2** (discouraged)
#
# With a wild `*` import. In this second syntax, the access to objects/functions can be
# greatly simplified. For example, we can use directly `NDDataset` without a prefix
# instead of `scp.NDDataset` but there is always a risk of overwriting some variables or
# functions already present in the namespace. Therefore, the first syntax is generally
# highly recommended.

# %%
from spectrochempy import *  # noqa

nd = NDDataset()

# %% [markdown]
# Alternatively, you can also load only the objects and function required by your
# application:

# %%
from spectrochempy import NDDataset

nd = NDDataset()


# %% [markdown]
# ## NDDataset, the main object
#
# NDDataset is a python object, actually a container, which can represent most of your
# multidimensional spectroscopic data.
# For instance, in the following we read data from a series of FTIR experiments,
# provided  by the OMNIC software

# %%
nd = NDDataset.read_omnic("irdata/nh4y-activation.spg")

# %% [markdown]
# Note that for this example, we use data stored in a `test` directory. For your own
# usage, you probably have to give the full pathname.

# %% [markdown]
# ### Display dataset information

# %% [markdown]
# Several ways are available to display the data we have just read and that are now
# stored in the dataset
#
# * **Printing** them, using the print function of python to get a short text version
# of the dataset information.

# %%
print(nd)

# %% [markdown]
# * **Displaying html**, inside a jupyter notebook, by just typing the name of the
# dataset (must be the last
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
# Note that as we plot wavenumbers as abscissa, by convention the coordinates' direction
# is reversed.
#
# This can be changed by using the keyword argument `reversed = False` .

# %% [markdown]
# ### Processing a dataset

# %% [markdown]
# Some arithmetic can be performed on such dataset. Here is an example where we subtract
# one reference spectrum to
# the whole nddataset that we have read above (`nd` ).

# %% [markdown]
# Let's take, e.g., the last row as reference

# %%
ref = nd[0]
_ = ref.plot()

# %% [markdown]
# Now suppress this ref spectrum to all other spectra of the whole dataset (additionally
# we mask the region of
# saturation

# %%
prefs.colormap = "jet"
prefs.colorbar = True
nds = nd - ref
nds[:, 1290.0:890.0] = scp.MASKED
_ = nds.plot_stack()

# %% [markdown]
# More details on available processing and analysis functions will be given later in
# this user guide.
