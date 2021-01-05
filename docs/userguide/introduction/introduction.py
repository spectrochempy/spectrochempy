# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Introduction
#
# The **SpectroChemPy** project was developed to provide advanced tools for processing and
# analyzing spectroscopic data, initially for internal purposes in the
# [LCS (https://www.lcs.ensicaen.fr)](https://www.lcs.ensicaen.fr).
#
# **SpectroChemPy** is essentially a library written in python language and which proposes objects (`NDDataset`,
# *NDPanel(not yet available)* and `Project`) to contain data, equipped with methods to analyze, transform or display
# this data in a simple way by the user.
#
# The processed data are mainly spectroscopic data from techniques such as IR, Raman or NMR, but they are not limited
# to this type of application, as any type of numerical data arranged in tabular form can generally serve as the main
# input.

# %% [markdown]
# ## How to get started
#
# <div class='alert alert-info'>
# <b>Note</b>
#
# We assume that the SpectroChemPy package has been correctly
# installed. if is not the case, please go to [SpectroChemPy installation
# procedure](../../gettingstarted/install/index.rst).
# </div>

# %% [markdown] nbsphinx-toctree={"maxdepth": 3}
# [interface](interface.ipynb)

# %% [markdown]
# ## Loading the API
#
# Before using SpectroChemPy, we need to load the **API (Application Programming Interface)**: it exposes many
# objects and functions.
#
# To load the API, you must import it using one of the following syntax.
#
# In the first syntax we load the library into a namespace called `scp` (you can choose whatever you want - except
# something already in use):

# %%
import spectrochempy as scp  # SYNTAX 1
nd = scp.NDDataset()


# %% [markdown]
# or in the second syntax, with a wild `*` import.

# %%
from spectrochempy import *
nd = NDDataset()

# %% [markdown]
# With the second syntax, as often in python, the access to objects/functions can be greatly simplified. For example,
# we can use "NDDataset" without a prefix instead of `scp.NDDataset` which is the first syntax) but there is always a
# risk of overwriting some variables or functions already present in the namespace.
# Therefore, the first syntax is generally highly recommended.
#
# Alternatively, you can also load only the onbjects and function required by your application:
#

# %%
from spectrochempy import NDDataset

nd = NDDataset()

# %% [markdown]
# If something goes wrong with during a cell execution,  a ``traceback`` is displayed.
#
# For instance, the object or method ``toto`` does not exist in the API, so an error (**ImportError**) is generated
# when trying to import this from the API.
#
# Here we catch the error with a conventional `try-except` structure

# %%
try:
    from spectrochempy import toto
except ImportError as e:
    scp.error_("OOPS, THAT'S AN IMPORT ERROR! : %s" % e)

# %% [markdown]
# The error will stop the execution if not catched.
#
# This is a basic behavior of python : on way to avoid. stoppping the execution without displaying a message is :

# %%
try:
    from spectrochempy import toto  # noqa: F811, F401
except Exception:
    pass

# %% [markdown]
# ## API Configuration
#
# Many options of the API can be set up

# %%
scp.set_loglevel(scp.INFO)

# %% [markdown]
# In the above cell, we have set the **log** level to display ``info`` messages, such as this one:

# %%
scp.info_('this is an info message!')
scp.debug_('this is a debug message!')

# %% [markdown]
# Only the info message is displayed, as expected.
#
# If we change it to ``DEBUG``, we should get the two messages

# %%
scp.set_loglevel(scp.DEBUG)

scp.info_('this is an info message!')
scp.debug_('this is a debug message!')

# %% [markdown]
# Let's now come back to a standard level of message for the rest of the Tutorial.

# %%
scp.set_loglevel(scp.WARNING)

scp.info_('this is an info message!')
scp.debug_('this is a debug message!')

# %% [markdown]
# Many other configuration items will be further described when necessary in the other chapters.

# %% [markdown]
# ## Units

# %% [markdown]
# The objets **ur**, **Quantity**  allows the manipulation of data with units, thanks to pint. (see [Units and
# Quantities](../units/index.ipynb))
#
# * **ur**: the unit registry
# * **Quantity**: a scalar or an array with some units

# %%
ur.cm / ur.s

# %%
x = Quantity(10., ur.cm / ur.s)
x * 2.

# %%
xa = Quantity(np.array((1, 2)), 'km')
xa[1] * 2.5

# %% [markdown]
# ## NDDataset, the main object

# %% [markdown]
# NDDataset is a python object, actually a container, which can represent most of your multidimensional spectroscopic
# data.
#
# For instance, in the following we read data from a series of FTIR experiments, provided  by the OMNIC software:

# %%
import os

nd = NDDataset.read_omnic(os.path.join('irdata', 'nh4y-activation.spg'))

# %% [markdown]
# Note that for this example, we use data stored in a ``test`` directory. For your own usage, you probably have to
# give the full pathname (see ... for the way to overcome this using `preferences` setting)

# %% [markdown]
# ### Display dataset information

# %% [markdown]
# Several ways are available to display the data we have jsut read and that are now stored in the dataset
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
# Let's plot first a 1D spectrum (for instance one row of nd)

# %%
row = nd[-1]
_ = row.plot()

# %% [markdown]
# or a column ...

# %%
col = nd[:, 3500.]  # note the indexing using wavenumber!
_ = col.plot_scatter()

# %% [markdown]
# 2D plots can be also generated as stacked plot

# %%
_ = nd.plot(method='stack')  # or nd.plot_stack()

# %% [markdown]
# or as a contour plot:

# %%
_ = nd.plot(method='map')  # or nd.plot_map()

# %% [markdown]
# Note that as we plot wavenumbers as abcissa, by convention the coordinates direction is reversed.
#
# This can be changed by using the keyword argument `reversed = False`.

# %% [markdown]
# ### Processing a dataset

# %% [markdown]
# Some arithmetic can be performed on such dataset. Here is an example where we subtract one reference spectrum to
# the whole nddataset that we have read above (`nd`).

# %% [markdown]
# Lets take, e.g., the last row as reference

# %%
ref = nd[0]
_ = ref.plot()

# %% [markdown]
# Now suppress this ref spectrum to all other spectra of the whole dataset

# %%
nds = nd - ref
_ = nds.plot(method='stack')

# %% [markdown]
# More details on available on available processing and analysis function will be given later in this user guide.

# %% [markdown]
# This was a short overview of the possibilities. To go further you can **Continue with ...**

# %% [markdown] nbsphinx-toctree={"maxdepth": 3}
# [Data Structures](../objects.rst)
