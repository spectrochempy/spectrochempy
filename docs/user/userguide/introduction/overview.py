# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Getting Started
#
# Currently **SpectroChemPy** can be used as a library for python script. 
# Indeed, at this time, there is no particular graphical user interface (GUI) available. This may change in the future
# but for now we prefer to recommend the use of `Jupyter Notebook` document (see https://jupyter.org/).

# %% [markdown]
# <div class="alert alert-info">
#
# **Note:** We assume the spectrochemistry package has been properly installed - if not please go to [Spectrochempy installation procedure](../../../gettingstarted/install/index.rst)
#
# </div>

# %% [markdown]
# ## Using `Jupyter Notebook`
#
# Assuming `Jupyter` is installed (*i.e.*, you have followed the [Spectrochempy installation procedure](../../../gettingstarted/install/index.rst)), go to your favorite user document folder (*e.g.,* `$HOME/workspace/) or any other folder you want to use to store your development).
#
#     $ cd ~/workspace
#
# From this location, type the following command:
#
#     $ jupyter notebook
#
# Your default Web explorer is now launched.
#
# <img src="../images/jupyter_home.png" alt="launched" width="690"/>
#
# Here you can create a new notebook and follow our [UserGuide](../index.rst) or [Tutorials](../../tutorials/index.rst).

# %% [markdown]
# ## Using `Jupyter Lab`
#
# Using the new application `jupyter lab` is very similar to the `jupyter notebook`. 
#
# We recommend to use this application as it is quite intuitive to use, and advantageously replace the traditional `jupyter notebook` application.
#
# To get more information on its usage, one can go to: 
# [https://jupyterlab.readthedocs.io](https://jupyterlab.readthedocs.io/en/stable/getting_started/overview.html)
#
# From you worspace folder or any other location on your computer, open a terminal and issue commands as follows:
#
#     $ jupyter lab
#
# The home window should looks similar to this:
#
# <img src="../images/jupyter_lab_home.png" alt="launch" width="690"/>
#
# From there, it is quite easy to create new notebooks or to navigate to already existing ones.
#
# Then follow our [UserGuide](../index.rst) or [Tutorials](../../tutorials/index.rst) to get strated with
# **SpectroChemPy**

# %% [markdown]
# ## Loading the API

# %% [markdown]
# Before using the package, we need to load the **API (Application Programming Interface)**
#
# The simplest way is to import all the objects and methods at once into your python namespace. The loading step may
# take several seconds due to the large number of method to import in the API namespace.

# %%
from spectrochempy import *

# %% [markdown]
# This is a simple way to do this, but you can also import method only when it is needed.
#
# For instances, one object very usefull in the following will be a nd-dataset to contain some data. Instead of
# issuing the previous command, one can do:

# %%
from spectrochempy import NDDataset

mydataset = NDDataset()

# %% [markdown]
# In the second line we have defined a new empty instance of **NDDataset**, which will be further use to load our data.

# %% [markdown]
# Another way to proceed is to not mix the API namespace with your normal python.
#
# In this case, you can simply do:
#

# %%
import spectrochempy as scp

mydataset = scp.NDDataset()

# %% [markdown]
# As such, the above command ``from spectrochempy import *``, lead to import of several objects and methods in the
# namespace.
#
# To get a list of all available methods or objects, type the following command (*remove the leading #, first*):

# %%
#APIref


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
    error_("OOPS, THAT'S AN IMPORT ERROR! : %s" % e)

# %% [markdown]
# The error will stop the execution if not catched.
#
# This is a basic behavior of python : on way to avoid. stoppping the execution without displaying a message is :

# %%
try:
    from spectrochempy import toto
except:
    pass

# %% [markdown]
# ### Configuration
#
# Many options of the API can be set up

# %%
set_loglevel(INFO)

# %% [markdown]
# In the above cell, we have set the **log** level to display ``info`` messages, such as this one:

# %%
info_('this is an info message!')
debug_('this is a debug message!')

# %% [markdown]
# Only the info message is displayed, as expected.
#
# If we change it to ``DEBUG``, we should get the two messages

# %%
set_loglevel(DEBUG)

info_('this is an info message!')
debug_('this is a debug message!')

# %% [markdown]
# Let's now come back to a standard level of message for the rest of the Tutorial.

# %%
set_loglevel(WARNING)

info_('this is an info message!')
debug_('this is a debug message!')
warning_('this is a warning message!')

# %% [markdown]
# ### Access to scientific libraries

# %% [markdown]
# Several libraries are imported with **SpectroChemPy** (so you don't need to re-import them):
#
# - **np** :  This is actually the **`numpy`** library, to perform numerical calculation on nD arrays.
# - **plt** : This is the **`matplotlib`** library, to allow plotting data
#
# Optionally, **scipy** and **sympy** can be available, if **SpectroChempy** can find these libraries installed on
# your system.

# %%
# %matplotlib inline
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)
plt.figure(figsize=(5, 2.5))
p, = plt.plot(x, y)
p.set_linewidth(2)
p.set_color('red')

# %% [markdown]
# ### Units

# %% [markdown]
# The objets **ur**, **Quantity**  allows the manipulation of data with units, thanks to pint. (see tutorial-1-units)
#
# * **ur**: the unit registry
# * **Quantity**: a scalar or an array with some units

# %%
ur.cm / ur.s

# %%
x = Quantity(10., 'km')
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
print_(nd)

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
#
