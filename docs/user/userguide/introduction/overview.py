# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
# # Overview

# %% [markdown]
# ## Loading the API

# %% [markdown]
#  
# <div class="alert alert-info">
#
# **Note:** We assume the spectrochemistry package has been properly installed - if not please go to the &#9655; __[Installation Guide](https://www.spectrochempy.fr/gettingstarted/install.html)__
#
# </div>

# %% [markdown]
# Before using the package, we need to load the **API**.
#
# **API** stands for **Application Programming Interface**, that is, in the case of **SpectroChemPy**, a set of methods, functions and various tools for building scripts used for solving spectroscopic analysis problems. 
#
# The simplest way is to import all the objects and methods at once into your python namespace, using a wildcard import (*****).

# %%
from spectrochempy import *

# %% [markdown]
# However, you can also import method only when it is needed.
#
# For instances, one object very usefull in the following will be a multidimensional dataset to contain some data. Instead of issuing the previous command, one can do:

# %%
from spectrochempy import NDDataset
mydataset = NDDataset()

# %% [markdown]
# In the second line we have defined a new empty instance of **NDDataset**, which will be further use to load our data.

# %% [markdown]
# Another way to proceed is to not mix the API namespace with your normal python namespace (this is slightly less straigthforward to use, but it may prevent problems with namespace collisions - *i.e.,* when two commands or variables have the same name in different librairies. 
#
# In this case, you can simply do:
#

# %%
import spectrochempy as sc
mydataset = sc.NDDataset()

# %% [markdown]
# As such, the above command ``from spectrochempy import *``, lead to import of several objects and methods in the namespace.

# %% [markdown]
# If something goes wrong during a cell execution,  a `traceback` is displayed.
#
# For instance, the object or method `toto` does not exist in the API, so an error (**ImportError**) is generated when trying to import this from the API. 
#
# Here we catch the error with a try except classical python structure

# %%
try:
    from spectrochempy import toto
except ImportError as e:
    print_("OOPS, I", e)

# %% [markdown]
# The error will stop the notebook execution if not catched. 
#
# This is a basic behavior of python: The way to avoid stoppping the execution without displaying a message is :

# %%
try:
    from spectrochempy import toto
except:
    pass

# %% [markdown]
# ## Configuration

# %% [markdown]
# Many options of the API can be set up

# %%
set_loglevel(INFO)

# %% [markdown]
# In the above cell, we have set the **logger** level to display ``info`` messages, such as this one:

# %%
error_('this is an error message!')
warning_('this is a warning message!')
info_('this is an info message!')
debug_('this is a debug message!')

# %% [markdown]
# Only the `info` message is displayed as well as the `warning` and `error` messages, as expected.
#
# If we change it to ``DEBUG``, we should get also the `debug` message (a a somewhat more verbose output).

# %%
set_loglevel(DEBUG)

error_('this is an error message!')
warning_('this is a warning message!')
info_('this is an info message!')
debug_('this is a debug message!')

# %% [markdown]
# Let's now come back to a standard level of message for the rest of the Tutorial.

# %%
set_loglevel(WARNING)

error_('this is an error message!')
info_('this is an info message!')
debug_('this is a debug message!')
warning_('this is a warning message!')

# %% [markdown]
# ## Access to scientific libraries

# %% [markdown]
# Several libraries are imported with **SpectroChemPy**:
#
# - **np**:  This is actually the `numpy` library, to perform numerical calculation on nD arrays. 
# - **plt**: This is the `matplotlib` library, to allow plotting data 
#
# Optionally, **scipy** and **sympy** can be available, if **SpectroChempy** can find these libraries installed on your system.
#
# <div class="alert warning">
#
# **Note:** The `numpy` and other libraries above cited are imported only if the wildcard import (*) is done. If it is not the case then you must import them manually (if you need them of course): 
#     
# - **import numpy as np** 
# - **import matplotlib.pyplot as plt**
#     
# </div>
#

# %%
from spectrochempy import *

x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)
plt.figure(figsize=(5,2.5))
p, = plt.plot(x,y)
p.set_linewidth(2)
p.set_color('red')

# %% [markdown]
# ## NDDataset, the main object

# %% [markdown]
# `NDDataset` is a python object, actually a container, which can represent most of your multidimensional spectroscopic data. A limitation however is that a dataset must contains homogeneous data, with the same units, same coordinates, etc. 
#
# If you need a more complex data structure, where heterogeneous data can be stored, you can use the `NDPanel`  object (see [NDPanel description](https://www.spectrochempy.fr/user/userguide/dataset/2_ndpanel.html)).
#
# As an example of `NDDataset`, in the following we read data from a series of FTIR experiments, provided by the OMNIC software:

# %%
nd = NDDataset.read_omnic(os.path.join('irdata', 'nh4y-activation.spg'))

# %% [markdown]
# Note that for this example, we use data stored in a ``test`` directory. For your own usage, you probably have to give the full pathname (see ... for the way to overcome this using `preferences` setting)

# %% [markdown]
# ### Display dataset information

# %% [markdown]
# Several ways are available to display the data we have jsut read and that are now stored in the dataset 
#
# * **Printing** them, using the print function of python to get a short text version of the dataset information.

# %%
print(nd)

# %% [markdown]
# A much longer (and colored) information text can be obtained using the spectrochempy provided `print_` function.

# %%
print_(nd)

# %% [markdown]
# * **Displaying html**, inside a jupyter notebook, by just typing the name of the dataset (must be the last instruction of a cell, however!)

# %%
nd

# %% [markdown]
# ### Plotting a dataset
#
# Let's plot first a 1D spectrum (for instance one row of nd)

# %%
import warnings

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
_ = nd.plot(method='stack') # or nd.plot_stack()

# %% [markdown]
# or as a contour plot: 

# %%
_ = nd.plot(method='map') # or nd.plot_map()

# %% [markdown]
# Note that as we plot wavenumbers as abcissa, by convention the coordinates direction is reversed.
#
# This can be changed by using the keyword argument `reversed = False`.

# %% [markdown]
# ### Processing a dataset

# %% [markdown]
# Some arithmetic can be performed on such dataset. Here is an example where we subtract one reference spectrum to the whole nddataset that we have read above (`nd`).

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
