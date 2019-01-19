# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Overview

# %% [markdown]
# ## Getting Started
#

# %% [markdown]
# ### Using Jupyter Notebook

# %% [markdown]
# Assuming Jupyter is installed (*i.e.*, you have followed the [Spectrochempy installation procedure](../../../gettingstarted/install)), go to the
# |scpy| folder. From this location, open a
# terminal and type something like:
#
# ![launch](../images/launch_jupyter.png)
#
# Your default explorer is now launched.
#
# ![launched](../images/jupyter_home.png)
#
# You can now navigate to the Tutorial notebooks in [tutorial]  or the notebooks used for this user guide.
#
# Click on the the fist notebook : **introduction.ipynb**
#
# ![introd](../images/jupyter_intro.png)

# %% [markdown]
# ### Using Jupyter Lab

# %% [raw]
# .. todo:

# %% [markdown]
# ### Loading the API

# %% [markdown]
# <div class="alert alert-info">
#
# **Note:** We assume the spectrochemistry package has been properly installed - if not please go to ``install``
#
# </div>

# %% [markdown]
# Before using the package, we need to load the **API (Application Programming Interface)**
#
# The simplest way is to import all the objects and methods at once into your python namespace.

# %%
from spectrochempy import *

# %% [markdown]
# but you can also import method only when it is needed.
#
# For instances, one object very usefull in the following will be a nd-dataset to contain some data. Instead of issuing the previous command, one can do:

# %%
from spectrochempy import NDDataset
mydataset = NDDataset()

# %% [markdown]
# In the second line we have defined a new empty instance of **NDDataset**, wich will be further use to load our data.

# %% [markdown]
# Another way to proceed is to not mix the API namespace with your normal python. 
#
# In this case, you can simply do:
#

# %%
import spectrochempy as sc
mydataset = sc.NDDataset()

# %% [markdown]
# As such, the above command ``from spectrochempy import *``, lead to import of several objects and methods in the namespace.
#
# To get a list of all available methods or objects, type the following command (*remove the leading #, first*):

# %%
# APIref


# %% [markdown]
# If something goes wrong with during a cell execution,  a ``traceback`` is displayed.
#
# For instance, the object or method ``toto`` does not exist in the API, so an error (**ImportError**) is generated when trying to import this from the API. Here we catch the error with a try except structure

# %%
try:
    from spectrochempy import toto
except ImportError as e:
    log.error("OOPS, THAT'S AN IMPORT ERROR! : %s"%e)

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
log.info('this is an info message!')
log.debug('this is a debug message!')

# %% [markdown]
# Only the info message is displayed, as expected.
#
# If we change it to ``DEBUG``, we should get the two messages

# %%
set_loglevel(DEBUG)

log.info('this is an info message!')
log.debug('this is a debug message!')

# %% [markdown]
# Let's now come back to a standard level of message for the rest of the Tutorial.

# %%
set_loglevel(WARNING)

log.info('this is an info message!')
log.debug('this is a debug message!')
log.warning('this is a warning message!')

# %% [markdown]
# ### Access to scientific libraries

# %% [markdown]
# Several libraries are imported with **SpectroChemPy** (so you don't need to re-import them):
#
# - **np** :  This is actually the **`numpy`** library, to perform numerical calculation on nD arrays. 
# - **plt** : This is the **`matplotlib`** library, to allow plotting data 
#
# Optionally, **scipy** and **sympy** can be available, if **SpectroChempy** can find these libraries installed on your system.

# %%
x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)
plt.figure(figsize=(5,2.5))
p, = plt.plot(x,y)
p.set_linewidth(2)
p.set_color('red')
#plt.show()

# %% [markdown]
# ### Units, uncertainties and measurements

# %% [markdown]
# The objets **ur**, **Quantity** and **Measurement**, allows the manipulation of data with units and uncertainties. (see tutorial-1-units-uncertainties)
#
# * **ur**: the unit registry
# * **Quantity**: a scalar or an array with some units
# * **Measurement**: a scalar with units and unertainty

# %%
ur.cm / ur.s

# %%
x = Quantity(10., 'km')
x * 2.

# %%
xa = Quantity(np.array((1,2)), 'km')
xa[1] * 2.5

# %%
y = Measurement(10, .1, 'km')
y  

# %%
z = x.plus_minus(.01)
z

# %% [markdown]
# **Warning**, this is not available for numpy.array based quantities !  For this we will use NDDataset described 
# below

# %%
try:
    za = xa.plus_minus(.01)
except AttributeError as e:
    log.error(e)


# %% [markdown]
# ## NDDataset, the main object

# %% [markdown]
# NDDataset is a python object, actually a container, which can represent most of your multidimensional spectroscopic data.
#
# For instance, in the following we read data from a series of FTIR experiments, provided  by the OMNIC software:

# %%
nd = NDDataset.read_omnic(os.path.join('irdata', 'NH4Y-activation.SPG'))

# %% [markdown]
# Note that for this example, we use data stored in a ``test`` directory. For your own usage, you probably have to give the full pathname (see ... for the way to overcome this using `preferences` setting)

# %% [markdown]
# ### Display dataset information

# %% [markdown]
# Several ways are available to display the data we have jsut read and that are now stored in the ``source`` dataset 
#
# * **Printing** them, using the print function of python to get a text version of the `source` information

# %%
print(nd)

# %% [markdown]
# * **Displaying html**, inside a jupyter notebook, by just typing the name of the dataset (must be the last instruction of a cell, however!)

# %%
nd

# %% [markdown]
# ### Plotting a dataset
#
# Let's plot first a 1D spectrum (for instance one row of nd)

# %%
row = nd[-1]
_ = row.plot()

_ = nd.plot(method='stack') # or nd.plot_stack()

# %% [markdown]
# or as a contour plot: 

# %%
_ = nd.plot(method='map')

# %% [markdown]
# Note that as we plot wavenumbers as abcissa, by convention the coordinates dirtection is reversed.
#
# This can be changed by using the keyword argument `reversed` = `False`.

# %% [markdown]
# ### Processing a dataset

# %% [markdown]
# Some arithmetic can be performed on such dataset. Here is an example where we subtract one reference spectrum to the whole nddataset that we have read above (`nd`).

# %% [markdown]
# Lets take, e.g., the last row as reference

# %%
ref = nd[-1]
_ = ref.plot() 

# %% [markdown]
# Now suppress this ref spectrum to all other spectra of the whole dataset

# %%
nds = nd - ref  
_ = nds.plot(method='stack')

# %% [markdown]
# More details on available on available processing and analysis function will be given later in this user guide.
#

# %% [code]{"hide_input": true}
# show() # Note : show all plots (only required if the notebook is exported as a script, to make all plots visibles)

# %% [markdown]
# ## File selector widget

# %% [markdown]
# A widget is provided to help with the selection of file names or directory.  

# %%
path = general_preferences.datadir
fs = FileSelector(path = path, filters=['spg','scp'])
fs

# %% [markdown]
# After validation of the selection, one can read the path and name of the selected files. 

# %%
fs.value, fs.path, fs.fullpath
