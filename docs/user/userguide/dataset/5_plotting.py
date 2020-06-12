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
# # Plotting a NDDataset

# %% [markdown]
# <div class='alert alert-info'>
#
# **NOTE**: By default, all matplolib figures will be plotted **inline** in this notebook. 
# to change this behaviour, uncomment (which means: remove the #) the next line (which must be placed before importing the  ``spectrochempy.api`` library!
#
# </div>

# %%
from spectrochempy import *

# %% [markdown]
# Let's first import a NDDataset from a file:

# %%
import os
dataset = NDDataset.read_omnic(os.path.join('irdata', 'nh4y-activation.spg'))
print(dataset.description)

# %% [markdown]
# To plot a dataset, use the `plot` command (generic plot). As the current NDDataset is 2D, a contour plot is displayed by default.

# %%
_ = dataset.plot(colorbar=True) # plot the source.  

# %%
#import matplotlib as mpl
#mpl.rcParams

# %% [markdown]
# The plot function return a reference to the subplot on which the data have been plotted.
# We can then use this reference to modify some element of the plot.
#
# For example, here we add a title:

# %%
title = dataset.ax.set_title('NH$_4$Y IR spectra during activation')
title.set_color('magenta')
title.set_fontstyle('italic')
title.set_fontsize(16)

# %% [markdown]
# Note that by default, *sans-serif* font are used for all text in the figure. 
#
# But if you prefer, *serif* font can be used instead. The easiest way to do this is to change the plot style:

# %%
_ = dataset.plot(style='serif')

# %% [markdown]
# Other styles are :
# * paper , which create figure suitable for two columns article (fig width: 3.4 inch)
# * poster
# * talk
# * grayscale

# %%
_ = dataset.plot(style='paper', colorbar=True)

# %% [markdown]
# To get a list of all available styles :

# %%
available_styles()

# %% [markdown]
# these styles can be combined

# %%
_ = dataset.plot(style=['sans','paper','grayscale'], colorbar=True)

# %% [markdown]
# New styles can also be created, using a simple dictionary:

# %%
mystyle={'image.cmap':'magma', 
         'font.size':10, 
         'font.weight':'bold', 
         'axes.grid':True}
#TODO: store these styles for further use
_ = dataset.plot(style=mystyle)

# %% [markdown]
# To display all entry for definig plot style, uncomment the next line:

# %%
#import matplotlib as mpl
#mpl.rcParams

# %% [markdown]
# ## Changing axis
# The `y` axis with timestamp in the above plots is not very informative, lets rescale it in hours and change the origin. 

# %%
dataset.y -= dataset.y[0]                # change origin
dataset.y.title = u'Aquisition time'    # change the title (default axis label)
dataset.y.to('hour')                    # change unit base
_ = dataset.plot()

# %% [markdown]
# By default, plots are done in contour mode.
#
# If you like to have stacked plot, you can use:

# %%
_ = dataset.plot(method='stack', style='sans', colorbar=False)

# %% [markdown]
# We can change or add labels to axes after creation of the dataset  #TODO

# %% [markdown]
# We can plot several datasets on the same figure

# %%
dataset.plot(method='stack', style='sans', colorbar=False)

so = dataset.copy()
so += 2

_ = so.plot(method='stack', colormap='jet', data_only=True, clear=False)
so.ax.set_ylim(-1,9)
