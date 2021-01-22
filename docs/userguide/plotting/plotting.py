# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.9.1
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
# ---

# %%
# flake8 : noqa

# %% [markdown]
# # Plotting

# %% [markdown]
# ## Load the API

# %% [markdown]
# First, before anything else, we import the spectrochempy API:

# %%
from spectrochempy import *

# %% [markdown]
# ## Loading the data

# %% [markdown]
# For sake of demonstration of the plotting capabilities of SpectroChemPy (based on [Matplotlib](
# https://matplotlib.org)), let's first import a NDDataset from a file and make some (optional) preparation of the
# data to display.

# %%
dataset = NDDataset.read('irdata/nh4y-activation.spg')

# %% [markdown]
# ## Preparing the data

# %% [markdown]
# We keep only the region that we want to display

# %%
dataset = dataset[:, 4000.:650.]

# %% [markdown]
# We change the y coordinated so that times start at 0 and put it in minutes

# %%
dataset.y -= dataset.y[0]
dataset.y.ito("minutes")
dataset.y.title = 'Time on stream'

# %% [markdown]
# We will also mask a region that we do not want to display

# %%
dataset[:, 1290.:920.] = MASKED

# %% [markdown]
# ## Selecting the output window

# %% [markdown]
# For the examples below, we use inline matplotlib figures (non interactive): this can be forced using the magic
# function before loading spectrochempy.:
# ```ipython3
# %matplotlib inline
# ```
# but it is also the default in `Jupyter lab` (so we don't really need to specify this). Note that when such magic
# function has been used, it is not possible to change the setting, except by resseting the notebook kernel.
#
# If one wants interactive displays (with selection, zooming, etc...) one can use:
# ```ipython3
#     %matplotlib widget
# ```
# However, this suffer (at least for us) some incompatibilities in `jupyter lab`... it is worth to try!
# If you can not get it working in `jupyter lab` and you need interactivity, you can use the following:
# ```ipython3
#     %matplotlib
# ```
# which has the effect of displaying the figures in independant windows using default matplotlib backend (e.g.,
# `Tk`), with all the interactivity of matplotlib.
#
# But you can explicitly request a different GUI backend:
# ```ipython3
#     %matplotlib qt
# ```

# %%
# # %matplotlib

# %% [markdown]
# ## Using Plotly (experimental)

# %% [markdown]
# For a better experience of interactivity inside a notebook, Spectro£ChemPy also offer the possibillity to use Plotly.
#
# See the dedicated tutorial: [here](...)

# %% [markdown]
# ## Default plotting

# %% [markdown]
# To plot the previously loaded dataset, it is very simple: we use the `plot` command (generic plot).
#
# The current NDDataset is 2D, a **stack plot** is displayed by default, with a **viridis** colormap.

# %%
prefs = dataset.preferences
prefs.reset()        # Reset to default plot preferences
_ = dataset.plot()

# %%
plt.rcParams

# %% [markdown]
# <div class="alert alert-block alert-info">
# <b>Tip: </b>
#
# Note, in the line above, that we used ` _ = ... `  syntax. This is to avoid any ouput but the plot from this statement.
# </div>

# %% [markdown]
# ## Changing the aspect of the plot

# %% [markdown]
# We can change the default plot configuration for this dataset (see below for an overview of the available
# configuration parameters.

# %%
prefs.figure.figsize = (6, 3)  # The default figsize is (6.8,4.4)
prefs.colorbar = True  # This add a color bar on a side
prefs.colormap = 'magma'  # The default colormap is viridis
prefs.axes.facecolor = '.95'  # Make the graph background colored in a ligth gray
prefs.axes.grid = True

_ = dataset.plot()

# %% [markdown]
# Note that, by default, <b>sans-serif</b> font are used for all text in the figure.
# But if you prefer, <b>serif</b>, or <b>monospace</b> font can be used instead:

# %%
#plt.style.use(['classic'])

prefs.font.family = 'serif'
# plt.rcParams['font.serif']=['Times New Roman',
#                          'Times',
#                          'Palatino',
#                          'DejaVu Serif',
#                          'Computer Modern Roman',
#                          'New Century Schoolbook',
#                          'serif']
dataset.plot();


# %%
prefs.font.family = 'monospace'
_ = dataset.plot()

# %%
plt.rcParams

# %% [markdown]
# ## Plotting 1D datasets

# %%
prefs.reset()
d1D = dataset[-1]  # select the last row of the previous 2D dataset
_ = d1D.plot(color='r')

# %% [markdown]
# ## Adding titles and annotations

# %% [markdown]
# The plot function return a reference to the subplot `ax` on which the data have been plotted.
# We can then use this reference to modify some element of the plot.
#
# For example, here we add a title and some annotations:

# %%
prefs.reset()
prefs.colorbar = False
prefs.colormap = 'terrain'
prefs.font.family = 'monospace'

ax = dataset.plot()
ax.grid(False)  # This temporary suppress the grid after the plot is done - not saved in prefs

# set title
title = ax.set_title('NH$_4$Y IR spectra during activation')
title.set_color('red')
title.set_fontstyle('italic')
title.set_fontsize(14)

# put some text
ax.text(1200., 1, 'Masked region\n (saturation)', rotation=90)

# put some fancy annotations (see matplotlib documentation to learn how to design this)
_ = ax.annotate('OH groups', xy=(3600., 1.25), xytext=(-10, -50), textcoords='offset points',
                arrowprops=dict(arrowstyle="fancy", color="0.5", shrinkB=5, connectionstyle="arc3,rad=-0.3", ), )

# %% [markdown]
# More information about annotation can be found in the [matplotlib documentation:  annotations](
# https://matplotlib.org/tutorials/text/annotations.html)

# %% [markdown]
# ## Changing the plot style using matpotlib style sheets

# %% [markdown]
#  The easiest way to to change the plot style may be to use presetted style such as those used in [matplotlib
#  styles](https://matplotlib.org/3.3.3/tutorials/introductory/customizing.html). This is directly included in the
#  preferences of SpectroChemPy

# %%
prefs.style = 'grayscale'
dataset.plot();

# %%
prefs.style = 'ggplot'
dataset.plot();

# %% [markdown]
# Other styles are :
# * paper , which create figure suitable for two columns article (fig width: 3.4 inch)
# * poster
# * talk

# %% [markdown]
# the styles can be combined:

# %%
prefs.reset()
prefs.style = 'grayscale', 'paper'
dataset.plot(colorbar=True);

# %% [markdown]
# Style specification can also be done directly in the plot method:

# %%
prefs.colormap = 'magma'
dataset.plot(style=['scpy', 'paper']);

# %% [markdown]
# To get a list of all available styles :

# %%
prefs.available_styles

# %% [markdown]
# Now to restore the default setting, you can use the reset function

# %%
prefs.reset()
dataset.plot();

# %% [markdown]
# ## Create your own style

# %% [markdown]
# If you want to create your own style for later use, you can use the command  `makestyle` (**warning**: you can not
# use `scpy`
# which is the READONLY default style).

# %%
prefs.makestyle('scpy')

# %% [markdown]
# If no name is provided a default name is used :`mydefault`

# %%
prefs.makestyle()

# %% [markdown]
# **Example:**
#

# %%
prefs.reset()
prefs.colorbar = True
prefs.colormap = 'jet'
prefs.font.family = 'monospace'
prefs.font.size = 14
prefs.axes.labelcolor = 'blue'
prefs.axes.grid = True
prefs.axes.grid_axis = 'x'

dataset.plot();

prefs.makestyle()

# %%
prefs.reset()
dataset.plot();  # plot with the default scpy style

# %%
prefs.style = 'mydefault'
dataset.plot();  # plot with our own style

# %% [markdown]
# ## Changing the type of plot

# %% [markdown]
# By default, plots are done in stack mode.
#
# If you like to have contour plot, you can use:

# %%
prefs.reset()

prefs.method_2D = 'map'  # this will change permanently the type of 2D plot
prefs.colormap = 'magma'
prefs.figure_figsize = (5, 3)
dataset.plot();

# %% [markdown]
# You can also, for an individual plot use specialised plot commands, such as plot_stack, plot_map, plot_waterfall,
# or plot_image:

# %%
prefs.axes_facecolor = 'white'
dataset.plot_image(colorbar=True);  # will use image_cmap preference!

# %%
prefs.reset()
dataset.plot_waterfall(figsize=(7, 4), y_reverse=True);

# %%
prefs.style = 'seaborn-paper'
dataset[3].plot(scatter=True, pen=False, me=30, ms=5)

# %% [markdown]
# ## Plotting several dataset on the same figure

# %% [markdown]
# We can plot several datasets on the same figure using the `clear` argument.

# %%
nspec = int(len(dataset) / 4)
ds1 = dataset[:nspec]  # split the dataset into too parts
ds2 = dataset[nspec:] - 2.  # add an ofset to the second part

ax1 = ds1.plot_stack()
ds2.plot_stack(ax=ax1, clear=False, zlim=(-2.5, 4));

# %% [markdown]
# For 1D datasets only, you can also use the `plot_multiple`mathod:

# %%
datasets = [dataset[0], dataset[10], dataset[20], dataset[50], dataset[53]]
labels = ['sample {}'.format(label) for label in ["S1", "S10", "S20", "S50", "S53"]]
prefs.reset()
prefs.axes.facecolor = '.99'
prefs.axes.grid = True
plot_multiple(method='scatter', me=10, datasets=datasets, labels=labels, legend='best');

# %% [markdown]
# ## Overview of the main configuration parameters

# %% [markdown]
# To display a dictionary of the current settings (**compared to those set by default**
# at API startup), you can simply type :

# %%
prefs

# %% [markdown]
# <div class="alert alert-block alert-warning">
#     <b>Warning</b> :
# Note that in the `dataset.preferences` dictionary (`prefs`), the parameters have a slightly different name, e.g.,
#     <b>figure_figsize</b> instead of <b>figure.fisize</b> which is the matplotlib syntax (In spectrochempy,
#     dot (`.`) cannot be used in paremeter name, and thus it is replaced by an underscore (`_`))
#
# Actually, in the jupyter notebook, or in scripts, both syntax can be used to read or write most of the preferences
# entries
#
#  </div>

# %% [markdown]
# To display the current values of **all parameters** correspondint to one group, e.g. `lines`, type:

# %%
prefs.lines

# %% [markdown]
# To display **help** on a single parameter, type:

# %%
prefs.help('lines_linewidth')

# %% [markdown]
# To view **all parameters**:

# %%
prefs.all()

# %% [markdown]
# ## A last graph for the road and fun...

# %%
prefs.font.family = 'fantasy'
import matplotlib.pyplot as plt
with plt.xkcd():
    # print(mpl.rcParams)
    prefs.lines.linewidth = 2
    ax = dataset[-1].plot(figsize=(7.5, 4))
    ax.text(2800., 1.5, "A XKCD plot! This is fun...")

# %% [markdown]
# If you get the error: "findfont: Font family ['Humor Sans'] not found. Falling back to DejaVu Sans.", it might be necessary to install the required font. For the above `Humor Sans` is required.
# * You can download it [here](https://github.com/shreyankg/xkcd-desktop/blob/master/Humor-Sans.ttf).
# * Install the `ttf` file into your system font's directory
#     - for windows: `C:/windows/fonts`
#     - osx: `~/Library/Fonts/`
#     - linux: `/usr/share/fonts/truetype`
# * Then you must delete the matplotlib font cache: Go to `<your-home-folder>/.matplotlib` and delete files such as:  `fontlist...`
#

# %% [markdown]
# import matplotlib.font_manager
# fm = matplotlib.font_manager.json_load(os.path.expanduser("~/.cache/matplotlib/fontlist-v330.json"))
# fm.findfont("serif", rebuild_if_missing=False)
# fm.findfont("serif", fontext="afm", rebuild_if_missing=False)
