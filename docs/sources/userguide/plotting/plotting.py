# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
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

# %% [markdown]
# # Plotting
#
# This section shows the main plotting capabilities of SpectroChemPy. Most of them are based on [Matplotlib](
# https://matplotlib.org), one of the most used plotting library for Python, and its
# [pyplot](https://matplotlib.org/stable/tutorials/introductory/pyplot.html) interface. While not mandatory, to follow
# this tutorial, some familiarity with this library can help, and we recommend a brief look at some
# [matplotlib tutorials](https://matplotlib.org/stable/tutorials/index.html) as well.
#
# Note that in the near future, SpectroChemPy should also offer the possibility to use [Plotly](https://plotly.com/)
# for a better interactivity inside a notebook.
#
# Finally, some commands and objects used here are described in-depth in the sections related to
# [import](../importexport/import.rst) and [slicing](../processing/slicing.rst) of NDDatasets and the *
# [NDDatasets](../objects/dataset/dataset.ipynb) themselves.
# %% [markdown]
# ## Load the API
# First, before anything else, we import the spectrochempy API:

# %%
import spectrochempy as scp

# %% [markdown]
# ## Loading the data
# For sake of demonstration we import a NDDataset consisting in infrared spectra from an omnic .spg file
# and make some (optional) preparation of the data to display
# (see also [Import IR Data](../importexport/importIR.rst)).

# %%
dataset = scp.read("irdata/nh4y-activation.spg")

# %% [markdown]
# ## Preparing the data

# %% [markdown]
#


# %%
dataset = dataset[:, 4000.0:650.0]  # We keep only the region that we want to display

# %% [markdown]
# We change the y coordinated so that times start at 0, put it in minutes and change its title/

# %%
dataset.y -= dataset.y[0]
dataset.y.ito("minutes")
dataset.y.title = "relative time on stream"

# %% [markdown]
# We also mask a region that we do not want to display

# %%
dataset[:, 1290.0:920.0] = scp.MASKED

# %% [markdown]
# ## Selecting the output window

# %% [markdown]
# For the examples below, we use inline matplotlib figures (non-interactive): this can be forced using the magic
# function before loading spectrochempy.:
# ```ipython3
# %matplotlib inline
# ```
# but it is also the default in `Jupyter lab` (so we don't really need to specify this). Note that when such magic
# function has been used, it is not possible to change the setting, except by resetting the notebook kernel.
#
# If one wants interactive displays (with selection, zooming, etc...) one can use:
# ```ipython3
#     %matplotlib widget
# ```
# However, this suffers (at least for us) some incompatibilities in `jupyter lab` ...
# it is worth to try!
# If you can not get it working in `jupyter lab` and you need interactivity, you can
# use the following:
# ```ipython3
#     %matplotlib
# ```
# which has the effect of displaying the figures in independent windows using default
# matplotlib backend (e.g.,
# `Tk` ), with all the interactivity of matplotlib.
#
# But you can explicitly request a different GUI backend:
# ```ipython3
#     %matplotlib qt
# ```

# %%
# %matplotlib inline

# %% [markdown]
# ## Default plotting

# %% [markdown]
# To plot the previously loaded dataset, it is very simple: we use the `plot` command (generic plot).
#
# As the current NDDataset is 2D, a **stack plot** is displayed by default, with a **viridis** colormap.

# %%
dataset.plot()

# %% [markdown]
# Note that the `plot()` method uses some of NDDataset attributes: the `NDDataset.x` coordinate `data` (here the
# wavenumber values), `name` (here 'wavenumbers'), `units` (here 'cm-1') as well as the `NDDataset.title`
# (here 'absorbance') and `NDDataset.units (here 'absorbance').
# %% [markdown]
# ## Changing the aspect of the plot

# %% [markdown]
# ### Change the `NDDataset.preferences`
# We can change the default plot configuration for this dataset by changing its `preferences' attributes
# (see at the end of this tutorial  for an overview of all the available parameters).

# %%
prefs = dataset.preferences  # we will use prefs instead of dataset.preference
prefs.figure.figsize = (6, 3)  # The default figsize is (6.8,4.4)
prefs.colorbar = True  # This add a color bar on a side
prefs.colormap = "magma"  # The default colormap is viridis
prefs.axes.facecolor = ".95"  # Make the graph background colored in a light gray
prefs.axes.grid = True

dataset.plot()


# %% [markdown]
# The colormap can also be changed by setting `cmap` in the arguments.
# If you prefer not using colormap, `cmap=None` should be used. For instance:

# %%
dataset.plot(cmap=None, colorbar=False)

# %% [markdown]
# Note that, by default, **sans-serif** font are used for all text in the figure.
# But if you prefer, **serif**, or *monospace* font can be used instead. For instance:

# %%
prefs.font.family = "monospace"
dataset.plot()

# %% [markdown]
# Once changed, the `NDDataset.preferences` attributes will be used for the subsequent plots, but can be reset to the
# initial defaults anytime using the `NDDataset.preferences.reset()` method. For instance:

# %%
print(f"font before reset: {prefs.font.family}")
prefs.reset()
print(f"font after reset: {prefs.font.family}")

# %% [markdown]
# It is also possible to change a parameter for a single plot without changing the `preferences` attribute by passing
# it as an argument of the `plot()`method. For instance, as in matplotlib, the default colormap is `viridis':

# %%
prefs.colormap

# %% [markdown]
# but 'magma' can be passed to the `plot()` method:

# %%
dataset.plot(colormap="magma")

# %% [markdown]
# while the `preferences.colormap` is still set to `viridis':

# %%
prefs.colormap

# %% [markdown]
# and will be used by default for the next plots:

# %%
dataset.plot()

# %% [markdown]
# ## Adding titles and annotations

# %% [markdown]
# The plot function return a reference to the subplot `ax` object on which the data have been plotted.
# We can then use this reference to modify some element of the plot.
#
# For example, here we add a title and some annotations:

# %%
prefs.reset()
prefs.colorbar = False
prefs.colormap = "terrain"
prefs.font.family = "monospace"

ax = dataset.plot()
ax.grid(
    False
)  # This temporarily suppress the grid after the plot is done but is not saved in prefs

# set title
title = ax.set_title("NH$_4$Y IR spectra during activation")
title.set_color("red")
title.set_fontstyle("italic")
title.set_fontsize(14)

# put some text
ax.text(1200.0, 1, "Masked region\n (saturation)", rotation=90)

# put some fancy annotations (see matplotlib documentation to learn how to design this)
_ = ax.annotate(
    "OH groups",
    xy=(3600.0, 1.25),
    xytext=(-10, -50),
    textcoords="offset points",
    arrowprops={
        "arrowstyle": "fancy",
        "color": "0.5",
        "shrinkB": 5,
        "connectionstyle": "arc3,rad=-0.3",
    },
)

# %% [markdown]
# More information about annotation can be found in the [matplotlib documentation:  annotations](
# https://matplotlib.org/stable/tutorials/text/annotations.html)

# %% [markdown]
# ## Changing the plot style using matplotlib style sheets

# %% [markdown]
#  The easiest way to change the plot style may be to use pre-defined styles such as those used in [matplotlib
#  styles](https://matplotlib.org/stable/tutorials/introductory/customizing.html). This is directly included in the
#  preferences of SpectroChemPy

# %%
prefs.style = "grayscale"
dataset.plot()

# %%
prefs.style = "ggplot"
dataset.plot()

# %% [markdown]
# Other styles are :
# * paper , which create figure suitable for two columns article (fig width: 3.4 inch)
# * poster
# * talk

# %% [markdown]
# the styles can be combined, so you can have a style sheet that customizes
# colors and a separate style sheet that alters element sizes for presentations:

# %%
prefs.reset()
prefs.style = "grayscale", "paper"
dataset.plot(colorbar=True)

# %% [markdown]
# As previously, style specification can also be done directly in the plot method without
# affecting the `preferences' attribute.

# %%
prefs.colormap = "magma"
dataset.plot(style=["scpy", "paper"])

# %% [markdown]
# To get a list of all available styles :

# %%
prefs.available_styles

# %% [markdown]
# Again, to restore the default setting, you can use the reset function

# %%
prefs.reset()
dataset.plot()

# %% [markdown]
# ## Create your own style
#
# If you want to create your own style for later use, you can use the command  `makestyle` (**warning**: you can not
# use `scpy` which is the READONLY default style:

# %%
prefs.makestyle("scpy")

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
prefs.colormap = "jet"
prefs.font.family = "monospace"
prefs.font.size = 14
prefs.axes.labelcolor = "blue"
prefs.axes.grid = True
prefs.axes.grid_axis = "x"

dataset.plot()

prefs.makestyle()

# %%
prefs.reset()
dataset.plot()  # plot with the default scpy style

# %%
prefs.style = "mydefault"
dataset.plot()  # plot with our own style

# %% [markdown]
# ## Changing the type of plot

# %% [markdown]
# By default, plots of 2D datasets are done in 'stack' mode. Other available modes are 'map', 'image', 'surface' and
# 'waterfall'.
#
# The default can be changed permanently by setting the variable `pref.method_2D` to one of these alternative modes,
# for instance if you like to have contour plot, you can use:

# %%
prefs.reset()

prefs.method_2D = "map"  # this will change permanently the type of 2D plot
prefs.colormap = "magma"
prefs.figure_figsize = (5, 3)
dataset.plot()

# %% [markdown]
# You can also, for an individual plot use specialised plot commands, such as `plot_stack()` , `plot_map()` ,
# `plot_waterfall()` , `plot_surface()` or `plot_image()` , or equivalently the generic `plot` function with
# the `method` parameter, i.e., `plot(method='stack')` , `plot(method='map')` , etc...
#
# These modes are illustrated below:
# %%
prefs.axes_facecolor = "white"
dataset.plot_image(colorbar=True)  # will use image_cmap preference!

# %% [markdown]
# Here we use the generic `plot()` with the `method' argument and we change the image_cmap:

# %%
dataset.plot(method="image", image_cmap="jet", colorbar=True)

# %% [markdown]
# The colormap normalization can be changed using the `norm` parameter, as illustrated below,
# for a centered colomap:

# %%
import matplotlib as mpl

norm = mpl.colors.CenteredNorm()
dataset.plot(method="image", image_cmap="jet", colorbar=True, norm=norm)

# %% [markdown]
# or below for a log scale (more information about colormap normalization can be found
# [here](https://matplotlib.org/stable/users/explain/colors/colormapnorms.html)).

# %%
norm = mpl.colors.LogNorm(vmin=0.1, vmax=4.0)
dataset.plot(method="image", image_cmap="jet", colorbar=True, norm=norm)

# %% [markdown]
# Below an example of a waterfall plot:

# %%
prefs.reset()
dataset.plot_waterfall(figsize=(7, 4), y_reverse=True)

# %% [markdown]
# And finally an example of a surface plot:

# %%
prefs.reset()
dataset.plot_surface(figsize=(7, 7), linewidth=0, y_reverse=True, autolayout=False)
# %% [markdown]
# ## Plotting 1D datasets

# %%
prefs.reset()
d1D = dataset[-1]  # select the last row of the previous 2D dataset
d1D.plot(color="r")

# %%
prefs.style = "seaborn-v0_8-paper"
dataset[3].plot(scatter=True, pen=False, me=30, ms=5)

# %% [markdown]
# ## Plotting several dataset on the same figure

# %% [markdown]
# We can plot several datasets on the same figure using the `clear` argument.

# %%
nspec = int(len(dataset) / 4)
ds1 = dataset[:nspec]  # split the dataset into too parts
ds2 = dataset[nspec:] - 2.0  # add an offset to the second part

ax1 = ds1.plot_stack()
ds2.plot_stack(ax=ax1, clear=False, zlim=(-2.5, 4))

# %% [markdown]
# For 1D datasets only, you can also use the `plot_multiple`method:

# %%
datasets = [dataset[0], dataset[10], dataset[20], dataset[50], dataset[53]]
labels = [f"sample {label}" for label in ["S1", "S10", "S20", "S50", "S53"]]
prefs.reset()
prefs.axes.facecolor = ".99"
prefs.axes.grid = True
scp.plot_multiple(
    method="scatter", me=10, datasets=datasets, labels=labels, legend="best"
)

# %% [markdown]
# ## Overview of the main configuration parameters

# %% [markdown]
# To display a dictionary of the current settings (**compared to those set by default**
# at API startup), you can simply type :

# %%
prefs

# %% [markdown]
# **Warning**: Note that with respect to matplotlib,the parameters in the `dataset.preferences` dictionary
# have a slightly different name, e.g. `figure_figsize` (SpectroChemPy) instead of `figure.figsize` (matplotlib syntax)
# (this is because in SpectroChemPy, dot (` .` ) cannot be used in parameter name,
# and thus it is replaced by an underscore (`_` ))
#

# %% [markdown]
# To display the current values of **all parameters** corresponding to one group, e.g. `lines` , type:

# %%
prefs.lines

# %% [markdown]
# To display **help** on a single parameter, type:

# %%
prefs.help("lines_linewidth")

# %% [markdown]
# To view **all parameters**:

# %%
prefs.all()
