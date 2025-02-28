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
#       jupytext_version: 1.16.7
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
#     version: 3.10.11
#   widgets:
#     application/vnd.jupyter.widget-state+json:
#       state: {}
#       version_major: 2
#       version_minor: 0
# ---

# %% [markdown]
# # Introduction
#
# The **SpectroChemPy** project was developed to provide advanced tools for processing and
# analyzing spectroscopic data, initially for internal purposes within the
# [LCS (https://www.lcs.ensicaen.fr)](https://www.lcs.ensicaen.fr).
#
# The **SpectroChemPy** project is essentially a library written in
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
# ## About this user's guide
#
# The ``User's guide and tutorials`` section is designed to give you a quick overview of the main features of **SpectroChemPy**. It does not cover all
# features, but should help you to get started quickly, and to find your way around.
# For more details on the various features, check out the [Public API reference](../../reference/index.rst) section which gives
# [Gallery of Examples](../../gettingstarted/examples/index.rst)
#
# ## What to do if questions arise
#
# If, despite this documentation, which we're constantly trying to improve,
# you still have unresolved questions, don't hesitate to post your question on the
# [discussion forum](https://github.com/spectrochempy/spectrochempy/discussions).
# You can also post on [StackOverflow](https://stackoverflow.com),
# if you prefer, but don't forget to put "Spectrochempy" in the title
# (as there are no tags yet to enable us to find new questions easily.

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
# To get started, you can follow this link : [Jupyter Lab interface](interface.rst).

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
from spectrochempy import *  # noqa: F403

nd = NDDataset()  # noqa: F405

# %% [markdown]
# Alternatively, you can also load only the objects and function required by your
# application:

# %%
from spectrochempy import NDDataset

nd = NDDataset()


# %%
nd

# %% [markdown]
# ## Where to go next?

# %% [markdown]
# The ``User's guide & tutorial`` section is divided into several sections describing the main features of spectrochempy.
#
# * [Core objects](../objects/index.rst) : This is really the starting point for understanding how SpectroChemPy works. In fact, virtually all the functions offered by the API use one of these objects: `NDDataset` (the most important!) , `Project` or `Script`.
#
# * [Import & export](../importexport/index.rst) : This part shows how to import spectroscopic data and transform it into an NDDataset, the object on which processsing and analysis procedures will then be applied.
#
# * [Processing](../processing/index.rst) : This section explains how to prepare datasets for future analysis. By processing, we mean methods which generally return the same dataset transformed by mathematical operations (for example, this may be a `sqrt` or `log` operation, or a `fft` operation or many others), or part of a dataset transformed by a slicing or concatenation operation.
#
# * [Analysis](../analysis/index.rst) : This section presents the analysis methods implemented in SpectroChemPy. Generally speaking, analysis methods are any methods used to extract properties or characteristics from one or more NDDatasets. Analysis methods include statistical methods (mean, standard deviation, etc.), peaks, integration, etc.
# But perhaps the most important is the implementation of various chemometrics methods (PCA, PLS-R, EFA, MCR-ALS, ...), data fit methods, and baseline extraction.
#
# * [Plotting](../plotting/plotting.ipynb) : Finally, this section attempts to give the basics for using the plot methods included in SpectroChemPy.  These methods do not claim to cover all needs, and so it may be worthwhile for the user to learn how to use packages such as matplotlib, on which the methods described in this section are based.
