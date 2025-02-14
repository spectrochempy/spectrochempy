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
#   nbsphinx:
#     orphan: true
#   widgets:
#     application/vnd.jupyter.widget-state+json:
#       state: {}
#       version_major: 2
#       version_minor: 0
# ---

# %% [markdown]
# # Starting Jupyter lab
#
# Currently, **SpectroChemPy** can be used as a library for python scripts.
#
# For ease of use, we recommend using the
# __[JupyterLab](https://jupyterlab.readthedocs.io/en/stable/getting_started/overview.html)__
# application or for those who are more comfortable programming,
# writing python scripts in a development environment such as
# __[PyCharm](https://www.jetbrains.com/fr-fr/pycharm/)__, __[VS
# Code](https://code.visualstudio.com)__ or __[Spyder](https://www.spyder-ide.org)__.
#
# To launch `Jupyter Lab` , open a terminal and follow the steps below:
#
# * Go to your favorite user document folder (*e.g.,* `$HOME/workspace` or
# any other folder you want to use to store your work).
# ```bash
# $ cd $HOME/workspace
# ```
# * Then type the following command:
# ```bash
# $ jupyter lab
# ```
#
# Your default browser should now be open, and the window should look like
# this:
#
# <img src='images/launch_lab.png' />
#
# From there, it is quite easy to create notebooks or to navigate to
# already existing ones.
#
# ## Create a new Jupyter notebook
#
# * Click on the Notebook python 3 icon
# * A new notebook is created
# * Enter your first command, in the displayed cell, and type `SHIFT+ENTER` to run the code
#
# ```ipython3
# from spectrochempy import *
# ```
#
# * You can rename the notebook using context menu in the sidebar
#
# <img src='images/enter_code.png' />
#
# * Then you can click on the `+` sign to create a new cell. This cell is by default a Code cell which can contain
# Python code, but you can also enter some text, in Markdown format. Choose the content type of the cell in the
# dropdown menu, or by typing `ESC+M` .
# <img src='images/enter_md.png' />
#
# ## Markdown cheat sheet
# To get more information on Markdown format, you can look [here](mdcheatsheet.ipynb)
#
# ## Using the application in a web browser
#
# <div class='alert alert-warning'>
#    <b>In Progress</b>
#
#    For the moment we don’t yet have a graphical interface to offer other
#    than Jupyter notebooks or python scripts. It is in any case our
#    preferred way of working with SpectroChemPy because it offers all the
#    necessary flexibility for a fast and above all reproducible realization
#    of the different tasks to be performed on spectroscopic data.
#
#    However, we have started to create a simple interface using Dash which
#    will allow in a future version to work perhaps more simply for those who
#    do not have the time or the will to learn to master the rudiments of
#    python or who do not wish to program.
#  </div>

# %%
