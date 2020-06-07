# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
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

# %% [markdown] {"pycharm": {"is_executing": false, "name": "#%% md\n"}}
# # Import Data
#
# This tutorial shows how to import data in SpectroChemPy. First, let's ``import spectrochempy as scp`` in the current
# namespace, so that all spectrochempy commands will be called as ```scp.method(<method parameters>)```.

# %% {"jupyter": {"outputs_hidden": false}, "pycharm": {"is_executing": false, "name": "#%%\n"}}
import spectrochempy as scp

# %% [markdown]
# ## Dialog boxes
#
# Retrieving Files and Directories, in day-to-day work is often made through Dialog Boxes. While we do not recommand
# this procedure for adanced usage (see below), it is quite easy to do that with **Scpy**. For instance, to import
# IR spectra in the Omnic format (.spa or .spg), the `read_omnic()` command passed without any argument:
#
#     X = scp.read_omnic()
#
# will open a dialog box such as shown in this this image:
#
# <center><img id='drawings' width='800px'  src='figures/OpenDialog.png'></img></center>
#
# The dialog Box allows selecting the file which data will be loaded in the variable `X`. Try for instance to run the
# cell below, and select an omnic spg datafile (select the .spg extension), which you can find in the `irdata`
# directory.
#
# > **Note**: the dialog box does not necessarily pops up in the foreground: check your task bar !

# %%
X = scp.read_omnic()
print(X)

# %% [markdown]
# If successful, the output of the above cell should read something like
#
#     Out[2] NDDataset: [float32] a.u. (shape: (y:2, x:5549))
#
# The size of the `y` and `x` dimension will depend, of course, of the file that you have selected ! If you did not
# select any file (e.g. by pressing 'cancel' in th Dialog Box, the result will be `None`, as nothing has been loaded
# in `X`.
#
# > **Note**: By default the Dialog Box opens in the current directory, i.e. the directory in which this notebook is
#   run.
#
# See below for more information
#
# - At the time of writing this tutorial (SepctroChemPy v.0.1.18), the following commands will behave similarly:
#     - `read_bruker_opus()` to open Opus (*.0, ...) files
#     - `read_csv()` to open csv files
#     - `read_dir()` to open readable files in a directory
#     - `read_jdx()` to open an IR JCAMP-DX datafile
#     - `read_matlab()` to open MATLAB (.mat) files including Eingenvector's Dataset objects
#     - `read_omnic()` to open omnic (spa and spg) files
#
# - The list of readers available will hopefully increase in future **Scpy** releases:-)

# %% [markdown]
# ## Import with explicit directory or file pathnames
#
# While the use of Dialog Box seems at first 'user-friendly', you will probably experience, that it is often NOT
# efficient because you will have to select the file *each time* the notebook (or the script) is run... Hence, the
# above commands can be used with the indication of the path to a directory, and/or to a filename.
#
# If only a directory is indicated, the dialog box will open in this directory.
# > Note that on Windows the path separator is a backslash `\`. However, in many contexts,
# > backslash is also used as an escape character in order to represent non-printable characters. To avoid problems,
# > either it has to be escaped itself, sing a double backslash or one can also use raw string literals
# > to represent Windows paths. These are string literals that have an `r` prepended to them. In raw string literals
# > the `\` represents a literal backslash: `r'C:\users\Brian'`:
#
# For instance, on Windows system, the two following commands are fully equivalent:
#
#     X = scp.read_omnic(directory='C:\\users\\Brian')
#
# or
#
#     X = scp.read_omnic(directory=r'C:\users\Brian')
#
# and will open the dialog box at the root directory of the `C:` drive.
#
# > **Note**: You can avoid to use the form `\\` or the use of raw strings by using conventional slash `/`. In python they play the path separator
# > role, as well in Windows than in other unix-based system (Linux, OSX, ...)
#

# %%
X = scp.read_omnic(directory=r'C:\users\Brian')

# %% [markdown]
# If a `filename` is passed in argument, like here:
#
#     X = scp.read_omnic('wodger.spg', directory='C:/')
#
# then SpectroChemPy will attempt opening a file named `wodger.spg` supposedly located in `C:\`.
#
#
# Imagine now that the file of interest is actually located in `C:\users\Brian\s\Life`. The following
# commands are all equivalent and will allow opening the file:
#
# - using only the full pathname of the file:
#          
#       X = scp.read_omnic('C:/users/Brian/s/Life/wodger.spg')
#       
# - or using a combination of directory and file pathnames:
#       
#       X = scp.read_omnic('wodger.spg', directory='C:/users/Brian/s/Life'
#       X = scp.read_omnic('Life/wodger.spg', directory='C:/users/Brian/s')
#       
# - etc...

# %% [markdown]
# ## A good practice: use relative paths
#
# The above directives require explicitly writing the absolute pathnames, which are virtually always computer specific.
# If, for instance, Brian has a project organized in a folder (`s`) with a directory dedicated to input data (`Life`)
# and a notebook for preprocessing (`welease.ipynb`) as illustrate below:
#
# ```
# C:\users
# |    +-- Brian
# |    |    +-- s
# |    |    |   +-- Life
# |    |    |   |   +-- wodger.spg
# |    |    |   +-- welease.ipynb
#
# ```
#
# Then running this project in John's Linux computer (e.g. in `\home\john\s_copy`) will certainly result in execution
# errors if absolute paths are used in the notebook:
#
#     OSError: Can't find this filename C:\users\Brian\s\life\wodger.spg
#
# In this respect, a good practice consists in using relative pathnames in scripts and notebooks.
# Fortunately, Spectrochempy readers use relative paths. If the given path is not absolute, then spectrochempy will search
# in the current directory. Hence the opening of the `spg` file from scripts in `welease.ipynb` can be made
# by the command:
#
#     X = scp.read_omnic('Life/wodger.spg'))
#
# or:
#
#     X = scp.read_omnic('wodger.spg', directory='Life')
#
# ## Good practice: use `os` or `pathlib` modules
#
# In python, working with pathnames is classically done with dedicated modules such as `os` or `pathlib` python modules.
# As `os`is automatically imported with Scpy, we mention the following methods that can be particularely useful:
#
# - `os.getcwd()`: returns the absolute path of the current working directory (i.e. the directory of the script)
#
# - `os.path.expanduser("~")` : returns the home directory of the user (e.g. the `C:\users\<username>` path on WIN
# platforms or `/home/<username>` on linux)
# - `os.path.join()`: concatenates intelligently path components.
#
#
# The interested readers will find more details on te use of these modules here:
#
#
# - [os - Miscellaneous operating system interfaces](https://docs.python.org/3/library/os.html)
# - [pathlib — Object-oriented filesystem paths](https://docs.python.org/3/library/pathlib.html)
#
# ## Another default search directory: `datadir`
#
# Spectrochempy comes also with the definition of a second default directory path where to look at the data:
# the `datadir` directory. It is defined in the variable `general_preferences.datadir` which is impotrted at the same
# time as spectrochempy. By default, `datadir` points in the 'scp_data\testdata' folder of spectrochempy:

# %%
import os
print(os.getcwd())
X = scp.read_omnic('wodger.spg')

# %%
print(scp.general_preferences.datadir)

# %% [markdown]
# It can be set to another pathname *permanently* (i.e. even after computer restart) by a new assignment:
#
# ```
# general_preferences.datadir = 'C:/users/Brian/s/Life'`
# ```
#
# This will change the default value in the spectrochempy preference file located in the hidden folder
# `.spectrochempy/` at the root of the user home directory.
#
# Finally, by default, the import functions used in Sepctrochempy will search the datafiles using this order of
# precedence:
#
#    1. try absolute path
#    2. try in current working directory
#    3. try in `datadir`
#    4. if none of these works: generate an OSError (file or directory not found)
#

# %% [markdown]
# --- This is the end of the tutorial ---
#
#
