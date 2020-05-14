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

# %% [markdown] {"pycharm": {"name": "#%% md\n", "is_executing": false}}
# # Import Data
#
# This tutorial shows how to import data in Spectrochempy. First, let's import spectrochempy in the current namespace.

# %% {"pycharm": {"name": "#%%\n", "is_executing": false}, "jupyter": {"outputs_hidden": false}}
from spectrochempy import *

# %% [markdown]
# # 1. Dialog boxes
#
# Retrieving Files and Directories, in day-to-day work is often made through Dialog Boxes. While we do not recommand this procedure for adanced usage (see below), it is quite easy to do that with **Scpy**. For instance, to import IR spectra in the Omnic format (.spa or .spg), the command:
#
# ```
# X = read_omnic()
# ```
#
# will open a dialog box such as shown in this this image:
#
# ![Drawing](figures/OpenDialog.png)
#
#
# The dialog Box allows selecting the file which data will be loaded in the variable `X`. Try for instance to run the cell below, and select an omnic spg datafile (select the .spg extension), which you can find in the `irdata` directory. 
#
# > Note: the dialog box does not necessarily pops up in the foreground: check your task bar ! 

# %%
X = read_omnic()
print(X)

# %% [markdown]
# If successful, the output of the above cell should read something like 
#
# ```
# Out[2] NDDataset: [float32] a.u. (shape: (y:2, x:5549))
# ```
#
# The size of the `y` and `x` dimension will depend, of course, of the file that you have selected ! If you did not select any file (e.g. by pressing 'cancel' in th Dialog Box, the result will be `None`, as nothing has been loaded in `X`.
#
# > Note: By default the Dialog Box opens in the current directory, i.e. the directory in which this notebook is run. See below for more information 
#
#
# - At the time of writing this tutorial (Scpy v.0.1.18), the following commands will behave similarly:
#     - `read_opus()` to open Opus (*.0, ...) files
#     - `read_csv()` to open csv files
#     - `read_dir()` to open readable files in a directory
#     - `read_jdx()` to open an IR JCAMP-DX datafile
#     - `read_matlab()` to open MATLAB (.mat) files including Eingenvector's Dataset objects
#     - `read_omnic()` to open omnic (spa and spg) files
#  
#  
# - The list of readers available will hopefully increase in future **Scpy** releases:-) 

# %% [markdown]
# # 2. Import with explicit directory or file pathnames
#
# While the use of Dialog Box seems at first 'user-friendly', you will probably experience, that it is often NOT efficient because you will have to select the file *each time* the notebook (or the script) is run... Hence, the above commands can be used with the indication of the path to a directory, and/or to a filename. 
#
# If only a directory is indicated, the dialog box will open in this directory. For instance, on a WIN system, the following command:
#
# ```
# X = read_omnic(directory='C:\\')
# ```
#
# will open the dialog box at the root directory of the `C:` drive. 
#
# > Note that in this command, the backslash (`\`) is repeated twice. This is a specificity of python (and a handful of other languages): `\` is the escape character, so if you type `X = read_omnic(directory='C:\')`, a `SyntaxError` will be raised because python expects a character after the first `\`.
#
# On the other hand if a `filename` is passed, like here: 
#
# ```
# X = read_omnic(directory='C:\\', filename='wodger.spg')
# ```
#
# then Scpy will attempt opening a file named `wodger.spg` supposedly located in `C:\`. 
#
# Imagine now that the file of interest is actually located in `C:\users\Brian\s\Life`. The following 
# commands are all equivalent and will allow opening the file: 
#
# - using only the full pathname of the file (note once again, the double backslashes):
#
#     ```
#     X = read_omnic(filename='C:\\users\\Brian\\s\\Life\\wodger.spg')
#     ```
#
#
# - more simply, without the `filename=` keyword: 
#
#     ```
#     X = read_omnic('C:\\users\\Brian\\s\\Life\\wodger.spg')
#     ```
#
#
# - or using a combination of directory and file pathnames:
#
#     ```
#     X = read_omnic(directory='C:\\users\\Brian\\s\\Life', filename='wodger.spg')
#     X = read_omnic(directory='C:\\users\\Brian\\s', filename='Life\\wodger.spg')
#     ```
#  
#  
# - etc...
#
#
# # 4. A good practice: use relative paths
#
# The above directives require explicitly writing the absolute pathnames, which are virtually always computer specific. If, for instance, Brian has a project organized in a folder (`s`) with a directory dedicated to input data (`Life`) and a notebook for preprocessing (`welease.ipynb`) as illustrate below:
#
# ```
# C:\users
# |    +-- Brian                   
# |    |    +-- s                 
# |    |    |   +-- Life          
# |    |    |   |   +-- wodger.spg
# |    |    |   +-- welease.ipynb 
# ```
#
# Then running this project in John's Linux computer (e.g. in `\home\john\s_copy`) will certainly result in execution errors if absolute paths are used in the notebook:  
#
# ```
# OSError: Can't find this filename C:\users\Brian\s\life\wodger.spg
# ``` 
#
# In this respect, a good practice consists in using relative pathnames in scripts/notebooks and fortunately, Spectrochempy readers use relative paths. If the given path is not absolute, then spectrochempy will search in the current directory. Hence the openening of the spg file from scripts in `welease.ipynb` can be made by the command: 
#
# ```
# X = read_omnic('Life\\wodger.spg'))
# ```
#
# or other variants such as:
#
# ```
# X = read_omnic('wodger.spg', directory='Life')
# X = read_omnic(filename='wodger.spg', directory='Life')
# ```
# # 5. Good practice: use `os` or `pathlib` modules
#
# In python, working with pathnames is classically done with dedicated modules such as `os` or `pathlib` python modules. As `os`is automatically imported with Scpy, we mention the following methods that can be particularely useful:
#
# - `os.getcwd()`: returns the absolute path of the current working directory (i.e. the directory of the script) 
# - `os.path.expanduser("~")` : returns the home directory of the user (e.g. the `C:\users\<username>` path on WIN platforms or `/home/<username>` on linux)
# - `os.path.join()`: concatenates intelligently path components.
#     
# The interested readers will find more details on te use of these modules here: 
#
# - [os - Miscellaneous operating system interfaces](https://docs.python.org/3/library/os.html)
# - [pathlib — Object-oriented filesystem paths](https://docs.python.org/3/library/pathlib.html)
#
# # 5. Another default search directory: `datadir`
#
# Spectrochempy comes also with the definition of a second default directory path where to look at the data: the `datadir` directory. It is defined in the variable `general_preferences.datadir` which is impotrted at the same time as spectrochempy. By default, `datadir` points in the 'scp_data\testdata' folder of spectrochempy:

# %%
print(general_preferences.datadir)

# %% [markdown]
# It can be set to another pathname *permanently* (i.e. even after computer restart) by a new assignment:
#
# ```
# general_preferences.datadir = 'C:\\Brian\\s\\Life'`
# ```
#
# This will change the default value in the spectrochempy preference file located in the hidden folder `.spectrochempy/` at the root of the user home directory.
#
# Finally, by default, the import functions used in Sepctrochempy will search the datafiles using this order of precedence:
#
#    1. try absolute path
#    2. try in current working directory
#    3. try in `datadir`
#    4. if none of these works: generate an OSError (file or directory not found)
#    
#     

# %% [markdown]
# --- This is the end of the tutorial ---
#
#
