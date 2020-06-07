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
# # Import IR Data
#
# This tutorial shows the specifics related to infrared data import in Spectrochempy. As prerequisite, the user is
# expected to have read the [Import Tutorial](Import.ipynb).
#
# Let's first import spectrochempy:

# %% {"jupyter": {"outputs_hidden": false}, "pycharm": {"name": "#%%\n"}}
import spectrochempy as scp
import os

# %% [markdown]
# ## Supported file formats
#
# At the time of writing of this tutorial (Scpy v.0.1.18), spectrochempy has the following readers which are specific
# to IR data:
#
# - `read_omnic()` to open omnic (spa and spg) files
# - `read_bruker_opus()` to open Opus (*.0, ...) files
# - `read_jdx()` to open an IR JCAMP-DX datafile
#
# General purpose data exchange formats such as  \*.csv or \*.mat will be treated in another tutorial (yet to come...)
# can also be read using:
#
# - `read_csv()` to open csv files
# - `read_matlab()` to open .mat files
#
# ## Import of OMNIC files
#
# [Thermo Scientific OMNIC](https://www.thermofisher.com/search/browse/category/us/fr/602580/FTIR%2C+NIR+%26amp%3B+
# Raman+Software+%26amp%3B+Libraries)
# software have two proprietary binary file formats:
# - .spa files that handle single spectra
# - .spg files which contain a group of spectra
#
# Both have been reverse engineered, hence allowing extracting their key data. The Omnic reader of
#  Spectrochempy (`read_omnic()`) has been developed based on posts in open forums on the .spa
#  file format and extended to .spg file formats.
#
#
# ### a) import spg file
#
# Let's import an .spg file from the `datadir` (see [Import Tutorial](Import.ipynb)) and display
# its main attributes:

# %% {"pycharm": {"name": "#%%\n"}}
X = scp.read_omnic('irdata/CO@Mo_Al2O3.SPG')
X

# %% [markdown]
# The displayed attibutes are detailed in the following.
#
# - `name` is the name of the group of spectra as it appears in the .spg file. OMNIC sets this name to the .spg
# filename used at the creation of the group. In this example, the name ("Group sust Mo_Al2O3_base line.SPG") differs
# from the filemane ("CO@Mo_Al2O3.SPG") because the latter has been changed from outside OMNIC (directly in th OS).
#
# - `author` is that of the creator of the NDDataset (not of the .spg file, which, to our knowledge, does not have
# thus type of attribute). The string is composed of the username and of the machine name as given by the OS:
# usermane@machinename.
#
# - "created" is the creation date of the NDDataset (again not that of the .spg file). The actual name of the attribute
# is `date` and can be accessed (or even changed) using `X.date`
#
# - `description` indicates the complete pathname of the .spg file. As the pathname is also given in the history (below)
# , it can be a good practice to give a self explaining description of the group, for instance:

# %%
X.description = 'CO adsorption on CoMo/Al2O3, difference spectra'
print(X.description)

# %% [markdown]
# or directly at the import:

# %%
X = scp.read_omnic('irdata//CO@Mo_Al2O3.SPG', description='CO@CoMo/Al2O3, diff spectra')
print(X.description)

# %% [markdown]
# - `history` records changes made to the dataset. Here, right after its creation, it has been sorted by date
# (see below).
#
# Then come the attributes related to the data themselves:
#
# - `title` (not to be confused with the `name` of the dataset) describes the nature of data (here absorbance)
#
# - "values" shows a sample of the first and last data and their units when they exist (here a.u. for absorbance units).
# The numerical values ar accessed through the`data` attibute and the units throut `units` attribute.

# %%
print(X.data)
print(X.units)

# %% [markdown]
# - `shape` is the same as the ndarray `shape` attribute and gives the shape of the data array, here 19 x 3112.
#
# Then come the attributes related to the dimensions of the dataset.
#
#
# - the `x` dimension has one coordinate made of the 3112 the wavenumbers.
#
#
# - the `y` dimension contains:
#
#     - one coordinate made of the 19 acquisition timestamps
#
#     - two labels
#
#         - the acquision date (UTC) of each spectrum
#
#         - the name of each spectrum.
#
#
# Note that the `x` and `y` dimensions are the second and first dimension respectively. Hence, `X[i,j]` will return
# the absorbance of the ith spetrum at the jth  wavenumber.
#
# **Note: acquisition dates and `y` axis**
#
# The acquisition timestamps are the *Unix times* of the acquisition, i.e. the time elapsed in seconds since the
# reference date of Jan 1st 1970, 00:00:00 UTC. In OMNIC, the acquisition time is that of the start of the acquisison.
# As such these may be not convenient to use directly (they are currently in the order of 1.5 billion...)
# With this respect, it can be convenient to shift the origin of time coordinate to that of the 1st spectrum,
# which has the index `0`:

# %%
X.y = X.y - X.y[0]
X.y

# %% [markdown]
# It is also possible to use the ability of Scpy to handle unit changes. For this one can use the  `to` or `ito` (inplace) methods.
#
#     val = val.to(some_units) 
#     val.ito(some_units)   # the same inplace

# %%
X.y.ito("minute")
X.y

# %% [markdown]
# Note that the valued that are displayed are rounded, not the values stored internally. Hence, the relative time in
# minutes of the last spectrum is:

# %%
# the last item of a NDDataset such as X can be referred by a negative index (-1). The values of the Coord object
# are accessed through the `values` attribute:
tf = X[-1].y.values
tf

# %% [markdown]
# which gives the exact time in seconds:

# %%
tf.ito('s')
tf

# %% [markdown]
# Finally, if the time axis needs to be shifted by 2 minutes for instance, it is also very easy to do so:

# %%
X.y = X.y + 2
X.y

# %% [markdown]
# **Note: The order of spectra**
#
# The order of spectra in OMNIC .spg files depends depends on the order in which the spectra were included in the OMNIC
# window before the group was saved. By default, sepctrochempy reorders the spectra by acquisistion date but the
# original OMNIC order can be kept using the `sortbydate=True` at the function call. For instance:

# %%
X2 = scp.read_omnic('irdata/CO@Mo_Al2O3.SPG', sortbydate=False)

# %% [markdown]
# In the present case this will not change nothing because the spectra in the OMNIC file wre already ordered by
# increasing data.
#
# Finally, it is worth mentioning that the NDDatasets can generally be manipulated as numpy ndarray. Hence, for
# instance, the following will inverse the order of the first dimension:

# %%
X = X[::-1, :]  # reorders the NDDataset along the first dimension going backward
X.y  # displays the `y` dimension

# %% [markdown]
# **Note: Case of groups with different wavenumbers**
#
# An OMNIC .spg file can contain spectra having different wavenumber axes (e.g. different spacings or wavenumber
# ranges). In its current implementation, the spg reader will purposedly return an error because such spectra
# *cannot* be included in a single NDdataset which, by definition, contains items that share common axes or dimensions !
# Future releases might include an option to deal with such a case and return a list of NDDasets. Let us know if you
# are interested in such a feature, see [Bug reports and enhancement requests]
# (https://www.spectrochempy.fr/dev/dev/issues.html).
#

# %% [markdown]
# ### b) Import of .spa files
#
# The import of a single spectrum follows exactly the same rules as that of the import of a group:

# %%
Y = scp.read_omnic('irdata/subdir/7_CZ0-100 Pd_101.SPA')
Y

# %% [markdown]
# The omnic reader can also import several spa files together, providing that they share a common axis for the
# wavenumbers. Tis is the case of the following files in the irdata/subdir directory: "7_CZ0-100 Pd_101.SPA", ...,
# "7_CZ0-100 Pd_104.spa". It is possible to import them in a single NDDataset by using the list of filenames
# in the function call:

# %%
list_files = ["7_CZ0-100 Pd_101.SPA", "7_CZ0-100 Pd_102.SPA", "7_CZ0-100 Pd_103.SPA", "7_CZ0-100 Pd_104.SPA"]
X = scp.read_omnic(list_files, directory='irdata/subdir')
print(X)

# %% [markdown]
# In such a case ase these .spa files are alone in the directory, a very convenient is the read_dir() method
# that will gather the .spa files together:

# %%
X = scp.read_dir('irdata/subdir')
print(X)

# %% [markdown] {"pycharm": {"name": "#%% md\n"}}
# ## Import of Bruker OPUS files
#
# [Bruker OPUS](https://www.bruker.com/products/infrared-near-infrared-and-raman-spectroscopy/
# opus-spectroscopy-software.html) files have also a proprietary file format. The Opus reader (`read_opus()`)
# of spectrochempy is essentially a wrapper of the python module
# [brukeropusreader](https://github.com/spectrochempy/brukeropusreader) developed by QED. It imports absorbance
# spectra (the AB block), acquisition times and name of spectra.
#
# The use of `read_opus()` is similar to that of  `read_omnic()` for .spa files. Hence, one can open sample
# Opus files contained in the `datadir` using:

# %%
Z = scp.read_opus(['test.0000', 'test.0001', 'test.0002'], directory='irdata/OPUS')
print(Z)

# %% [markdown]
# or:

# %%
Z2 = scp.read_dir('irdata/OPUS')
print(Z2)

# %% [markdown]
# Note that supplementary informations as to the imported spectra can be obtained by the direct use of
# `brukeropusreader`. For instance:

# %%
from brukeropusreader import read_file  # noqa: E402

opusfile = os.path.join(scp.general_preferences.datadir, "irdata", "OPUS", "test.0000")  # the full pathname of the file
Z3 = read_file(opusfile)  # returns a dictionary of the data and metadata extracted
Z3.keys()  # returns the key of the dictionary

# %%
Z3['Optik']  # looks what is the Optik block:

# %% [markdown]
# ## Import/Export of JCAMP-DX files
#
# [JCAMP-DX](http://www.jcamp-dx.org/) is an open format initially developped for IR data and extended to
# other spectroscopies. At present, the JCAMP-DX reader implemented in Spectrochempy is limited to IR data and
# AFFN encoding (see R. S. McDonald and Paul A. Wilks, JCAMP-DX: A Standard Form for Exchange of Infrared Spectra in
# Readable Form, Appl. Spec., 1988, 1, 151–162. doi:10.1366/0003702884428734 fo details).
#
# The JCAMP-DX reader of spectrochempy has been essentially written to read again the jcamp-dx files exported by
# spectrochempy `write_jdx()` writer.
#
# Hence, for instance, the first dataset can be saved in the JCAMP-DX format:

# %%
X.write_jdx('CO@Mo_Al2O3.jdx')

# %% [markdown]
# then used (and maybe changed) by a 3rd party software, and re-imported in spectrochempy:

# %%
newX = scp.read_jdx('CO@Mo_Al2O3.jdx')
os.remove('CO@Mo_Al2O3.jdx')
print(newX)

# %% [markdown]
# It is important to note here that the conversion to JCAMP-DX changes the ast digits of absorbances and wavenumbers:

# %%
print('Mean change in absorbance: {}'.format((X.data - newX.data).mean()))
print('Mean change in wavenumber: {}'.format((X.x.data - newX.x.data).mean()))

# %% [markdown]
# This is much beyond the experimental accuracy but can lead to undesirable effects. For instance:

# %%
try:
    X - newX
except ValueError as e:
    print(e)

# %% [markdown]
# returns an error because of the small shift of coordinates. We will see in another tutorial how to re-align datasets
# and deal with these small problems. It is worth noticing that similar distorsions arise in commercial softwares,...
# except that the user is not notified.

# %% [markdown]
# -- this is the end of this tutorial --
