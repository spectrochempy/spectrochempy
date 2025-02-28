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
#     version: 3.13.2
#   widgets:
#     application/vnd.jupyter.widget-state+json:
#       state: {}
#       version_major: 2
#       version_minor: 0
# ---

# %% [markdown]
# # Import of OMNIC files
#
# Thermo Scientific [OMNIC](https://www.thermofisher.com/search/results?query=OMNIC)
# software have two proprietary binary file formats:
#
# - .spa files that handle single spectra
# - .spg files which contain a group of spectra
#
# Both have been reverse engineered, hence allowing extracting their key data.
# The Omnic reader of Spectrochempy ( `read_omnic()` ) has been developed based on
# posts in open forums on the .spa file format and extended to .spg file formats.

# %% [markdown]
# ## Import spg file
#
# Let's import an .spg file from the `datadir` (see :ref:`import.ipynb` for details)):
# and display its main attributes:

# %%
import spectrochempy as scp

# %% {"pycharm": {"name": "#%%\n"}}
X = scp.read_omnic("irdata/CO@Mo_Al2O3.SPG")
X

# %% [markdown]
# The displayed attributes are detailed in the following:
#
# - `name` is the name of the group of spectra as it appears in the .spg file. OMNIC
#   sets this name to the .spg filename used at the creation of the group.
#   In this example, the name ("Group sust Mo_Al2O3_base line.SPG") differs
#   from the filename (`"CO@Mo_Al2O3.SPG"`) because the latter has been changed from
#   outside OMNIC (directly in the OS).
#
# - `author` is that of the creator of the NDDataset (not of the .spg file, which, to
#   our knowledge, does not have
#   this type of attribute). The string is composed of the username and of the machine
#   name as given by the OS, e.g., `"username@machinename"`.
#   It can be accessed and changed using `X.author` .
#
# - `created` is the creation date of the NDDataset (again not that of the .spg file).
#   It can be accessed (or even changed) using `X.created` .
#
# - `description` indicates the complete pathname of the .spg file. As the pathname is
#   also given in the history (below), it can be a good practice to give a
#   self-explaining description of the group, for instance:


# %%
X.description = "CO adsorption on CoMo/Al2O3, difference spectra"
X.description

# %% [markdown]
# or directly at the import:

# %%
X = scp.read_omnic("irdata//CO@Mo_Al2O3.SPG", description="CO@CoMo/Al2O3, diff spectra")
X.description

# %% [markdown]
# - `history` records changes made to the dataset. Here, right after its creation, it
#   has been sorted by date (see below).
#
# Then come the attributes related to the data themselves:
#
# - `title` (not to be confused with the `name` of the dataset) describes the nature
#   of data (here **absorbance** ).
#
# - `values` shows the data as quantity (with their units when they exist - here a.u.
#   for absorbance units).
#
# - The numerical values ar accessed through the `data` attribute and the units
#   throughout `units` attribute.

# %%
X.values

# %%
X.data

# %%
X.units

# %% [markdown]
# - `shape` is the same as the ndarray `shape` attribute and gives the shape of the
#   data array, here 19 x 3112.
#
# Then come the attributes related to the dimensions of the dataset.
#
# - `x` : this dimension has one coordinate (a `Coord` object) made of the 3112 the
#   wavenumbers.

# %%
print(X.x)
X.x

# %% [markdown]
# - `y` : this dimension contains:
#
#   - one coordinate made of the 19 acquisition timestamps
#   - two labels:
#
#     - the acquisition date (UTC) of each spectrum
#     - the name of each spectrum.

# %%
X.y

# %% [markdown]
# - `dims` : Note that the `x` and `y` dimensions are the second and first
#    dimension respectively. Hence, `X[i,j]` will return the absorbance of the ith
#    spectrum at the jth  wavenumber.
#    However, this is subject to change, for instance if you perform operation on your
#    data such as
#    [Transposition](../processing/transformations.ipynb#Transposition). At any time
#    the attribute `dims` gives the correct names (which can be modified) and order of
#    the dimensions.

# %%
X.dims

# %% [markdown]
# ### Acquisition dates and `y` axis
#
# The acquisition timestamps are the *Unix times* of the acquisition, i.e. the time
# elapsed in seconds since the
# reference date of Jan 1st 1970, 00:00:00 UTC.

# %%
X.y.values

# %% [markdown]
# In OMNIC, the acquisition time is that of the start of the acquisition.
# As such these may be not convenient to use directly (they are currently in the order
# of 1.5 billion...)
# With this respect, it can be convenient to shift the origin of time coordinate to
# that of the 1st spectrum,
# which has the index `0` :

# %%
X.y = X.y - X.y[0]
X.y.values

# %% [markdown]
# Note that you can also use the inplace subtract operator to perform the same
# operation.

# %%
X.y -= X.y[0]

# %% [markdown]
# It is also possible to use the ability of SpectroChemPy to handle unit changes. For
# this one can use the `to` or `ito` (inplace) methods.
#
# ```ipython
# val = val.to(some_units)
# val.ito(some_units)   # the same inplace
# ```

# %%
X.y.ito("minute")
X.y.values

# %% [markdown]
# As shown above, the values of the `Coord` object are accessed through the `values`
# attribute. To get the last
# values corresponding to the last row of the `X` dataset, you can use:

# %%
tf = X.y.values[-1]
tf

# %% [markdown]
# Negative index in python indicates the position in a sequence from the end, so -1
# indicate the last element.

# %% [markdown]
# Finally, if for instance you want the `x` time axis to be shifted by 2 minutes, it
# is also very easy to do so:

# %%
X.y = X.y + 2
X.y.values

# %% [markdown]
# or using the inplace add/subtract operator:

# %%
X.y -= 2  # this restore the previous coordinates
X.y.values

# %% [markdown]
# ### The order of spectra
#
# The order of spectra in OMNIC .spg files depends on the order in which the spectra
# were included in the OMNIC
# window before the group was saved. By default, spectrochempy reorders the spectra
# by acquisition date but the
# original OMNIC order can be kept using the `sortbydate=True` at the function call.
# For instance:

# %%
X2 = scp.read_omnic("irdata/CO@Mo_Al2O3.SPG", sortbydate=False)

# %% [markdown]
# In the present case, this will change nothing because the spectra in the OMNIC file
# were already ordered by increasing data.
#
# Finally, it is worth mentioning that a `NDDataset` can generally be manipulated as
# numpy ndarray. Hence, for
# instance, the following will inverse the order of the first dimension:

# %%
X = X[::-1]  # reorders the NDDataset along the first dimension going backward
X.y.values  # displays the `y` dimension

# %% [markdown]
# <div class='alert alert-info'>
# <b>Note</b>
#
# <strong>Case of groups with different wavenumbers</strong> <br/>
# An OMNIC .spg file can contain spectra having different wavenumber axes (e.g.
# different spacings or wavenumber
# ranges). In its current implementation, the spg reader will purposely return an error
# because such spectra
# <i>cannot</i> be included in a single NDDataset which, by definition, contains items that
# share common axes or dimensions !
# Future releases might include an option to deal with such a case and return a list of
# NDDatasets. Let us know if you
# are interested in such a feature, see <a href="https://www.spectrochempy.fr/devguide/issues.html">Bug reports and enhancement requests.</a>
# </div>
#

# %% [markdown]
# ## Import of .spa files
#
# The import of a single spectrum follows exactly the same rules as that of the import
# of a group:

# %%
scp.read_omnic("irdata/subdir/7_CZ0-100_Pd_101.SPA")

# %% [markdown]
# The omnic reader can also import several spa files together, providing that they share
# a common axis for the wavenumbers.
#
# This is the case of the following files in the irdata/subdir directory:
# "7_CZ0-100 Pd_101.SPA", ..., "7_CZ0-100 Pd_104.spa".
#
# It is possible to import them in a single NDDataset by using
# the list of filenames
# in the function call:

# %%
list_files = (
    "7_CZ0-100_Pd_101.SPA",
    "7_CZ0-100_Pd_102.SPA",
    "7_CZ0-100_Pd_103.SPA",
    "7_CZ0-100_Pd_104.SPA",
)
scp.read_omnic(list_files, directory="irdata/subdir", name="Merged 7_CZ0-100 Pd")

# %% [markdown]
# When compatible .spa files are alone in a directory, a very convenient is to call the
# read_omnic method
# using only the directory path as argument that will gather the .spa files together:

# %%
scp.read_omnic("irdata/subdir/1-20")

# %% [markdown]
# In the case  where not all files are compatibles, they are returned in different NDDatasets(with independent merging).
#
# For example:

# %%
Y = scp.read_omnic("irdata/subdir/")
Y

# %% [markdown]
# Here we get a list of two NDDataset because there is two type of file in the directory (`.spa` and `.srs`).
#
# The desired dataset can be obtained using a list:

# %%
Y[1]

# %% [markdown]
# Other ways to select only the required file with extension (`.spa`)are:
#
# - writing a list as previously explicitely  listing the required files.
# - using a more specific reader:

# %%
scp.read_spa("irdata/subdir/")

# %% [markdown]
# - using a pattern filter

# %%
scp.read_omnic("irdata/subdir/", pattern="*.spa")

# %% [markdown]
# One advantage of the latter solution is a greter flexibility. For instance the lollowing will select only the `*101.spa` and `*102.spa`:

# %%
scp.read_omnic("irdata/subdir/", pattern="*10[12].spa", merge=False)

# %% [markdown]
# ## Handling Metadata

# %% [markdown]
# Here is an example of accessing metadata

# %%
X = scp.read_omnic("irdata/CO@Mo_Al2O3.SPG")
print(f"Title: {X.title}")
print(f"Origin: {X.origin}")
print(f"Description: {X.description}")

# %% [markdown]
# and now do some modifications:

# %%
X.title = "Modified title"
X.origin = "OMNIC measurement"
X.description = "Modified description"
print("Modified metadata:")
print(f"Title: {X.title}")
print(f"Origin: {X.origin}")
print(f"Description: {X.description}")

# %% [markdown]
# Reading the metadata now reflect the change

# %%
X.title

# %% [markdown]
# ## Error Handling

# %% [markdown]
# When trying to read file, it is a good practice to handle errors explicitely. For example:

# %%
try:
    X = scp.read_omnic("nonexistent_file.spa")
except FileNotFoundError:
    scp.error_(FileNotFoundError, "File not found")
except Exception as e:
    scp.error_(f"Error reading file: {e}")

# %% [markdown]
# ## Advanced Data Operations

# %% [markdown]
# Example of data manipulation:

# %%
X = scp.read_omnic("irdata/CO@Mo_Al2O3.SPG")

# %% [markdown]
# - Baseline correction

# %%
X_corrected = X - X[0]  # Subtract first spectrum as baseline

# %% [markdown]
# - Normalization

# %%
X_normalized = X / X.max()

# %%
print("Original data shape:", X.shape)
print("Max value before normalization:", X.max())
print("Max value after normalization:", X_normalized.max())
