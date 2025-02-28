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
#   toc-showcode: false
#   toc-showtags: false
#   widgets:
#     application/vnd.jupyter.widget-state+json:
#       state: {}
#       version_major: 2
#       version_minor: 0
# ---

# %% [markdown]
# # Import Data in SpectroChemPy
#
# This tutorial shows how to import data in **SpectroChemPy (SCPy)** .
#
# First, let's `import spectrochempy as scp` in the current
# namespace, so that all spectrochempy commands will be called as
# `scp.method(<method parameters>)` .

# %%
import spectrochempy as scp

# %% [markdown]
# ## Generic read command

# %% [markdown]
# To read a file containing spectrocopic data or similar,
# the `read` method can be used.
# This method will try to guess the file format based on
# the file extension.

# %%
X = scp.read("wodger.spg")

# %% [markdown]
# The above command will extract the data from the file `wodger.spg`
# and store it in a `NDDataset` object named `X`.
# To display information about the dataset, simply type `X`
# in a cell and run it.
# %%
X

# %% [markdown]
# In this case, the data were in an OMNIC file format,
# and the `read` method guessed it correctly using the file name extension.
# The `read` method can also read other file formats, such as
# OPUS, JCAMP-DX, CSV, MATLAB, TOPSPIN, etc. or even a directory.

# %% [markdown]
# ## Using a specific reader

# %% [markdown]
# Instead of using the generic read method, you can also use a specific
# reader, such as `read_omnic`, `read_opus`, `read_csv`, `read_jcamp`, etc.
# These methods are more specific and will only read the file format
# they are. For example, `read_omnic` will only read OMNIC files.

# %% [markdown]
# The following table lists the available file readers in SCPy
# along with the corresponding file formats and extensions they support:
#
# | Reader         | File Formats                                      | Extensions      |
# |---------------|-------------------------------------------------|----------------|
# | read_omnic,<br/>read_spa,<br/>read_spg,<br/>read_srs    | Thermo Scientific/Nicolet OMNIC files          | .spa, .spg, .srs     |
# | read_opus     | Bruker OPUS files                              | .0, .1, .000, ... |
# | read_csv      | Comma-Separated Values (CSV) files             | .csv           |
# | read_jcamp, <br/>read_dx| JCAMP-DX spectral data files                   | .dx, .jdx      |
# | read_matlab,<br/>read_mat   | MATLAB files                                   | .mat, .dso     |
# | read_topspin  | Bruker TopSpin NMR files                       | fid, ser, 1r, 1i, 2rr... |
# | read_labspec  | LABSPEC6 spectral data files                   | .txt           |
# | read_wire,<br/>read_wdf | Renishaw Wire files                     | .wdf           |
# | read_scp      | SpectroChemPy-specific files                   | .scp           |
# | read_soc,<br/>read_ddr,<br/>read_hdr,<br/>read_sdr     | Surface Optics Corporation files               | .ddr, .hdr, .sdr |
# | read_galactic | Galactic spectral files                        | .spc           |
# | read_quadera  | Pfeiffer Vacuum QUADERA mass spectrometer files | .txt           |
# | read          | Generic reader (automatically detects format)  | -              |
# | read_dir      | Reads all supported files in a directory       | -              |
# | read_zip      | Reads files from a ZIP archive                 | .zip           |
# | read_carroucell | Reads files from a carrousel experiment directory | -          |
#
# The `read_dir` function scans a directory and reads all supported files,
# returning a list of `NDDataset` objects.
#
# Other reader functions return either a single `NDDataset` or multiple `NDDataset`
# objects, depending on the file type and content.
#
# Further details on specific cases are provided below. See the section [Reading directories](#Reading-directories).

# %% [markdown]
# ## Using relative or absolute pathnames

# %% [markdown]
# In the above examples, the file `wodger.spg` was read from the current working directory.
#
# If the file is located in another directory, the full path to the file can be provided. For example:
#
# ```ipython
# X = scp.read('/users/Brian/s/Life/wodger.spg')
# ```
#
# or, for Windows:
#
# ```ipython
# X = scp.read(r'C:\users\Brian\s\Life\wodger.spg')
# ```
#
# Notes:
# - The path separator is a backslash `\` on Windows, but in many contexts, backslash is also used as an escape character to
#   represent non-printable characters.
#   To avoid problems, either it has to be escaped itself, a double backslash `\\`, or one can also use raw string literals
#   to represent Windows paths.
#   These are string literals that have an `r` prepended to them. In raw string literals, the `\\` represents
#   a literal backslash: `r'C:\users\Brian'`.
# - In python, the slash `/` is used as the path separator in all systems (Windows, Linux, OSX, ...).
#   So it can be used in all cases. For exemple:
#
#   ```ipython
#   X = scp.read('C:/users/Brian/s/Life/wodger.spg')
#
#   ```
#
# - The use of relative pathnames is a good practice. SpectroChemPy readers use relative paths.
#   If the given path is not absolute,
#   then SpectroChemPy will search relative to the current directory or to a directory specified using the `directory`keywords.
#
#   For example:
#
#   ```ipython
#   X = scp.read('wodger.spg', directory='C:/users/Brian/s/Life')
#   X = scp.read('Life/wodger.spg', directory='C:\\users\\Brian\\s')
#
#   ```
#
# - The `os` or `pathlib` modules can be used to work with pathnames.
#   See the section
#   [Good practice:Use of os or pathlib packages](#Use-os-or-pathlib-packages).
# - The `preferences.datadir` variable can be used to set a default directory where to look for data.
#   See the section [Another default search directory: datadir](#Another-default-search-directory:-datadir).
#

# %% [markdown]
# ## Good practices
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# ### Use relative paths
#
# As path are system dependent, it is a good practice to use relative pathnames in scripts and notebooks.
#
# If, for instance, Brian has a project organised in a folder (`s` ) with a directory
# dedicated to input data (`Life` )
# and a notebook for preprocessing (`welease.ipynb` ) as illustrate below:
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
# Then running this project in John's Linux computer (e.g. in `/home/john/s_copy` )
# will certainly result in execution
# errors if absolute paths are used in the notebook:
#
# ```text
# OSError: Can't find this filename C:\users\Brian\s\life\wodger.spg
# ```
#
# Fortunately, SpectroChemPy readers use relative paths. If the given path is not
# absolute, then SpectroChemPy will search in the current directory. Hence, the opening
# of the `spg` file from scripts in `welease.ipynb` can be made
# by the command:
#
# ```ipython
# X = scp.read('Life/wodger.spg')
# ```
#
# or:
#
# ```ipython
# X = scp.read('wodger.spg', directory='Life')
# ```
#
# ### Use os or pathlib packages
#
# In python, working with pathnames is classically done with dedicated modules such as
# `os` or `pathlib` python modules.
# With `os` we mention the following methods that can be particularly useful:
#
# ```ipython
# import os
# os.getcwd()              # returns the absolute path of the current working directory
#                          # preferences.datadir
# os.path.expanduser("~")  # returns the home directory of the user
# os.path.join('path1','path2','path3', ...)
#                          # intelligently concatenates path components
#                          # using the system separator (`/` or `\\` )
# ```
#
# Using `Pathlib` is even simpler:
#
# ```ipython
# from pathlib import Path
# Path.cwd()               # returns the absolute path of the current working directory
# Path.home()              # returns the home directory of the user
# Path('path1') / 'path2' / 'path3' / '...'   # intelligently concatenates path
# components
# ```
#
# The interested readers will find more details on the use of these modules here:
#
# - [os](https://docs.python.org/3/library/os.html)
# - [pathlib](https://docs.python.org/3/library/pathlib.html)
#
# #### Another default search directory: datadir
#
# Spectrochempy also comes with the definition of a second default directory path where
# to look at the data: the `datadir` directory. It is defined in the variable `preferences.datadir` which
# is imported at the same time as spectrochempy. By default, `datadir` points in the
# '$HOME/.spectrochempy/tesdata' directory.:

# %%
DATADIR = scp.preferences.datadir
DATADIR

# %% [markdown]
# DATADIR is already a pathlib object and so can be used easily

# %%
scp.read_omnic(DATADIR / "wodger.spg")

# %% [markdown]
# It can be set to another pathname *permanently* (i.e., even after computer restart)
# by a new assignment:
#
# ```python
# scp.preferences.datadir = 'C:/users/Brian/s/Life'
# ```
#
# This will change the default value in the SCPy preference file located in the hidden
# folder
# ` .spectrochempy/` at the root of the user home directory.
#
# Finally, by default, the import functions used in Spectrochempy will search the data
# files using this order of
# precedence:
#
# 1. try absolute path
# 2. try in current working directory
# 3. try in `datadir`
# 4. if none of these works: generate an OSError (file or directory not found)
#
# %% [markdown]
# ## Reading directories

# %% [markdown]
# The `read_dir` function is designed to read an entire directory, create NDDatasets for each file, and finally merge all compatible datasets. Let's see an example:

# %% [markdown]
# - Here is a list of the files presents in `DATADIR/irdata/subdir/`

# %%
folder = DATADIR / "irdata" / "subdir"
[str(item.relative_to(DATADIR)) for item in folder.glob("*.*")]

# %% [markdown]
# - Now read all files in the `DATADIR/irdata/subdir/` directory  (*i.e.,*, four `.spa` files and one `.srs` file). Any file in unknown format will be ignored silently:

# %%
scp.read_dir(folder)
# %% [markdown]
# The above command have read all files in the `DATADIR/irdata/subdir/` directory and merged them into two groups of compatible NDDatasets:
#
# * a first `NDDataset` object (id: 0, shape [335,1868]) comes from the single `.srs` file.
# * a second `NDDataset` object (id: 1, shape [335,1868]) comes from the merging of four `.spa` files.

# %% [markdown]
# Merging  compatible NDDataset is the default behavior of `read_dir`  (or  equivalently `read`). If you want to read the files separately, you can use the `merge=False` keyword:
# %%
scp.read_dir(folder, merge=False)

# %% [markdown]
# As expected the result is a list of 5 NDDataset objects, one for each file in the directory.

# %% [markdown]
# ## Additional options for reading directories

# %% [markdown]
# The `read_dir`/`read` function has additional options to control the behavior of the reading process:
#
# - `recursive`: if `True`, the function will scan the directory recursively and read all supported files in all subdirectories.
# - `pattern`: a string or a list of strings that can be used to filter the files to be read. Only files whose name matches the pattern will be read.

# %% [markdown]
# Let's see an example with the `recursive` option:
#
# First we list files in all directories under `DATADIR/irdata/subdir/`:

# %%
[str(item.relative_to(DATADIR)) for item in folder.glob("**/*.*")]

# %% [markdown]
# the ìrdata/subdir/` directory contains two subdirectory `1-20` and `20-50` with two additional `.spa` files.
#
# Now we read all files (a total of 9) in the `DATADIR/irdata/subdir/` directory and its subdirectories:
# %%
scp.read_dir(folder, recursive=True, merge=False)
# %% [markdown]
# and we allow merging them:
# %%
scp.read_dir(folder, recursive=True)
# %% [markdown]
# As the 8 `.spa` files are compatible, they are merged into a single `NDDataset` object. The `.srs` file is read separately.

# %% [markdown]
# Specific reader can equivalently read folder recursively:
# %%
scp.read_omnic(folder, recursive=True)

# %% [markdown]
# Let's see an example with the `pattern` option:
#
# We read all files in the `DATADIR/irdata/subdir/` directory and its subdirectories, but only those with the `.spa` extension and whose name contains the string `4`:
# %%
scp.read_dir(folder, recursive=True, pattern="*4*")

# %% [markdown]
# The above pattern "\*4\*" match only two files which are then merged and returned as a single `NDDataset` object.

# %% [markdown]
# This `pattern` option is obviously interesting to select only a type of extension:

# %%
scp.read_dir(folder, recursive=True, pattern="*.spa")

# %%
scp.read(folder, recursive=True, pattern="*.spa")  # equivalent

# %%
scp.read_omnic(folder, recursive=True, pattern="*4.spa")  # equivalent

# %% [markdown]
# This way the ".srs" file is ignored.

# %% [markdown]
# ## Reading files from a ZIP archive

# %% [markdown]
# The `read_zip` function is designed to read files from a ZIP archive. It can be used to read a single file
# or all files in the archive. As usual, by default all files are merged. The `merge` keyword can be used to
# read the files separately.

# %%
scp.read(
    "https://eigenvector.com/wp-content/uploads/2019/06/corn.mat_.zip", merge=False
)
