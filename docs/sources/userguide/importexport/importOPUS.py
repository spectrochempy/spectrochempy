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

# %% [markdown] {"editable": true, "slideshow": {"slide_type": ""}}
# ## Import of Bruker OPUS files
#
# [Bruker OPUS](https://www.bruker.com/en/products-and-solutions/infrared-and-raman/opus-spectroscopy-software.html)
# files have also a proprietary file format. The Opus reader (`read_opus()` ) of spectrochempy is essentially a wrapper of the python module
# [brukeropus](https://github.com/joshduran/brukeropus) developed by Josh Duran.
# The use of `read_opus()` is similar to that of  `read_omnic()`.

# %% [markdown]
# ### Basic loading of OPUS file

# %% [markdown]
# Individual Opus files can be opened and loaded as a new NDDataset using:

# %% {"editable": true, "slideshow": {"slide_type": ""}}
import spectrochempy as scp

# %% {"editable": true, "slideshow": {"slide_type": ""}}
Z = scp.read_opus("irdata/OPUS/test.0002")
Z

# %% [markdown] {"editable": true, "slideshow": {"slide_type": ""}}
# <div class='alert alert-info'>
# <b>Note:</b><br/>
# In the previous use of <strong>read_opus()</strong>, we have assumed that the file is in <strong>datadir</strong> folder (see the <a href="../import.html">import tutorial</a> or more details on this). <br/>If this not the case, you should use an absolute path or a relative path to the current notebook folder.
# </div>

# %% [markdown] {"editable": true, "slideshow": {"slide_type": ""}}
# For multifile loading, one can use:

# %% {"editable": true, "slideshow": {"slide_type": ""}}
Z1 = scp.read_opus("test.0000", "test.0001", "test.0002", directory="irdata/OPUS")
Z1

# %% [markdown] {"editable": true, "slideshow": {"slide_type": ""}}
# <div class='alert alert-info'>
# <b>Note:</b><br/>
#     By default all files in the given directory are merged as a single dataset if they are compatibles (*i.e.*, same shape, same type of experiment...).<br/>
#     If they cannot be merged due to imcompatible shape or type, separate datasets are returned with dataset merged in different groups.

# %% [markdown] {"editable": true, "slideshow": {"slide_type": ""}}
# For instance in the following, two dataset will be returned:

# %% {"editable": true, "slideshow": {"slide_type": ""}}
LZ1 = scp.read_opus(
    "test.0000", "test.0001", "test.0002", "background.0", directory="irdata/OPUS"
)
LZ1

# %% [markdown] {"editable": true, "slideshow": {"slide_type": ""}}
# Multifile loading can also be achieved by only specifying the directory to read, if all files in a directory must be read:

# %% {"editable": true, "slideshow": {"slide_type": ""}}
LZ = scp.read_opus("irdata/OPUS")
LZ

# %% [markdown] {"editable": true, "slideshow": {"slide_type": ""}}
# As previously two datasets are returned due to the imcompatible types and shapes of the experiments.
#
# If one desire to load of files into separate datasets, then set the **merge** attribute to False.

# %% {"editable": true, "slideshow": {"slide_type": ""}}
LZ1 = scp.read_opus("irdata/OPUS", merge=False)
LZ1

# %% [markdown]
# ### Loading given type of OPUS spectra

# %% [markdown]
# By default absortion spectra (**AB**) are load, if present in the file.
#
# Opus file however can contain several types of spectra and they can be retrieved eventually using the correct type in the call to `read_opus`:
#
# Types possibly availables and readables by `read_opus` are listed here:
#
# - `AB`: Absorbance (default if present in the file)
# - `TR`: Transmittance
# - `KM`: Kubelka-Munk
# - `RAM`: Raman
# - `EMI`: Emission
# - `RFL`: Reflectance
# - `LRF`: log(Reflectance)
# - `ATR`: ATR
# - `PAS`: Photoacoustic
# - `RF`: Single-channel reference spectra
# - `SM`: Single-channel sample spectra
# - `IGRF`: Reference interferogram
# - `IGSM`: Sample interferogram
# - `PHRF`: Reference phase
# - `PHSM`: Sample phase

# %% [markdown]
# It is possible to know which are the other types availables in the original file:

# %% {"editable": true, "slideshow": {"slide_type": ""}}
Z.meta.other_data_types

# %% [markdown] {"editable": true, "slideshow": {"slide_type": ""}}
# Thus if one wants to load the single-channel sample spectra, the read function syntax would be:

# %%
ZSM = scp.read_opus("irdata/OPUS/test.0002", type="SM")
ZSM

# %% [markdown] {"editable": true, "slideshow": {"slide_type": ""}}
# ### Reading OPUS file Metadata
#
# As just seen above, more informations can be obtained on the experiment and spectrometer parameters using the dataset metadata (**meta** attribute).
#
# For instance, to get a display of all metadata:
#

# %% {"editable": true, "slideshow": {"slide_type": ""}}
Z.meta

# %% [markdown] {"editable": true, "slideshow": {"slide_type": ""}}
# To display only a sub-group of metadata, you can use :

# %% {"editable": true, "slideshow": {"slide_type": ""}}
Z.meta.params.fourier_transform

# %% [markdown] {"editable": true, "slideshow": {"slide_type": ""}}
# or a single parameter:

# %% {"editable": true, "slideshow": {"slide_type": ""}}
Z.meta.params.optical.bms

# %%
Z.meta.params.optical.bms.value

# %% [markdown] {"editable": true, "slideshow": {"slide_type": ""}}
# ### Acting on a parameter

# %% [markdown] {"editable": true, "slideshow": {"slide_type": ""}}
# Metadata modification, addition, deletion are forbidden by default (`readonly=True`).
#
# For instance, if one want to add a new parameters, the following is not permitted and then raise an error:

# %% {"editable": true, "slideshow": {"slide_type": ""}}
try:
    Z.meta.xxx = "forbidden"
except ValueError:
    scp.error_("meta data dictionary is read only!")

# %% [markdown] {"editable": true, "slideshow": {"slide_type": ""}}
# To add a new value (or to modifify/delete an existing value), the `readonly` flag must be unset before the operation:

# %% {"editable": true, "slideshow": {"slide_type": ""}}
Z.meta.readonly = False
Z.meta.xxx = "permitted"
Z.meta

# %% [markdown] {"editable": true, "slideshow": {"slide_type": ""}}
# It is advised to set back the `readonly` flag to True.
