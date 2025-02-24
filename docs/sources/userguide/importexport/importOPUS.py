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
# ## Import of Bruker OPUS files
#
# [Bruker OPUS](https://www.bruker.com/en/products-and-solutions/infrared-and-raman/opus-spectroscopy-software.html)
# files have also a proprietary file format. The Opus reader (`read_opus()` ) of spectrochempy is essentially a wrapper of the python module
# [brukeropus](https://github.com/joshduran/brukeropus) developed by Josh Duran.
# The use of `read_opus()` is similar to that of  `read_omnic()`.
#
# Hence, one can open sample Opus files contained in the `datadir` using:

# %%
import spectrochempy as scp

Z = scp.read_opus("irdata/OPUS/test.0002")
Z

# %% [markdown]
# For multifile loading, one can use:

# %%
Z1 = scp.read_opus("test.0000", "test.0001", "test.0002", directory="irdata/OPUS")
Z1

# %% [markdown]
# By default all files in a directory are merged in a single dataset if they are compatibles (same shape, same type of experiment).
# If they cannot be merged, separate datasets are returned with dataset merged in differnt groups.

# %%
LZ1 = scp.read_opus(
    "test.0000", "test.0001", "test.0002", "background.0", directory="irdata/OPUS"
)
LZ1

# %% [markdown]
# or, if all files in a directory must be read:

# %%
LZ = scp.read_opus("irdata/OPUS")
LZ

# %% [markdown]
# By default all files in a directory are merged if they are compatible.
# In this case they cannot automatically merged in a single datasets because .
# So compatible files are merged (4 of them) but one remain unmerged.
#
# If one desire to load of files into separate datasets, then set the merge attribute to False.

# %%
LZ1 = scp.read_opus("irdata/OPUS", merge=False)
LZ1

# %% [markdown]
# <div class='alert alert-info'>
# <b>Note</b>
#
# By default absortion spectra (AB) are load, if present in the file.
#
# Opus file however can contain several files and they can be retrieved eventually using the correct type in the call to `read_opus`:
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
#
# </div>

# %% [markdown]
# It is possible to know which are the other types availables in the original file:

# %%
Z.meta.other_data_types

# %% [markdown]
# Thus if one wants to load the single-channel sample spectra, the read function syntax would be:

# %%
ZSM = scp.read_opus("irdata/OPUS/test.0002", type="SM")
print(ZSM)

# %% [markdown]
# ### Reading OPUS file Metadata
#
# As just seen above, more informations can be obtained on the experiment and spectrometer parameters using the dataset metadata (`meta`attribute).
#
# For instance:

# %%
Z.meta

# %%
metadata = Z.meta

# %% [markdown]
# The metadata object is a readonly dictionary-like object (which can be nested, i.e., it can contains other metadata objects).
# It can be accessed as follows:
#
# - List parameter blocks presents at the first level in the metadata (with the corresponding access key)

# %%
for k, v in metadata.items():
    print(k, v)

    # print(f"* {v.name} [{k}]")

# %% [markdown]
# - Access and list the blocks contained in `params`

# %%
for (
    kk,
    vv,
) in (
    metadata.params.items()
):  # note here that we use the block key as an attribute of the metadata object
    print(f"* {vv.name} [{kk}]")

# %% [markdown]
#   Note: if you just need the keys but not the actual name of the block, it is simpler to do this:

# %%
print(metadata.params.keys())

# %%
print(metadata.params)

# %% [markdown]
# - Access parameters of a given block

# %%

# %% [markdown]
# ## Import/Export of JCAMP-DX files
#
# [JCAMP-DX](http://www.jcamp-dx.org/) is an open format initially developed for IR
# data and extended to
# other spectroscopies. At present, the JCAMP-DX reader implemented in Spectrochempy is
# limited to IR data and
# AFFN encoding (see R. S. McDonald and Paul A. Wilks, JCAMP-DX: A Standard Form for
# Exchange of Infrared Spectra in
# Readable Form, Appl. Spec., 1988, 1, 151–162. doi:10.1366/0003702884428734 for
# details).
#
# The JCAMP-DX reader of spectrochempy has been essentially written to read again the
# jcamp-dx files exported by
# spectrochempy `write_jdx()` writer.
#
# Hence, for instance, the first dataset can be saved in the JCAMP-DX format:

# %%
S0 = X[0]
print(S0)
S0.write_jcamp("CO@Mo_Al2O3_0.jdx", confirm=False)

# %% [markdown]
# Then used (and maybe changed) by a 3rd party software, and re-imported in
# spectrochempy:

# %%
newS0 = scp.read_jcamp("CO@Mo_Al2O3_0.jdx")
print(newS0)


# %% [markdown]
# It is important to note here that the conversion to JCAMP-DX changes the last digits
# of absorbance and wavenumbers:


# %%
def difference(x, y):
    from numpy import abs
    from numpy import max

    nonzero = y.data != 0
    error = abs(x.data - y.data)
    max_relative_error = max(error[nonzero] / abs(y.data[nonzero]))
    return max(error), max_relative_error


# %%
max_error, max_rel_error = difference(S0, newS0)
print(f"Max absolute difference in absorbance: {max_error:.3g}")
print(f"Max relative difference in absorbance: {max_rel_error:.3g}")

# %%
max_error, max_rel_error = difference(S0.x, newS0.x)
print(f"Max absolute difference in wavenumber: {max_error:.3g}")
print(f"Max relative difference in wavenumber: {max_rel_error:.3g}")

# %% [markdown]
# But this is much beyond the experimental accuracy of the data and has
