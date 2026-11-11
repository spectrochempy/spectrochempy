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
# # Import/Export of JCAMP-DX files
#
# [JCAMP-DX](https://artefacts.ceda.ac.uk/formats/jcamp_dx/) is an open format
# used by several spectroscopies. Use `scp.read_jcamp()` or `scp.jcamp.read()`
# to open infrared spectra and supported processed NMR spectra in 1D and 2D.
# The reader selects the workflow from the data type declared in the file. If that type
# is ambiguous, use `jcamp_kind="ir"` or `jcamp_kind="nmr"`.
#
# The IR reader supports AFFN-encoded spectra on a regular grid, including
# linked series on a shared grid. Reading processed NMR spectra requires
# the [SpectroChemPy NMR plugin](https://www.spectrochempy.fr/projects/nmr/latest/getting_started/installation.html).
# For a supported real 2D spectrum stored in JCAMP-DX NTUPLES, install
# nmrglue-ng 1.3 or later and call `scp.read_jcamp("cosy.jdx")`. The reader
# returns a 2D dataset with the F1 and F2 axes from the file.
#
# The example below shows how to export and read back an infrared spectrum.
#

# %%
import spectrochempy as scp

X = scp.read_omnic("irdata//CO@Mo_Al2O3.SPG")
S0 = X[0]
S0

# %%
S0.write_jcamp("CO@Mo_Al2O3_0.jdx", confirm=False)

# %% [markdown]
# The namespace API is also available for writing:

# %%
scp.jcamp.write(S0, "CO@Mo_Al2O3_0.jdx", confirm=False, overwrite=True)

# %% [markdown]
# Then used (and maybe changed) by a 3rd party software, and re-imported in
# spectrochempy:

# %%
newS0 = scp.read_jcamp("CO@Mo_Al2O3_0.jdx")
newS0

# %% [markdown]
# The namespace API works identically:

# %%
newS0 = scp.jcamp.read("CO@Mo_Al2O3_0.jdx")
newS0

# %% [markdown]
# It is important to note here that the conversion to JCAMP-DX changes the last digits
# of absorbance and wavenumbers:

# %% [markdown]
# The writer preserves the numeric values it writes. It does not silently convert
# wavelength to wavenumber or transmittance to absorbance. Current export support
# is therefore limited to exact-scale JCAMP mappings:
#
# - x: `cm^-1`, `um`, `nm`, or no unit (`ARBITRARY UNITS`);
# - y: `absorbance`, `transmittance`, or no unit (`ARBITRARY UNITS`).
#
# Other merely convertible or named arbitrary units are rejected until you
# convert them explicitly before export.


# %%
from spectrochempy.utils.compare import difference

# %%
max_error, max_rel_error = difference(S0, newS0)
print(f"Max absolute difference in absorbance: {max_error:.3g}")
print(f"Max relative difference in absorbance: {max_rel_error:.3g}")

# %%
max_error, max_rel_error = difference(S0.x, newS0.x)
print(f"Max absolute difference in wavenumber: {max_error:.3g}")
print(f"Max relative difference in wavenumber: {max_rel_error:.3g}")

# %% [markdown]
# But this is much beyond the experimental accuracy of the data.
