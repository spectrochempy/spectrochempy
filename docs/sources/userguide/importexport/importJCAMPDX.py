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
# [JCAMP-DX](http://www.jcamp-dx.org/) is an open format initially developed for IR
# data and extended to other spectroscopies. At present, the JCAMP-DX reader implemented
# in SpectroChemPy is limited to IR data and AFFN encoding (see <cite data-cite="mcdonald:1988">McDonald and Wilks, 1988</cite>.
#
# The JCAMP-DX reader of SpectroChemPy has been essentially written to read JCAMP-DX files
# exported by the SpectroChemPy `write_jdx()` writer.
#

# %%
import spectrochempy as scp

X = scp.read_omnic("irdata//CO@Mo_Al2O3.SPG")
S0 = X[0]
S0

# %%
S0.write_jcamp("CO@Mo_Al2O3_0.jdx", confirm=False)

# %% [markdown]
# Then used (and maybe changed) by a 3rd party software, and re-imported in
# spectrochempy:

# %%
newS0 = scp.read_jcamp("CO@Mo_Al2O3_0.jdx")
newS0

# %% [markdown]
# It is important to note here that the conversion to JCAMP-DX changes the last digits
# of absorbance and wavenumbers:


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
