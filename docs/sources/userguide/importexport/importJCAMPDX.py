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
# JCAMP-DX is a text exchange format used across several spectroscopies.
# Files with the same extension (`.dx` or `.jdx`) can contain different
# kinds of spectral data. Use `scp.jcamp.read()` to read an existing file.
#
# ```{=rst}
# See :cite:t:`mcdonald:1988` for the original infrared exchange format.
# ```

# %% [markdown]
# ## Check the technique and the data table
#
# The reader selects a route from the `DATA TYPE` header. The table below
# describes which representations SpectroChemPy accepts for each technique.
#
# | Technique | Supported data | Limits |
# | --- | --- | --- |
# | Infrared | `XYDATA` with a regular spectral grid, including AFFN and DIFDUP encodings | Linked (`LINK`) spectra must share their grid and units. |
# | Raman | One explicit ASCII X/Y table in `XYDATA` or `XYPOINTS`, with Raman shift in cm⁻¹ | Compressed tables and linked containers are outside this profile. |
# | UV–visible | One explicit ASCII X/Y table in `XYDATA` or `XYPOINTS`, with wavelength in nm | Compressed tables and linked containers are outside this profile. |
# | Processed NMR 1D | One real spectrum in a regular `X++(Y..Y)` table, with the NMR plugin | The profile excludes raw FIDs and peak lists. |
# | Processed NMR 2D | One supported real F1/F2 plane in an `NTUPLES` spectrum | Requires `spectrochempy-nmr` 0.4.0 or later. Complex or hypercomplex planes and higher-dimensional spectra are outside this profile. |
#
# For Raman and UV–visible spectra the reader keeps the supplied coordinates,
# including irregular grids. The number of X/Y pairs must match `NPOINTS`.
# Reading several files gives separate spectra, even with `merge=True`.
# For UV–visible files, absorbance and transmittance are recognized from
# `YUNITS`. For Raman, the ordinate label is kept as text.
#
# ### Choose the route when the header is ambiguous
#
# When the file does not declare a recognized technique and you know it
# from the instrument or export settings, pass `jcamp_kind` explicitly:
#
# ```python
# import spectrochempy as scp
#
# spectrum = scp.jcamp.read("spectrum.dx", jcamp_kind="raman")
# ```
#
# Supported choices are `"auto"` (the default), `"ir"`, `"raman"`,
# `"uvvis"`, and `"nmr"`. An explicit choice must agree with a recognized
# declaration. A linked file with conflicting technique declarations is
# rejected. If the table layout or units are unsupported, re-export from
# the instrument software using a supported representation.
#
# ### Read processed NMR data
#
# Install a compatible [SpectroChemPy NMR plugin](https://www.spectrochempy.fr/projects/nmr/latest/getting_started/installation.html).
# The public reader delegates to the plugin and returns an `NDDataset`.
# A supported 2D file preserves the F1 and F2 coordinates from the
# exported plane. See the [NMR reading guide](https://www.spectrochempy.fr/projects/nmr/latest/userguide/reading_data.html)
# for other acquisition and exchange formats.

# %% [markdown]
# ### Read a processed NMR spectrum
#
# With a compatible SpectroChemPy NMR plugin (version 0.3 or later),
# the reader recognizes the NMR data type declared in this synthetic JCAMP-DX
# file and returns a spectrum on a chemical-shift axis in ppm.

# %%
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

from packaging.version import Version

import spectrochempy as scp

try:
    nmr_version = Version(version("spectrochempy-nmr"))
except PackageNotFoundError:
    nmr_version = None

if nmr_version is not None and nmr_version >= Version("0.3"):
    nmr_jcamp = """##TITLE=Synthetic proton spectrum
##JCAMP-DX=5.01
##DATA TYPE=NMR SPECTRUM
##DATA CLASS=XYDATA
##.OBSERVE FREQUENCY=400
##.OBSERVE NUCLEUS=^1H
##XUNITS=PPM
##YUNITS=ARBITRARY UNITS
##FIRSTX=8
##LASTX=0
##NPOINTS=9
##XFACTOR=1
##YFACTOR=1
##XYDATA=(X++(Y..Y))
8 0 1 2 10 2 1 6 1 0
##END=
"""
    with TemporaryDirectory() as directory:
        filename = Path(directory) / "proton.jdx"
        filename.write_text(nmr_jcamp, encoding="ascii")
        proton = scp.read_jcamp(filename)

    _ = proton.plot()

# %% [markdown]
# ## Export and read back an infrared spectrum
#
# The JCAMP-DX writer exports real infrared spectral data. The read support
# described above includes techniques beyond the writer's scope. For other
# export options, see [Export and share data](export.rst).

# %%
import spectrochempy as scp

X = scp.read_omnic("irdata//CO@Mo_Al2O3.SPG")
S0 = X[0]
S0

# %%
_ = S0.write_jcamp("CO@Mo_Al2O3_0.jdx", confirm=False)

# %% [markdown]
# The namespace API is also available for writing:

# %%
_ = scp.jcamp.write(S0, "CO@Mo_Al2O3_0.jdx", confirm=False, overwrite=True)

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
