# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: title,-all
#     formats: ipynb,py:percent
#     notebook_metadata_filter: all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3
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
#     version: 3.9.1
#   widgets:
#     application/vnd.jupyter.widget-state+json:
#       state: {}
#       version_major: 2
#       version_minor: 0
# ---

# %% [markdown]
# # FTIR interferogram processing
#
# A situation where we need transform of real data is the case of FTIR interferogram.

# %%
import spectrochempy as scp
from spectrochempy.core.units import ur

# %%
ir = scp.read_spa("irdata/interferogram/interfero.SPA")

# %% [markdown]
# By default, the interferogram is displayed with an axis in points (no units).

# %%
prefs = ir.preferences
prefs.figure.figsize = (7, 3)
ir.plot()
print("number of points = ", ir.size)

# %% [markdown]
# Plotting a zoomed region around the maximum of the interferogram
# (the so-called `ZPD`: `Zero optical Path Difference` ) we can see that it is located
# around the 64th points. The FFT processing will need this information,
# but it will be determined automatically.

# %%
ir.plot(xlim=(0, 128))

# %% [markdown]
# The `x` scale of the interferogram can also be displayed as a function of optical
# path difference. For this we just make `show_datapoints` to False:

# %%
ir.x.show_datapoints = False
ir.plot(xlim=(-0.04, 0.04))

# %% [markdown]
# Note that the `x` scale of the interferogram has been calculated using the laser
# frequency indicated in the original omnic file. It is stored in the `meta` attribute
# of the NDDataset:

# %%
print(ir.meta.laser_frequency)

# %% [markdown]
# If absent, it can be set using the `set_laser_frequency()` method, e.g.:

# %%
ir.x.set_laser_frequency(15798.26 * ur("cm^-1"))

# %% [markdown]
# Now we can perform the Fourier transform. By default, no zero-filling level is applied
# prior the Fourier transform for FTIR. To add some level of zero-filling,
# use the `zf` method.

# %%
ird = ir.dc()
ird = ird.zf(size=2 * ird.size)
irt = ird.fft()

irt.plot(xlim=(3999, 400))

# %% [markdown]
# A `Happ-Genzel` (Hamming window) apodization can also be applied prior to the
# Fourier transformation in order to decrease the H2O narrow bands.

# %%
ird = ir.dc()
irdh = ird.hamming()
irdh.zf(inplace=True, size=2 * ird.size)
irth = irdh.fft()
irth.plot(xlim=(3999, 400))

# %% [markdown]
# ## Comparison with the OMNIC processing.
#
# Here we compare the OMNIC processed spectra of the same interferogram and ours in red.
# One can see that the results are very close

# %%
irs = scp.read_spa("irdata/interferogram/spectre.SPA")
prefs.figure.figsize = (7, 6)
ax = irs.plot(label="omnic")
(irt + 0.4).plot(c="red", linestyle="solid", clear=False, label="no hamming")
(irth + 0.2).plot(c="green", linestyle="solid", clear=False, label="hamming")
ax.set_xlim(4000.0, 400.0)
ax.set_ylim(0.0, 3.1)
_ = ax.legend()
