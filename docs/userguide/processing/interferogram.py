# -*- coding: utf-8 -*-
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
#       jupytext_version: 1.9.1
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

# %%
import spectrochempy as scp
from spectrochempy.units import ur

# %% [markdown]
# ## FTIR interferogram processing
#
# A situation where we need transform of real data is the case of FTIR interferograms.

# %%
ir = scp.read_spa("irdata/interferogram/interfero.spa")
ir.dc(inplace=True)
_ = ir.plot()

# %%
_ = ir.plot(xlim=(0, 0.25))

# %% [markdown]
# Note that the time scale of the interferogram has been calculated using a laser frequency of 15798.26 cm$^{-1}$. If this is not correct you can change it using the `set_laser_frequency` coordinate method:

# %%
ir.x.set_laser_frequency(15798.26 * ur('cm^-1'))

# %% [markdown]
# Now we can perform the Fourier transform. By default one zero-filling level is applied prior the Fourier transform.

# %%
ird = ir.dc()
#ird = ird.hamming()
irt = ird.fft()
_ = irt.plot(xlim=(3999, 400))

# %% [markdown]
# ### Comparison with the OMNIC processing. 
#
# Here we compare the OMNIC processed spectra of the same interferogram and ours in red. One can see that the results are very close

# %%
irs = scp.read_spa("irdata/interferogram/spectre.spa")
_ = irs.plot(lw=1)
_ = irt.plot(c='red', clear=False, xlim=(3999, 400))
