# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.7.1
# ---

# %%

# %%

# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

# %% [markdown]
"""
Sine bell and squared Sine bell window multiplication
=====================================================

In this example, we use sine bell or squared sine bell window multiplication to apodize a NMR signal in the time domain.
"""

# %%
import spectrochempy as scp
import os

path = os.path.join(scp.preferences.datadir, 'nmrdata', 'bruker', 'tests', 'nmr', 'topspin_1d')
dataset1D = scp.read_topspin(path, expno=1, remove_digital_filter=True)

# %% [markdown]
# Normalize the dataset values and reduce the time domain

# %%
dataset1D /= dataset1D.real.data.max()  # normalize
dataset1D = dataset1D[0.:15000.]

# %% [markdown]
# Apply Sine bell window apodization with parameter ssb=2, which correspond to a cosine function

# %%
new1, curve1 = scp.sinm(dataset1D, ssb=2, retfunc=True, inplace=False)

# this is equivalent to
_ = dataset1D.sinm(ssb=2, retfunc=True, inplace=False)

# or also
_ = scp.sp(dataset1D, ssb=2, pow=1, retfunc=True, inplace=False)

# %% [markdown]
# Apply Sine bell window apodization with parameter ssb=2, which correspond to a sine function

# %%
new2, curve2 = dataset1D.sinm(ssb=1, retfunc=True, inplace=False)

# %% [markdown]
# Apply Squared Sine bell window apodization with parameter ssb=1 and ssb=2

# %%
new3, curve3 = scp.qsin(dataset1D, ssb=2, retfunc=True, inplace=False)

new4, curve4 = dataset1D.qsin(ssb=1, retfunc=True, inplace=False)

# %% [markdown]
# Apply shifted Sine bell window apodization with parameter ssb=8 (mixed sine/cosine window)

# %%
new5, curve5 = dataset1D.sinm(ssb=8, retfunc=True, inplace=False)

# %% [markdown]
# Plotting

# %%
p = dataset1D.plot(zlim=(-2, 2), color='k')

curve1.plot(color='r', clear=False)
new1.plot(data_only=True, color='r', clear=False, label=' sinm with ssb= 2 (cosine window)')

curve2.plot(color='b', clear=False)
new2.plot(data_only=True, color='b', clear=False, label=' sinm with ssb= 1 (sine window)')

curve3.plot(color='m', clear=False)
new3.plot(data_only=True, color='m', clear=False, label=' qsin with ssb= 2')

curve4.plot(color='g', clear=False)
new4.plot(data_only=True, color='g', clear=False, label=' qsin with ssb= 1')

curve5.plot(color='c', ls='--', clear=False)
new5.plot(data_only=True, color='c', ls='--', clear=False, label=' sinm with ssb= 8', legend='best')

# scp.show()  # uncomment to show plot if needed (not necessary in jupyter notebook)
