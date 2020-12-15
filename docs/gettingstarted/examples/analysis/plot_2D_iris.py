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
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %matplotlib inline

# %%

# ======================================================================================================================
#  Copyright (©) 2015-2020 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

# %% [markdown]
"""
2D-IRIS analysis example
=========================

In this example, we perform the 2D IRIS analysis of CO adsorption on a sulfide catalyst.
"""

# %%
import os

import spectrochempy as scp

# %% [markdown]
# Upload dataset

# %%
X = scp.NDDataset.read_omnic(os.path.join('irdata', 'CO@Mo_Al2O3.SPG'))

# %% [markdown]
# X has two coordinates: `wavenumbers` named "x" and `timestamps` (i.e. the time of recording)
# named "y".
# %%
print(X.coordset)

# %% [markdown]
# We want to replace the timestamps ("y") by pressure coordinates.
#
# **Note**: To replace a coordinate always use its name not the index (i.e. "y" in the present case) or `update`
# method. See the API reference or our User Guide for more information on this.

# %%
p = [0.00300, 0.00400, 0.00900, 0.01400, 0.02100, 0.02600, 0.03600,
     0.05100, 0.09300, 0.15000, 0.20300, 0.30000, 0.40400, 0.50300,
     0.60200, 0.70200, 0.80100, 0.90500, 1.00400]

X.y = scp.Coord(p, title='pressure', units='torr')

# %% [markdown]
# Select and plot the spectral range of interest
# %%
X_ = X[:, 2250.:1950.]
X_.plot()

# %% [markdown]
# Perform IRIS without regularization (the verbose flag can be set to True to have information on the running process)
# %%
param = {
        'epsRange': [-8, -1, 50],
        'kernel': 'langmuir'
        }

iris = scp.IRIS(X_, param, verbose=True)

# %% [markdown]
# Plots the results
# %%
iris.plotdistribution()
iris.plotmerit()

# %% [markdown]
# Perform  IRIS with regularization, manual search
# %%
param = {
        'epsRange': [-8, -1, 50],
        'lambdaRange': [-10, 1, 12],
        'kernel': 'langmuir'
        }

iris = scp.IRIS(X_, param, verbose=True)
iris.plotlcurve()
iris.plotdistribution(-7)
iris.plotmerit(-7)

# %% [markdown]
# Now try an automatic search of the regularization parameter:

# %%
param = {
        'epsRange': [-8, -1, 50],
        'lambdaRange': [-10, 1],
        'kernel': 'langmuir'
        }

iris = scp.IRIS(X_, param, verbose=True)
iris.plotlcurve()
iris.plotdistribution(-1)
iris.plotmerit(-1)

scp.show()  # uncomment to show plot if needed (not necessary in jupyter notebook)
