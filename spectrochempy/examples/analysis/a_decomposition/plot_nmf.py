# -*- coding: utf-8 -*-
# %%
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# flake8: noqa
"""
NMF analysis example
====================

"""
# %%
# Import the spectrochempy API package
import spectrochempy as scp

# %%
# Prepare the dataset to NMF factorize
# -------------------------------------

# %%
# Here we use a FTIR dataset corresponding the dehydration of a NH4Y zeolite
# and recorded in the OMNIC format.
dataset = scp.read_omnic("irdata/nh4y-activation.spg")

# %%
# Mask some columns (features) wich correspond to saturated part of the spectra.
# Note taht we use float number for defining the limits for masking as coordinates
# (integer numbers would mean point index and s would lead t incorrect results)
dataset[:, 882.0:1280.0] = scp.MASKED

# %%
# Make sure all data are positive. For this we use the math fonctionalities of NDDataset
# objects (:meth:`min` function to find the minimum value of the dataset
# and the `-` operator for subtrating this value to all spectra of the dataset.
dataset -= dataset.min()

# %%
# Plot it for a visual check
_ = dataset.plot()

# %%
# Create a NMF object
# -------------------
#
# As argument of the object constructor we define log_level to ``"INFO"`` to
# obtain verbose output during fit, and we set the number of component to use at 4.
model = scp.NMF(n_components=4, log_level="INFO")

# %%
# Fit the model
# -------------
_ = model.fit(dataset)

# Get the results
# ---------------
#
# The concentration :math:`C` and the transposed matrix of spectra :math:`S^T` can
# be obtained as follow
C = model.transform()
St = model.components


# %%
# Plot results
# ------------
_ = C.T.plot(title="Concentration", colormap=None, legend=C.k.labels)

# %%
#
m = St.ptp()
for i in range(St.shape[0]):
    St.data[i] -= i * m / 2
ax = St.plot(title="Components", colormap=None, legend=St.k.labels)
ax.set_yticks([])

# %%
# This ends the example ! The following line can be uncommented if no plot shows when
# running the .py script with python

# scp.show()
