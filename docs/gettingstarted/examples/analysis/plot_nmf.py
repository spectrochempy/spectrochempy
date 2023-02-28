# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# flake8: noqa
"""
NDDataset NMF analysis example
-------------------------------

"""
# %%
# Import the spectrochempy API package
import spectrochempy as scp

# %%
# Prepare the dataset to NMF factorize

# %%
# Load a dataset
dataset = scp.read_omnic("irdata/nh4y-activation.spg")

# %%
# mask some columns (features)
dataset[:, 882.0:1280.0] = scp.MASKED

# %%
# make sure all data are positive
dataset -= dataset.min()

# %%
# plot it
_ = dataset.plot()

# %%
# Create a NMF object, fit the dataset and extract the C and St matrices
nmf_model = scp.NMF(
    used_components=4, init="random", random_state=12345, log_level="INFO"
)
C = nmf_model.fit_transform(dataset)
St = nmf_model.components


# %%
# plot separated line
C.T.plot(title="Concentration", colormap=None, legend=C.x.labels)

m = St.ptp()
for i in range(St.shape[0]):
    St.data[i] -= i * m / 2
St.plot(title="Components", colormap=None, legend=St.y.labels)

# %%
# uncomment the line below to see plot if needed (not necessary in jupyter notebook)
scp.show()
