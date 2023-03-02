# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# flake8: noqa
"""
NDDataset FastICA analysis example
-------------------------------

"""
# %%
# Import the spectrochempy API package
import spectrochempy as scp

# %%
# Load a dataset
dataset = scp.read_omnic("irdata/nh4y-activation.spg")
dataset[:, 882.0:1280.0] = scp.MASKED
# dataset -= dataset[-1]  # difference spectra
print(dataset)
_ = dataset.plot()

# %%
# Create a FastICA object and fit the dataset
X = dataset.T
ica = scp.FastICA(used_components=6)
S = ica.fit_transform(X)  # Reconstruct signals

# %%
# plot separated line
m = S.ptp()
ST = S.T
for i in range(ST.shape[0]):
    ST.data[i] -= i * m / 2

ST.plot(colormap=None, legend=True)
# %%
# uncomment the line below to see plot if needed (not necessary in jupyter notebook)

scp.show()
