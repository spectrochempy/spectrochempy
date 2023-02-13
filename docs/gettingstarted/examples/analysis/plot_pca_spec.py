# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# flake8: noqa
"""
NDDataset PCA analysis example
-------------------------------
In this example, we perform the PCA dimensionality reduction of a spectra
dataset

"""

import spectrochempy as scp

############################################################
# Load a dataset

dataset = scp.read_omnic("irdata/nh4y-activation.spg")
print(dataset)
_ = dataset.plot()

""
dataset[
    :, 882.0:1280.0
] = scp.MASKED  # remember: use float numbers for slicing (not integer)
_ = dataset.plot()

##############################################################
# Create a PCA object
pca = scp.PCA(dataset, centered=False)

##############################################################
# Reduce the dataset to a lower dimensionality (number of
# components is automatically determined: 3 in this case)

S, LT = pca.reduce(n_pc=0.999)

print(LT)

###############################################################
# Finally, display the results graphically
# ScreePlot
_ = pca.screeplot()

########################################################################################################################
# Score Plot
_ = pca.scoreplot(1, 2)

########################################################################################################################
# Score Plot for 3 PC's in 3D
_ = pca.scoreplot(1, 2, 3)

##############################################################################
# Displays the 3 loadings

_ = LT.plot(legend=True)

# uncomment the line below to see plot if needed (not necessary in jupyter notebook)
# scp.show()

""
