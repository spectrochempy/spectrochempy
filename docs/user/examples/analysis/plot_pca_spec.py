# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================
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
dataset.plot_stack()

##############################################################
# Create a PCA object
pca = scp.PCA(dataset, centered=False)

##############################################################
# Reduce the dataset to a lower dimensionality (number of
# components is automatically determined)

S, LT = pca.reduce(n_pc=.99)

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
# Displays the 4-first loadings

LT[:4].plot_stack()

#show() # uncomment to show plot if needed()