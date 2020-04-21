# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================
"""
PCA analysis example
---------------------
In this example, we perform the PCA dimensionality reduction of the classical
``iris`` dataset.

"""

import spectrochempy as scp

############################################################
# Upload a dataset form a distant server

dataset = scp.upload_IRIS()

##############################################################
# Create a PCA object
pca = scp.PCA(dataset, centered=True)

##############################################################
# Reduce the dataset to a lower dimensionality (number of
# components is automatically determined)

S, LT = pca.reduce(n_pc='auto')

print(LT)

###############################################################@
# Finally, display the results graphically
#
# ScreePlot
_ = pca.screeplot()

########################################################################################################################
# ScorePlot of 2 PC's
_ = pca.scoreplot(1, 2, color_mapping='labels')

########################################################################################################################
# or in 3D for 3 PC's
_ = pca.scoreplot(1, 2, 3, color_mapping='labels')

scp.show() # uncomment to show plot if needed()