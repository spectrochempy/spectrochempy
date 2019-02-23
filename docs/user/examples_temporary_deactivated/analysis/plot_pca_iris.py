# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2019 LCS
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

S, LT = pca.transform(n_pc='auto')

print(LT)

###############################################################@
# Finally, display the results graphically

#TODO: make the following work!
#_ = pca.screeplot()
#_ = pca.scoreplot(1, 2, color_mapping='labels')
#_ = pca.scoreplot(1, 2, 3, color_mapping='labels')

