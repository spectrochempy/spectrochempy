# -*- coding: utf-8 -*-
#
# =============================================================================
# Copyright (©) 2015-2018 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# =============================================================================
"""
PCA analysis example
---------------------
In this example, we perform the PCA dimensionality reduction of the classical
``iris`` dataset.

"""

from spectrochempy import api

############################################################
# upload a dataset form a distant server

dataset = api.upload_IRIS()
print(dataset)

##############################################################
# create a PCA object
pca = api.PCA(dataset, centered=True)

##############################################################
# reduce the dataset to a lower dimensionality (number of
# components is automatically determined)

LT, S = pca.transform(n_pc='auto')

print(LT)

###############################################################@
# display the results graphically

_ = pca.screeplot()
_ = pca.scoreplot(1, 2, color_mapping='labels')
_ = pca.scoreplot(1,2,3, color_mapping='labels')

api.show()

