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

from spectrochempy import core

############################################################
# Upload a dataset form a distant server

dataset = core.upload_IRIS()
print(dataset)

##############################################################
# Create a PCA object
pca = core.PCA(dataset, centered=True)

##############################################################
# Reduce the dataset to a lower dimensionality (number of
# components is automatically determined)

LT, S = pca.transform(n_pc='auto')

print(LT)

###############################################################@
# Finally, display the results graphically

_ = pca.screeplot()
_ = pca.scoreplot(1, 2, color_mapping='labels')
_ = pca.scoreplot(1,2,3, color_mapping='labels')

core.show()

