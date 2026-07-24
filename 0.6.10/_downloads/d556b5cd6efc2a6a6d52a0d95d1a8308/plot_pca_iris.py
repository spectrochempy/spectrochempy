# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# flake8: noqa
"""
PCA example (iris dataset)
--------------------------
In this example, we perform the PCA dimensionality reduction of the classical `iris`
dataset (Ronald A. Fisher.
"The Use of Multiple Measurements in Taxonomic Problems. Annals of Eugenics, 7, pp.179-188, 1936).

"""
# %%
# First we laod the spectrochempy API package
import spectrochempy as scp

# %%
# load a dataset from scikit-learn
dataset = scp.load_iris()

# %%
# Create a PCA object
# Here, the number of components  wich is used by the model is automatically determined
# using `n_components="mle"`\. Warning: `mle` cannot be used when
# n_observations < n_features.
pca = scp.PCA(n_components="mle")
# %%
# Fit dataset with the PCA model
pca.fit(dataset)
# %%
# The number of components found is 3:
pca.n_components
# %%
# It explain 99.5 % of the variance
pca.cumulative_explained_variance[-1].value
# %%
# We can also specify the amount of explained variance to compute how much components
# are needed (a number between 0 and 1 for n_components is required to do this).
# we found 4 components in this case
pca = scp.PCA(n_components=0.999)
pca.fit(dataset)
pca.n_components
# %%
# the 4 components found are in the `components` attribute of pca. These components are
# often called loadings in PCA analysis.
loadings = pca.components
loadings
# %%
# Note: it is equivalently possible to use the `loadings` attribute of pca, which
# produce the same results.
pca.loadings
# %%
# To Reduce the data to a lower dimensionality using these three components, we use the
# transform methods. The results is often called `scores` for PCA analysis.
scores = pca.transform()
scores
# %%
# Again, we can also use the `scores` attribute to get this results
scores = pca.scores
scores
# %%
# The figures of merit (explained and cumulative variance) confirm that
# these 4 PC's explain 100% of the variance:
#
pca.printev()
# %%
# These figures of merit can also be displayed graphically
#
# The ScreePlot
_ = pca.screeplot()
# %%
# The score plots can be used for classification purposes. The first one - in 2D for the
# 2 first PC's - shows that the first PC allows distinguishing Iris-setosa (score of
# PC#1 < -1) from other species (score of PC#1 > -1), while more PC's are required
# to distinguish versicolor from viginica.
_ = pca.scoreplot(scores, 1, 2, color_mapping="labels")
# %%
# The second one - in 3D for the 3 first PC's - indicates that a thid PC won't allow
# better distinguishing versicolor from viginica.
ax = pca.scoreplot(scores, 1, 2, 3, color_mapping="labels")
ax.view_init(10, 75)

# %%
# This ends the example ! The following line can be uncommented if no plot shows when
# running the .py script with python

# scp.show()
