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
# %%
# Import the spectrochempy API package
import spectrochempy as scp

# %%
# Load a dataset

dataset = scp.read_omnic("irdata/nh4y-activation.spg")
dataset = dataset[:, 2000.0:4000.0]  # remember float number to slice from coordinates
print(dataset)
dataset.plot_stack()

# %%
# Create a PCA object
pca = scp.PCA(dataset, centered=False)

# %%
# Reduce the dataset to a lower dimensionality (number of
# components is automatically determined)

S, LT = pca.reduce(n_pc=0.99)

print(LT)

# %%
# Finally, display the results graphically
# ScreePlot
_ = pca.screeplot()

# %%
# Score Plot
_ = pca.scoreplot(1, 2)

# %%
# Score Plot for 3 PC's in 3D
_ = pca.scoreplot(1, 2, 3)

# %% labeling scoreplot with spectra labels
# Our dataset has already two columns of labels for the spectra but there are little
# too long for display on plots.
S.y.labels

# %%
# So we define some short labels for each component, and add them as a third column:
labels = [lab[:6] for lab in dataset.y.labels[:, 1]]
# we cannot change directly the label as S is read-only, but use the method `labels`
pca.labels(labels)  # Note this does not replace previous labels, but adds a column.

# %%
# now display thse
_ = pca.scoreplot(1, 2, show_labels=True, labels_column=2, labels_every=5)

# %%
# Displays the 4-first loadings

LT[:4].plot_stack()

scp.show()  # uncomment to show plot if needed (not necessary in jupyter notebook)
