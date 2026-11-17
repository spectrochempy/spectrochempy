# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: D205, D400, D415
"""
Compare iris measurements with PCA
==================================

Project the four iris measurements onto two principal components and
color the scores by species. The input plot shows petal size in the
original units. See the :doc:`PCA tutorial </userguide/analysis/pca>`
for component selection and interpretation.
"""

import matplotlib.pyplot as plt

import spectrochempy as scp

flowers = scp.load_iris()
# %%
# View two of the four measured features. Each point is one flower.

_, ax = plt.subplots()
species = flowers.y.labels
for name in sorted(set(species)):
    samples = flowers[species == name]
    _ = ax.scatter(
        samples[:, "petal_length"].data.ravel(),
        samples[:, "petal_width"].data.ravel(),
        label=name,
        alpha=0.7,
    )
_ = ax.set(xlabel="Petal length / cm", ylabel="Petal width / cm")
_ = ax.legend()

# %%
# PCA projects all four measured features onto two components.

pca = scp.PCA(n_components=2)
_ = pca.fit(flowers)
_ = pca.plot_score(components=(1, 2), color_mapping="labels")
