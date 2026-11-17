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
color the scores by species. See the :doc:`PCA tutorial </userguide/analysis/pca>`
for component selection and interpretation.
"""

import spectrochempy as scp

flowers = scp.load_iris()
pca = scp.PCA(n_components=2)
_ = pca.fit(flowers)
_ = pca.plot_score(components=(1, 2), color_mapping="labels")
