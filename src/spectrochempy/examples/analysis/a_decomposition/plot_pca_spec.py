# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa: D205, D400, D415
"""
Compare IR spectra with PCA scores
==================================

Mask a saturated band in the NH₄Y activation series, fit a PCA model, and
compare the spectra in a score plot.
See the :doc:`PCA tutorial </userguide/analysis/pca>` for the effect of
masking, the explained variance, and the interpretation of scores and loadings.
"""

import spectrochempy as scp

spectra = scp.read_omnic("irdata/nh4y-activation.spg")[::5]
spectra[:, 882.0:1280.0] = scp.MASKED

pca = scp.PCA(n_components=0.999)
_ = pca.fit(spectra)
_ = pca.plot_score(components=(1, 2))
