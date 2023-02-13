# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
"""
This module implement several wrapper to scikit-learn model and estimators
"""

from sklearn.decomposition import PCA as _PCA

from spectrochempy.analysis.pca import PCA


class SKL_PCA(PCA):
    """
    PCA analysis is here done using the sklearn PCA estimator.

    We just implement fit, reduce, reconstruct and fit_reconstruct
    """

    def __init__(
        self,
        *,
        log_level="WARNING",
        config=None,
        warm_start=False,
        **kwargs,
    ):

        # call the super class for initialisation
        super().__init__(
            log_level=log_level,
            warn_start=warm_start,
            config=config,
            **kwargs,
        )

    def fit(self, X):
        return _PCA
