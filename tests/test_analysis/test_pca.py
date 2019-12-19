# -*- coding: utf-8 -*-
#
# ======================================================================================================================
# Copyright (©) 2015-2020 LCS
# Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory
# ======================================================================================================================

""" Tests for the PCA module

"""
import pytest
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.analysis.pca import PCA
from spectrochempy.utils import MASKED, show
from spectrochempy.core import info_
from spectrochempy.utils.testing import assert_array_almost_equal, assert_array_equal
from spectrochempy.core import HAS_SCIKITLEARN

# test pca
# ---------

def test_pca(IR_dataset_2D):

    dataset = IR_dataset_2D.copy()

    # with masks
    dataset[:, 1240.0:920.0] = MASKED  # do not forget to use float in slicing

    pca = PCA(dataset)
    info_(pca)

    pca.printev(n_pc=5)

    assert str(pca)[:3] == '\nPC'

    pca.screeplot(n_pc=0.999)

    pca.screeplot(n_pc='auto')

    pca.scoreplot((1,2))

    pca.scoreplot(1,2, 3)

    show()

@pytest.mark.skipif(not HAS_SCIKITLEARN, reason="scikit-learn library not loaded")
def test_compare_scikit_learn(IR_dataset_2D):
    
    import numpy as np
    from sklearn.decomposition import PCA as PCAs

    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

    pcas = PCAs(n_components=2)
    pcas.fit(X)

    info_('')
    for i in range(2):
        info_(pcas.singular_values_[i], pcas.explained_variance_ratio_[i]*100.)

    pca = PCA(NDDataset(X))
    pca.printev(n_pc=2)

    assert_array_almost_equal(pca.sv.data, pcas.singular_values_)
    assert_array_almost_equal(pca.ev_ratio.data, pcas.explained_variance_ratio_*100.)

    X = IR_dataset_2D.data

    pcas = PCAs(n_components=5)
    pcas.fit(X)

    info_('')
    for i in range(5):
        info_(pcas.singular_values_[i], pcas.explained_variance_ratio_[i]*100.)

    dataset = X.copy()
    pca = PCA(NDDataset(dataset))

    info_('')
    for i in range(5):
        info_(pca.sv.data[i], pca.ev_ratio.data[i])

    pca.printev(n_pc=5)

    assert_array_almost_equal(pca.sv.data[:5], pcas.singular_values_[:5], 4)
    assert_array_almost_equal(pca.ev_ratio.data[:5], pcas.explained_variance_ratio_[:5]*100., 4)
