# -*- coding: utf-8 -*-

# ======================================================================================================================
#  Copyright (©) 2015-2021 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.                                  =
#  CeCILL-B FREE SOFTWARE LICENSE AGREEMENT - See full LICENSE agreement in the root directory                         =
# ======================================================================================================================

""" Tests for the PCA module

"""
import pytest
import numpy as np

from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.core.analysis.pca import PCA
from spectrochempy.utils import MASKED, show
from spectrochempy.utils.testing import assert_array_almost_equal

HAS_SCIKITLEARN = False
try:
    from sklearn.decomposition import PCA as sklPCA

    HAS_SCIKITLEARN = True
except ImportError:
    pass


# test pca
# ---------


def test_pca():
    dataset = NDDataset.read("irdata/nh4y-activation.spg")

    # with masks
    dataset[:, 1240.0:920.0] = MASKED  # do not forget to use float in slicing

    pca = PCA(dataset)

    pca.printev(n_pc=5)

    assert str(pca)[:3] == "\nPC"

    pca.screeplot(n_pc=0.999)

    pca.screeplot(n_pc="auto")

    pca.scoreplot((1, 2))

    pca.scoreplot(1, 2, 3)

    show()


@pytest.mark.skipif(not HAS_SCIKITLEARN, reason="scikit-learn library not loaded")
def test_compare_scikit_learn():
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

    pcas = sklPCA(n_components=2)
    pcas.fit(X)

    pca = PCA(NDDataset(X))
    pca.printev(n_pc=2)

    assert_array_almost_equal(pca.sv.data, pcas.singular_values_)
    assert_array_almost_equal(pca.ev_ratio.data, pcas.explained_variance_ratio_ * 100.0)

    dataset = NDDataset.read("irdata/nh4y-activation.spg")
    X = dataset.data

    pcas = sklPCA(n_components=5)
    pcas.fit(X)

    dataset = X.copy()
    pca = PCA(NDDataset(dataset))

    pca.printev(n_pc=5)

    assert_array_almost_equal(pca.sv.data[:5], pcas.singular_values_[:5], 4)
    assert_array_almost_equal(
        pca.ev_ratio.data[:5], pcas.explained_variance_ratio_[:5] * 100.0, 4
    )
