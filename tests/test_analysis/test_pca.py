# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# flake8: noqa

"""
Tests for the PCA module

"""
import numpy as np
from utils.optional import import_optional_dependency

from spectrochempy.analysis.pca import PCA
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.utils import MASKED, show
from spectrochempy.utils.testing import assert_array_almost_equal

# test pca
# ---------


def test_pca():
    dataset = NDDataset.read("irdata/nh4y-activation.spg")

    # with masks
    dataset[:, 1240.0:920.0] = MASKED  # do not forget to use float in slicing

    pca = PCA(dataset)

    assert str(pca)[:3] == "\nPC"

    # test alternative options
    pca = PCA(dataset, centered=False, standardized=True, scaled=True)

    assert str(pca)[:3] == "\nPC"

    _, LT = pca.reduce(n_pc="auto")
    assert LT.shape[0] == 15

    dataset_hat = pca.reconstruct(n_pc=5)
    assert (dataset_hat - dataset).max() < 0.4

    # test with observations < variables
    pca2 = PCA(dataset[:, 4000.0:3995.0])
    _, LT2 = pca2.reduce(n_pc="auto")
    assert LT2.shape[0] == 5

    # test text and plots outputs

    pca.printev(n_pc=5)

    pca.screeplot(n_pc=5)

    pca.screeplot(n_pc="auto")

    pca.scoreplot((1, 2))

    pca.scoreplot(1, 2, 3)

    pca.plotmerit

    show()


def test_compare_scikit_learn():

    try:
        import_optional_dependency("scikit-learn")
    except ImportError:
        return

    from sklearn.decomposition import PCA as sklPCA

    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

    pcas = sklPCA(n_components=2)
    pcas.fit(X)

    pca = PCA(NDDataset(X))
    pca.printev(n_pc=2)

    assert_array_almost_equal(pca._sv.data, pcas.singular_values_)
    assert_array_almost_equal(pca.ev_ratio.data, pcas.explained_variance_ratio_ * 100.0)

    dataset = NDDataset.read("irdata/nh4y-activation.spg")
    X1 = dataset.copy().data

    pcas = sklPCA(n_components=5, svd_solver="full")
    pcas.fit(X1)

    X2 = NDDataset(dataset.copy())
    pca = PCA(X2)

    pca.printev(n_pc=5)

    assert_array_almost_equal(pca._sv.data[:5], pcas.singular_values_[:5], 4)
    assert_array_almost_equal(
        pca.ev_ratio.data[:5], pcas.explained_variance_ratio_[:5] * 100.0, 4
    )

    show()


def _test_issue_15():
    x = NDDataset.read_omnic("irdata/nh4y-activation.spg")
    my_pca = PCA(x)
    my_pca.reconstruct(n_pc=3)
