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

from spectrochempy.analysis.pca import PCA
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.utils import MASKED, exceptions, show, testing
from spectrochempy.utils.optional import import_optional_dependency

# test pca
# ---------


def test_pca():
    dataset = NDDataset.read("irdata/nh4y-activation.spg")

    # with masks
    dataset[:, 1240.0:920.0] = MASKED  # do not forget to use float in slicing

    pca = PCA()
    try:
        str(pca)[:3] == "\nPC"
    except exceptions.NotFittedError:
        pass

    pca.fit(dataset)
    assert str(pca)[:3] == "\nPC"

    # test alternative options
    pca = PCA(centered=False, standardized=True, scaled=True)
    pca.fit(dataset)

    assert str(pca)[:3] == "\nPC"

    _, LT = pca.reduce(n_pc="auto")
    assert LT.shape[0] == 15

    dataset_hat = pca.reconstruct(n_pc=5)
    assert (dataset_hat - dataset).max() < 0.4

    # test fit_reconstruct
    pca = PCA(centered=False, standardized=True, scaled=True)
    dataset_hat_2 = pca.fit_reconstruct(dataset, n_pc=5)
    assert testing.assert_dataset_equal(dataset_hat, dataset_hat_2)

    # test fit + reconstruct
    pca = PCA(centered=False, standardized=True, scaled=True)
    dataset_hat_3 = pca.fit(dataset).reconstruct(n_pc=5)
    assert testing.assert_dataset_equal(dataset_hat, dataset_hat_3)

    # test with observations < variables
    pca2 = PCA()
    pca2.fit(dataset[:, 4000.0:3995.0])
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

    import matplotlib.pyplot as plt

    from spectrochempy import MASKED, NDDataset
    from spectrochempy.utils import testing

    dataset = NDDataset.read("irdata/nh4y-activation.spg")
    dataset[:, 1240.0:920.0] = MASKED  # do not forget to use float in slicing

    pca1 = PCA()
    pca1.fit(dataset)

    assert pca1._X.shape == (55, 5216), "missing row or col removed"
    assert testing.assert_dataset_equal(
        pca1.X, dataset
    ), "input dataset should be reflected in the internal variable X"

    # display scores
    scores1 = pca1.reduce(n_components=2)
    pca1.scoreplot(scores1, 1, 2)
    plt.show()

    # show all calculated loadings
    loadings = pca1.components  # all calculated loadings

    # show only some loadings
    loadings1 = pca1.get_components(n_components=2)
    loadings1.plot(legend=True)
    plt.show()

    # reconstruct
    X_hat = pca1.reconstruct(scores1)
    pca1.plotmerit(dataset, X_hat)
    plt.show()

    # printev
    pca1.printev(n_components=5)
    s = pca1.__str__(n_components=5)

    # two other valid ways to get the reduction
    # 1
    scores2 = PCA().fit_reduce(dataset, n_components=2)
    assert testing.assert_dataset_equal(scores2, scores1)
    # 2
    scores3 = PCA().fit(dataset).reduce(n_components=2)
    assert testing.assert_dataset_equal(scores3, scores1)

    import matplotlib.pyplot as plt

    from spectrochempy import MASKED, NDDataset
    from spectrochempy.utils import testing

    dataset = NDDataset.read("irdata/nh4y-activation.spg")
    dataset[:, 1240.0:920.0] = MASKED  # do not forget to use float in slicing

    pca1 = PCA()
    pca1.fit(dataset)

    assert pca1._X.shape == (55, 5216), "missing row or col removed"
    assert testing.assert_dataset_equal(
        pca1.X, dataset
    ), "input dataset should be reflected in the internal variable X"

    # display scores
    scores1 = pca1.reduce(n_components=2)
    pca1.scoreplot(scores1, 1, 2)
    plt.show()

    # show all calculated loadings
    loadings = pca1.components  # all calculated loadings

    # show only some loadings
    loadings1 = pca1.get_components(n_components=2)
    loadings1.plot(legend=True)
    plt.show()

    # reconstruct
    X_hat = pca1.reconstruct(scores1)
    pca1.plotmerit(dataset, X_hat)
    plt.show()

    # printev
    pca1.printev(n_components=5)
    s = pca1.__str__(n_components=5)

    # two other valid ways to get the reduction
    # 1
    scores2 = PCA().fit_reduce(dataset, n_components=2)
    assert testing.assert_dataset_equal(scores2, scores1)
    # 2
    scores3 = PCA().fit(dataset).reduce(n_components=2)
    assert testing.assert_dataset_equal(scores3, scores1)
