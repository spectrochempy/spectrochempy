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
import matplotlib.pyplot as plt
import numpy as np

from spectrochempy.analysis.pca import PCA
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.utils import MASKED, exceptions, show, testing

# test pca
# ---------


def test_pca():

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
    try:
        str(s)[:3] == "\nPC"
    except exceptions.NotFittedError:
        pass
    # two other valid ways to get the reduction
    # 1
    scores2 = PCA().fit_reduce(dataset, n_components=2)
    assert testing.assert_dataset_equal(scores2, scores1)
    # 2
    scores3 = PCA().fit(dataset).reduce(n_components=2)
    assert testing.assert_dataset_equal(scores3, scores1)

    # function in Analysis configurable
    print(PCA.help)
