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

from spectrochempy.analysis._analysisutils import NotFittedError
from spectrochempy.analysis.pca import PCA
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.utils import MASKED, exceptions, show, testing



# test pca
# ---------
def test_pca():

    dataset = NDDataset.read("irdata/nh4y-activation.spg")

    pca = PCA()
    pca.fit(dataset)
    assert pca._X.shape == (55, 5549)
    testing.assert_dataset_equal(
        pca.X, dataset
    ), "input dataset should be reflected in the internal variable X"

    # set n_components during init
    pca = PCA(used_components=5)
    assert pca.used_components == 5
    try:
        # the private attribute _n_components should not exist at this time
        _ = pca._n_components
    except NotFittedError:
        pass
    try:
        # so the n_components public attrribute.
        _ = pca.n_components
    except NotFittedError:
        pass

    pca = PCA(used_components=6)
    try:
        # _X initialized only when fit is used
        _ = pca._X.shape
    except NotFittedError:
        pass

    # Fit the model
    res = pca.fit(dataset)
    assert res is pca, "fit return self"

    # now the n_components has been defined
    assert pca.n_components == 6

    # try a wrong number of used_components  <= min(n_observations, n_features)
    try:

        pca = PCA(used_components=56)
        pca.fit(dataset)
    except ValueError:
        pass

    # try other ways to define n_components
    try:
        pca = PCA(used_components="mle")
        pca.fit(dataset)
    except ValueError as exc:
        assert (
            exc.args[0]
            == "used_components='mle' is only supported if n_observations >= n_features"
        )

    pca = PCA(used_components=0.99)  # in % of explained variance
    pca.fit(dataset)
    assert pca.n_components == 7

    # TODO: test other svd solvers

    # masked
    dataset[:, 1240.0:920.0] = MASKED  # do not forget to use float in slicing
    pca = PCA()
    pca.fit(dataset)
    assert pca._X.shape == (55, 5216), "missing row or col should be removed"
    assert pca.X.shape == (55, 5549), "missing row or col restored"
    testing.assert_dataset_equal(
        pca.X, dataset
    ), "input dataset should be reflected in the internal variable X (where mask is restored)"

    # much better fit when masking eratic data
    # more variance explained with less components
    pca = PCA(used_components=0.999)  # in % of explained variance
    pca.fit(dataset)
    assert pca.n_components == 4

    # get the loadings (actually the components) and scores
    assert pca.loadings.shape == (4, 5549)
    assert pca.scores.shape == (55, 4)

    # equivalent to the property scores,
    # transform can have additional n_components parameters
    scores = pca.transform(dataset, n_components=2)
    assert scores == pca.scores[:, :2]

    # if dataset is the same as used in fit, it is optional to pass it to the transform
    # method
    scores = pca.transform(n_components=2)
    assert scores == pca.scores[:, :2]

    # display scores
    pca.scoreplot(scores, 1, 2)
    plt.show()

    # show all calculated loadings
    loadings = pca.components  # all calculated loadings

    # show only some loadings
    loadings1 = pca.get_components(n_components=3)
    loadings1.plot(legend=True)
    plt.show()

    # inverse_transform / reconstruct
    X_hat = pca.inverse_transform(scores)

    # if scores was not determined or the X used for score is the same as in fit
    # score is optional
    X_hat = pca.inverse_transform()

    pca.plotmerit(dataset, X_hat)
    plt.show()

    # printev
    pca.printev(n_components=4)
    s = pca.__str__(n_components=4)

    # Another valid way to get the dimensionality reduction
    scores2 = PCA().fit_transform(dataset, n_components=2)
    testing.assert_array_almost_equal(
        scores2.data,
        scores.data,
    )

    # get variance
    ev = pca.explained_variance
    assert isinstance(ev, NDDataset)
    assert ev.shape == (pca.n_components,)
    assert ev.x.title == "components"
