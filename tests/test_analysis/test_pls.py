# -*- coding: utf-8 -*-
# ======================================================================================
# Copyright (©) 2015-2023 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# flake8: noqa

"""
Tests for the PLS module

"""

from numpy.testing import assert_almost_equal
from sklearn.cross_decomposition import PLSRegression

from spectrochempy.analysis.pls import PLS
from spectrochempy.core.readers.download import download


# test pls
# ---------
def test_pls():

    datasets = download("http://www.eigenvector.com/data/Corn/corn.mat")
    Xc = datasets[-3][:57]  # corn spectra, calibration
    Xv = datasets[-3][57:]  # corn spectra, validation
    Yc = datasets[4][:57]  # properties values, calibration
    Yv = datasets[4][57:]  # properties values  validation
    yc = Yc[:, 0]  # moisture
    yv = Yv[:, 0]

    # get arrays for comparison with direct use of sklearn
    Xc_array = Xc.data
    Yc_array = Yc.data
    yc_array = yc.data
    Xv_array = Xv.data
    Yv_array = Yv.data
    yv_array = yv.data

    pls1 = PLS(used_components=5)
    pls1.fit(Xc, yc)
    pls1_ = PLSRegression(n_components=5)
    pls1_.fit(Xc_array, yc_array)

    pls2 = PLS(used_components=5)
    pls2.fit(Xc, Yc)
    pls2_ = PLSRegression(n_components=10)
    pls2_.fit(Xc_array, Yc_array)

    # check fit is OK and that appropriate X metadata are passed to loadings and scores
    # note that scipy loadings are transposed w.r.t. sklearn loadings so that they are consistent
    # with matrix multiplication, i.e.:
    # X = x_scores @ x_loadings + residuals
    # Y = y_scores @ y_loadings + residuals
    #
    assert (pls1.x_loadings.data == pls1_.x_loadings_.T).all
    assert pls1.x_loadings.x == Xc.x
    assert (pls1.y_loadings.data == pls1_.y_loadings_.T).all
    assert (pls1.x_scores.data == pls1_.x_scores_).all
    assert pls1.x_scores.y == Xc.y
    assert (pls1.y_scores.data == pls1_.y_scores_).all
    assert pls1.x_scores.y == Yc.y

    # check R^2 on calibration data...
    assert pls1.score() == pls1_.score(Xc_array, yc_array)
    # ... and validation data
    assert_almost_equal(pls1.score(Xv, yv), pls1_.score(Xv_array, yv_array), decimal=10)

    # check predict()
    y_hat = pls1.predict(Xv)
    y_hat_ = pls1_.predict(Xv_array)
    assert (y_hat.data == y_hat_).all

    # same with copy=False
    y_hat = pls1.predict(Xv, copy=False)
    y_hat_ = pls1_.predict(Xv_array, copy=False)
    assert (y_hat.data == y_hat_).all

    # check transform() with calibration data
    x_scores = pls1.transform()  # this is equivalent to pls1.transform(Xc)
    x_scores_ = pls1_.transform(Xc_array)
    assert (x_scores.data == x_scores_).all

    # check transform() with validation data
    x_scores = pls1.transform(Xv)
    x_scores_ = pls1_.transform(Xv_array)
    assert (x_scores.data == x_scores_).all

    # check transform() with X and y calibration data
    x_scores, y_scores = pls1.transform(both=True)
    x_scores_, y_scores_ = pls1_.transform(Xc_array, yc_array)
    assert (y_scores.data == y_scores_).all

    # check transform() with X and y validation data
    x_scores, y_scores = pls1.transform(Xv, yv)
    x_scores_, y_scores_ = pls1_.transform(Xv_array, yv_array)
    assert (y_scores.data == y_scores_).all

    # same as above with in-place normalization
    x_scores, y_scores = pls1.transform(Xv, yv, copy=False)
    x_scores_, y_scores_ = pls1_.transform(Xv_array, yv_array, copy=False)
    assert (y_scores.data == y_scores_).all

    # check fit_transform() with calibration data
    x_scores = pls1.fit_transform(Xc, yc)
    x_scores_, y_scores_ = pls1_.fit_transform(Xc_array, yc_array)
    assert (x_scores.data == x_scores_).all

    # check inverse_transform() with calibtration data
    x_hat = pls1.inverse_transform(
        x_scores
    )  # this is equivalent to pls1.transform(x_scores)
    x_hat_ = pls1_.inverse_transform(x_scores_)
    assert (x_hat.data == x_hat_).all

    # check inverse_transform() with validation data
    x_scores, y_scores = pls1.transform(Xv, yv)
    x_hat, y_hat = pls1.inverse_transform(x_scores, y_scores)
    x_scores_, y_scores_ = pls1_.transform(Xv_array, yv_array)
    x_hat_, y_hat_ = pls1_.inverse_transform(x_scores_, y_scores_)
    assert (y_hat.data == y_hat_).all

    # check plots
    pls1.plotmerit()
    pls1.plotparity()
