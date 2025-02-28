# ======================================================================================
# Copyright (©) 2015-2025 LCS - Laboratoire Catalyse et Spectrochimie, Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

"""
Tests for the PLSRegression module

"""

from os import environ

import matplotlib.pyplot as plt
import pytest
from numpy.testing import assert_almost_equal
from sklearn.cross_decomposition import PLSRegression as sklPLSRegression

import spectrochempy as scp
from spectrochempy.analysis.crossdecomposition.pls import PLSRegression
from spectrochempy.core.readers.importer import read
from spectrochempy.utils import docstrings as chd
from spectrochempy.utils.constants import MASKED
from spectrochempy.utils.testing import assert_dataset_equal


# test docstring
# but this is not intended to work with the debugger - use run instead of debug!
@pytest.mark.skipif(
    environ.get("PYDEVD_LOAD_VALUES_ASYNC", None),
    reason="debug mode cause error when checking docstrings",
)
def test_PLS_docstrings():
    chd.PRIVATE_CLASSES = []  # do not test private class docstring
    module = "spectrochempy.analysis.crossdecomposition.pls"
    chd.check_docstrings(
        module,
        obj=scp.PLSRegression,
        # exclude some errors - remove whatever you want to check
        exclude=["SA01", "EX01", "ES01", "GL11", "GL08", "PR01"],
    )


# test pls
# ---------
def test_pls():
    datasets = read("http://www.eigenvector.com/data/Corn/corn.mat", merge=False)
    # information: [20x59 char ]    Information about the data
    # m5spec: [80x700 dataset] Spectra on instrument m5
    # mp5spec: [80x700 dataset] Spectra on instrument mp5
    # mp6spec: [80x700 dataset] Spectra on instrument mp6
    # propvals: [80x4 dataset]   Property values for samples
    # m5nbs: [ 3x700 dataset] NBS glass stds on m5
    # mp5nbs: [ 4x700 dataset] NBS glass stds on mp5
    # mp6nbs: [ 4x700 dataset] NBS glass stds on mp6
    assert len(datasets) == 7

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

    pls1 = PLSRegression(n_components=5)
    pls1.fit(Xc, yc)
    pls1_ = sklPLSRegression(n_components=5)
    pls1_.fit(Xc_array, yc_array)

    pls2 = PLSRegression(n_components=5)
    pls2.fit(Xc, Yc)
    pls2_ = sklPLSRegression(n_components=10)
    pls2_.fit(Xc_array, Yc_array)

    # check fit is OK and that appropriate X metadata are passed to loadings and scores
    # note that scipy loadings are transposed w.r.t. sklearn loadings so that they are consistent
    # with matrix multiplication, i.e.:
    # X = x_scores @ x_loadings + residuals
    # Y = y_scores @ y_loadings + residuals
    #
    assert_almost_equal(pls1.x_loadings.data, pls1_.x_loadings_.T)
    assert pls1.x_loadings.x == Xc.x
    assert_almost_equal(pls1.y_loadings.data, pls1_.y_loadings_.squeeze().T)
    assert_almost_equal(pls1.x_scores.data, pls1_.x_scores_)
    assert pls1.x_scores.y == Xc.y
    assert_almost_equal(pls1.y_scores.data, pls1_.y_scores_)
    assert pls1.x_scores.y == Yc.y

    # check R^2 on calibration data...
    assert pls1.score() == pls1_.score(Xc_array, yc_array)
    # ... and validation data
    assert_almost_equal(pls1.score(Xv, yv), pls1_.score(Xv_array, yv_array), decimal=10)

    # check predict()
    y_hat = pls1.predict(Xv)
    y_hat_ = pls1_.predict(Xv_array)
    assert_almost_equal(y_hat.data, y_hat_.squeeze())

    # check transform() with calibration data
    x_scores = pls1.transform()  # this is equivalent to pls1.transform(Xc)
    x_scores_ = pls1_.transform(Xc_array)
    assert_almost_equal(x_scores.data, x_scores_)

    # check transform() with validation data
    x_scores = pls1.transform(Xv)
    x_scores_ = pls1_.transform(Xv_array)
    assert_almost_equal(x_scores.data, x_scores_)

    # check transform() with X and y calibration data
    x_scores, y_scores = pls1.transform(both=True)
    x_scores_, y_scores_ = pls1_.transform(Xc_array, yc_array)
    assert_almost_equal(y_scores.data, y_scores_)

    # check transform() with X and y validation data
    x_scores, y_scores = pls1.transform(Xv, yv)
    x_scores_, y_scores_ = pls1_.transform(Xv_array, yv_array)
    assert_almost_equal(y_scores.data, y_scores_)

    # check fit_transform() with calibration data
    x_scores = pls1.fit_transform(Xc, yc)
    x_scores_, y_scores_ = pls1_.fit_transform(Xc_array, yc_array)
    assert_almost_equal(x_scores.data, x_scores_)

    # check inverse_transform() with calibration data
    x_hat = pls1.inverse_transform(
        x_scores
    )  # this is equivalent to pls1.transform(x_scores)
    x_hat_ = pls1_.inverse_transform(x_scores_)
    assert_almost_equal(x_hat.data, x_hat_)

    # check inverse_transform() with validation data
    x_scores, y_scores = pls1.transform(Xv, yv)
    xv_hat, yv_hat = pls1.inverse_transform(x_scores, y_scores)
    x_scores_, y_scores_ = pls1_.transform(Xv_array, yv_array)
    xv_hat_, yv_hat_ = pls1_.inverse_transform(x_scores_, y_scores_)
    assert_almost_equal(xv_hat.data, xv_hat_.squeeze())
    assert_almost_equal(yv_hat.data, yv_hat_.squeeze(), 3)
    # todo: check why only 3 decimals

    # Test masked data, x axis
    pls2 = PLSRegression(n_components=5)
    Xc[:, 1600.0:1800.0] = MASKED  # corn spectra, calibration
    pls2.fit(Xc, yc)

    assert pls2._X.shape == (57, 599), "missing row or col should be removed"
    assert pls2.X.shape == (57, 700), "missing row or col restored"
    (
        assert_dataset_equal(
            pls2.X,
            Xc,
            data_only=True,
        ),
        "input dataset should be reflected in the internal variable X (where mask is restored)",
    )

    # check plots
    pls1.plotmerit()
    plt.show()
    pls1.parityplot()
    plt.show()
    pls2.plotmerit()
    plt.show()
