# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

"""
Tests for the PCA module

"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import spectrochempy as scp
from spectrochempy.analysis._base._analysisbase import NotFittedError
from spectrochempy.analysis.decomposition.pca import PCA
from spectrochempy.core.dataset.nddataset import NDDataset
from spectrochempy.utils import docutils as chd
from spectrochempy.utils import testing


def test_PCA_docstrings():
    chd.PRIVATE_CLASSES = []  # do not test private class docstring
    module = "spectrochempy.analysis.decomposition.pca"
    chd.check_docstrings(
        module,
        obj=scp.PCA,
        # exclude some errors - remove whatever you want to check
        exclude=["SA01", "EX01", "EX02", "ES01", "GL11", "GL08", "PR01"],
    )


@pytest.fixture()
def low_rank_pca_dataset():
    u1 = np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0]) / np.sqrt(6.0)
    u2 = np.array([1.0, -1.0, 0.0, 1.0, -1.0, 0.0]) / 2.0
    u3 = np.array([1.0, 1.0, -2.0, 1.0, 1.0, -2.0]) / np.sqrt(12.0)
    data = np.column_stack(
        [
            6.0 * u1,
            3.0 * u2,
            u3,
            np.zeros(6),
            np.zeros(6),
        ]
    )
    return scp.NDDataset(
        data,
        coordset=[
            scp.Coord.arange(6, title="sample"),
            scp.Coord.arange(5, title="feature"),
        ],
        units="absorbance",
        title="synthetic PCA matrix",
    )


@pytest.fixture()
def expected_variance_ratio():
    return 100.0 * np.array([36.0, 9.0, 1.0, 0.0, 0.0]) / 46.0


def test_pca_prefit_n_components():
    pca = PCA(n_components=5)
    assert pca.n_components == 5
    with pytest.raises(NotFittedError):
        _ = pca._n_components

    pca = PCA(n_components=3)
    with pytest.raises(NotFittedError):
        _ = pca._X.shape


def test_pca_fit_and_variance(low_rank_pca_dataset, expected_variance_ratio):
    dataset = low_rank_pca_dataset
    pca = PCA()
    res = pca.fit(dataset)

    assert res is pca
    assert pca._X.shape == (6, 5)
    assert pca.n_components == 5
    testing.assert_dataset_equal(pca.X, dataset)

    assert pca.loadings.shape == (5, 5)
    assert pca.loadings.dims == ["k", "x"]
    assert pca.scores.shape == (6, 5)
    assert pca.scores.dims == ["y", "k"]

    assert isinstance(pca.explained_variance, NDDataset)
    assert pca.ev.shape == (5,)
    assert pca.ev.k.title == "components"
    assert pca.ev.title == "explained variance"
    assert_allclose(pca.ev.data, [7.2, 1.8, 0.2, 0.0, 0.0])
    assert_allclose(pca.ev_ratio.data, expected_variance_ratio)
    assert_allclose(pca.ev_cum.data, np.cumsum(expected_variance_ratio))


def test_pca_n_components_validation(low_rank_pca_dataset):
    dataset = low_rank_pca_dataset

    with pytest.raises(ValueError, match="n_components=6"):
        PCA(n_components=6).fit(dataset)

    wide_dataset = scp.NDDataset(
        np.arange(24.0).reshape(4, 6),
        coordset=[scp.Coord.arange(4), scp.Coord.arange(6)],
    )
    with pytest.raises(
        ValueError,
        match="n_components='mle' is only supported if n_observations >= n_features",
    ):
        PCA(n_components="mle").fit(wide_dataset)


def test_pca_float_threshold_component_selection(low_rank_pca_dataset):
    dataset = low_rank_pca_dataset

    pca = PCA(n_components=0.95)
    pca.fit(dataset)
    assert pca.n_components == 2
    assert pca.loadings.shape == (2, 5)
    assert pca.scores.shape == (6, 2)

    pca = PCA(n_components=0.99)
    pca.fit(dataset)
    assert pca.n_components == 3
    assert pca.loadings.shape == (3, 5)
    assert pca.scores.shape == (6, 3)


def test_pca_masked_columns(low_rank_pca_dataset, expected_variance_ratio):
    dataset = low_rank_pca_dataset
    masked = dataset.copy()
    masked[:, 4] = scp.MASKED

    pca = PCA()
    pca.fit(masked)

    assert pca._X.shape == (6, 4), "fully masked columns should be removed"
    assert pca.X.shape == (6, 5), "masked columns should be restored"
    testing.assert_dataset_equal(pca.X, masked)
    assert_allclose(pca.ev_ratio.data, expected_variance_ratio[:4])

    pca = PCA(n_components=0.95)
    pca.fit(masked)
    assert pca.n_components == 2
    assert pca.loadings.shape == (2, 5)
    assert pca.scores.shape == (6, 2)


def test_pca_transform_fit_transform_and_inverse(low_rank_pca_dataset):
    dataset = low_rank_pca_dataset
    pca = PCA(n_components=3)
    pca.fit(dataset)

    scores = pca.transform(dataset, n_components=2)
    assert scores == pca.scores[:, :2]
    scores_without_dataset = pca.transform(n_components=2)
    assert scores_without_dataset == pca.scores[:, :2]

    pca2 = PCA()
    scores2 = pca2.fit_transform(dataset, n_components=2)
    assert_allclose(np.abs(scores2.data), np.abs(scores.data))

    X_hat_2 = pca.inverse_transform(scores)
    assert X_hat_2.shape == dataset.shape
    X_hat = pca.inverse_transform()
    assert_allclose(X_hat.data, dataset.data, atol=1.0e-12)
    assert X_hat.title == dataset.title
    assert X_hat.units == dataset.units
    assert X_hat.dims == dataset.dims


def test_pca_reporting(low_rank_pca_dataset):
    dataset = low_rank_pca_dataset
    pca = PCA(n_components=3)
    pca.fit(dataset)

    pca.printev(n_components=4)
    s = pca.__str__(n_components=4)
    assert "PC\tEigenvalue" in s
    assert "#1" in s
