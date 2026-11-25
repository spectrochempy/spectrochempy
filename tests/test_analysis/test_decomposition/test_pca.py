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
from spectrochempy.core.dataset.coordset import CoordSet
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


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, "2"])
def test_pca_method_n_components_must_be_positive_integer(low_rank_pca_dataset, bad):
    # n_components given to transform, inverse_transform, get_components and
    # printev is validated in the base class (0 used to fail with an obscure
    # error and -1 to drop the last component silently).
    pca = PCA(n_components=3).fit(low_rank_pca_dataset)
    match = "n_components must be a positive integer"
    with pytest.raises(ValueError, match=match):
        pca.transform(n_components=bad)
    with pytest.raises(ValueError, match=match):
        pca.inverse_transform(n_components=bad)
    with pytest.raises(ValueError, match=match):
        pca.get_components(n_components=bad)
    with pytest.raises(ValueError, match=match):
        pca.printev(n_components=bad)
    with pytest.raises(ValueError, match=match):
        pca.plot_scree(n_components=bad)


@pytest.mark.parametrize("bad", [0, -1])
def test_nmf_and_fast_ica_method_n_components_must_be_positive_integer(bad):
    rng = np.random.default_rng(0)
    X = NDDataset(np.abs(rng.normal(size=(8, 6))))
    nmf = scp.NMF(n_components=2, max_iter=500).fit(X)
    with pytest.raises(ValueError, match="n_components must be a positive integer"):
        nmf.get_components(n_components=bad)
    ica = scp.FastICA(n_components=2).fit(X)
    for method in (ica.transform, ica.inverse_transform, ica.get_components):
        with pytest.raises(
            ValueError, match="n_components must be a positive integer"
        ):
            method(n_components=bad)


def test_pca_method_n_components_valid_values(low_rank_pca_dataset):
    pca = PCA(n_components=3).fit(low_rank_pca_dataset)
    assert pca.transform(n_components=np.int64(2)).shape[1] == 2
    assert pca.get_components(n_components=1).shape[0] == 1
    assert pca.get_components(n_components=None).shape[0] == 3
    with pytest.warns(UserWarning, match="cannot be greater"):
        assert pca.transform(n_components=5).shape[1] == 3


def test_pca_inverse_transform_uses_n_components(low_rank_pca_dataset):
    # inverse_transform(n_components=k) used to ignore k and reconstruct with
    # all the fitted components.
    pca = PCA(n_components=3).fit(low_rank_pca_dataset)
    sk = pca._pca
    scores = sk.transform(low_rank_pca_dataset.data)
    expected = scores[:, :1] @ sk.components_[:1] + sk.mean_
    assert_allclose(pca.inverse_transform(n_components=1).data, expected)
    assert_allclose(
        pca.inverse_transform(pca.transform(), n_components=1).data, expected
    )
    assert_allclose(
        pca.inverse_transform(n_components=None).data, pca.inverse_transform().data
    )


def test_pca_whiten_inverse_transform_uses_n_components():
    rng = np.random.default_rng(0)
    X = NDDataset(rng.normal(size=(12, 8)))
    pca = PCA(n_components=4, whiten=True).fit(X)
    sk = pca._pca
    scores = sk.transform(X.data)
    expected = (
        scores[:, :2] @ (np.sqrt(sk.explained_variance_[:2, None]) * sk.components_[:2])
        + sk.mean_
    )
    assert_allclose(pca.inverse_transform(n_components=2).data, expected)
    with pytest.warns(UserWarning, match="X_transform size"):
        reconstructed = pca.inverse_transform(pca.transform(n_components=2))
    assert_allclose(reconstructed.data, expected)
    # the fitted model is left intact
    assert sk.components_.shape == (4, 8)
    assert sk.explained_variance_.shape == (4,)
    assert_allclose(pca.transform().data, scores)


def test_nmf_and_fast_ica_inverse_transform_use_n_components():
    rng = np.random.default_rng(0)
    X = NDDataset(np.abs(rng.normal(size=(10, 8))))

    nmf = scp.NMF(n_components=3, max_iter=500).fit(X)
    W = nmf._nmf.transform(X.data)
    assert_allclose(
        nmf.inverse_transform(n_components=2).data,
        W[:, :2] @ nmf._nmf.components_[:2],
    )

    ica = scp.FastICA(n_components=3).fit(X)
    S = ica._fast_ica.transform(X.data)
    expected = S[:, :2] @ ica._fast_ica.mixing_[:, :2].T + ica._fast_ica.mean_
    assert_allclose(ica.inverse_transform(n_components=2).data, expected)
    # Scores with fewer columns than fitted components used to raise
    # (the pre-existing warning about the X_transform size is unchanged)
    with pytest.warns(UserWarning, match="X_transform size"):
        reconstructed = ica.inverse_transform(ica.transform(n_components=2))
    assert_allclose(reconstructed.data, expected)
    assert ica._fast_ica.mixing_.shape == (8, 3)


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
    assert X_hat.title == "reconstruction"
    assert X_hat.units == dataset.units
    assert X_hat.dims == dataset.dims


def test_pca_transform_handles_partial_coordset_without_observation_axis():
    x = scp.Coord(np.linspace(400.0, 700.0, 4), title="wavelength", units="nm")
    dataset = scp.NDDataset(
        np.array(
            [
                [0.0, 1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 3.0, 4.0, 5.0],
            ]
        ),
        units="absorbance",
    )
    dataset.coordset = CoordSet(x=x)

    pca = PCA(n_components=2).fit(dataset)

    scores = pca.transform()
    assert scores.shape == (3, 2)
    assert scores.dims == ["y", "k"]
    assert scores.coordset is not None
    assert scores.y.is_empty
    assert scores.k.title == "components"

    scores_prop = pca.scores
    assert scores_prop.shape == (3, 2)
    assert scores_prop.y.is_empty
    assert_allclose(scores_prop.data, scores.data)


def test_pca_reporting(low_rank_pca_dataset):
    dataset = low_rank_pca_dataset
    pca = PCA(n_components=3)
    pca.fit(dataset)

    pca.printev(n_components=4)
    s = pca.__str__(n_components=4)
    assert "PC\tEigenvalue" in s
    assert "#1" in s
