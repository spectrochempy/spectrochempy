# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa

"""PR4 geometry and mask policy tests for analysis outputs."""

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.analysis.crossdecomposition.pls import PLSRegression
from spectrochempy.analysis.decomposition.mcrals import MCRALS
from spectrochempy.analysis.decomposition.nmf import NMF
from spectrochempy.analysis.decomposition.pca import PCA
from spectrochempy.analysis.decomposition.simplisma import SIMPLISMA
from spectrochempy.analysis.decomposition.svd import SVD


@pytest.fixture()
def geometry_source_dataset():
    y = scp.Coord(np.arange(6.0), title="obs", units="s")
    x = scp.Coord(np.arange(5.0), title="feat", units="nm")
    return scp.NDDataset(
        np.array(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0],
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [2.0, 3.0, 4.0, 5.0, 6.0],
                [3.0, 4.0, 5.0, 6.0, 7.0],
                [4.0, 5.0, 6.0, 7.0, 8.0],
                [5.0, 6.0, 7.0, 8.0, 9.0],
            ]
        ),
        coordset=[y, x],
        title="source",
        name="source",
        units="absorbance",
    )


@pytest.fixture()
def pls_geometry_inputs():
    rng = np.random.default_rng(0)
    xobs = scp.Coord(np.arange(6.0) + 10.0, title="train observations")
    xfeat = scp.Coord(np.arange(5.0) + 50.0, title="x features")
    yobs = scp.Coord(np.arange(6.0) + 100.0, title="target observations")
    yfeat = scp.Coord(np.array([700.0, 800.0, 900.0]), title="target variables")
    X = scp.NDDataset(
        rng.normal(size=(6, 5)),
        coordset=[xobs, xfeat],
        title="X train",
        name="xtrain",
    )
    Y = scp.NDDataset(
        rng.normal(size=(6, 3)),
        coordset=[yobs, yfeat],
        title="Y train",
        name="ytrain",
    )
    return X, Y


def test_pca_factor_masks_restore_only_exact_source_axes(geometry_source_dataset):
    source = geometry_source_dataset

    masked_column = source.copy()
    masked_column[:, 4] = scp.MASKED
    pca_column = PCA(n_components=2).fit(masked_column)
    assert pca_column.scores.mask is np.False_
    assert pca_column.scores.shape == (6, 2)
    np.testing.assert_array_equal(pca_column.components.mask, masked_column.mask[:2])
    assert pca_column.components.shape == (2, 5)
    np.testing.assert_allclose(pca_column.components.x.data, masked_column.x.data)

    masked_row = source.copy()
    masked_row[2, :] = scp.MASKED
    pca_row = PCA(n_components=2).fit(masked_row)
    np.testing.assert_array_equal(
        np.all(np.asarray(pca_row.scores.mask), axis=-1),
        np.all(np.asarray(masked_row.mask), axis=-1),
    )
    assert pca_row.components.mask is np.False_
    np.testing.assert_allclose(pca_row.scores.y.data, masked_row.y.data)


def test_pca_reconstruction_restores_exact_full_geometry(geometry_source_dataset):
    source = geometry_source_dataset
    masked = source.copy()
    masked[2, 3] = scp.MASKED

    pca = PCA(n_components=2).fit(masked)
    reconstruction = pca.inverse_transform()

    assert reconstruction.shape == masked.shape
    assert reconstruction.dims == masked.dims
    np.testing.assert_allclose(reconstruction.x.data, masked.x.data)
    np.testing.assert_allclose(reconstruction.y.data, masked.y.data)
    np.testing.assert_array_equal(reconstruction.mask, masked.mask)

    reconstruction.mask[0, 0] = True
    assert not masked.mask[0, 0]


def test_reconstruction_direct_input_does_not_leak_fitted_geometry(
    geometry_source_dataset,
):
    source = geometry_source_dataset
    fitted_source = source.copy()
    fitted_source[:, 4] = scp.MASKED
    pca = PCA(n_components=2).fit(fitted_source)

    other = source.copy()
    other[:, 4] = scp.MASKED
    other.name = "other"
    other.y = scp.Coord(np.arange(6.0) + 1000.0, title="other observations")
    direct_scores = pca.transform(other)
    direct_reconstruction = pca.inverse_transform(direct_scores)
    arraylike_reconstruction = pca.inverse_transform(direct_scores.data)

    assert direct_reconstruction.mask is np.False_
    assert direct_reconstruction.shape == (6, 4)
    np.testing.assert_allclose(direct_reconstruction.y.data, other.y.data)
    np.testing.assert_allclose(direct_reconstruction.x.data, fitted_source.x.data[:4])
    assert arraylike_reconstruction.mask is np.False_
    assert arraylike_reconstruction.coordset is None


def test_nmf_reconstruction_and_components_restore_exact_feature_axis(
    geometry_source_dataset,
):
    source = geometry_source_dataset + 1.0
    masked = source.copy()
    masked[:, 4] = scp.MASKED

    nmf = NMF(
        n_components=2,
        init="nndsvda",
        max_iter=500,
        random_state=0,
    ).fit(masked)

    np.testing.assert_array_equal(nmf.components.mask, masked.mask[:2])
    reconstruction = nmf.inverse_transform()
    np.testing.assert_array_equal(reconstruction.mask, masked.mask)
    np.testing.assert_allclose(reconstruction.x.data, masked.x.data)


def test_profiles_restore_only_exact_supported_axes(geometry_source_dataset):
    source = geometry_source_dataset + 1.0

    simplisma_input = source.copy()
    simplisma_input[3, :] = scp.MASKED
    sma = SIMPLISMA(n_components=2).fit(simplisma_input)
    np.testing.assert_array_equal(
        np.all(np.asarray(sma.C.mask), axis=-1),
        np.all(np.asarray(simplisma_input.mask), axis=-1),
    )
    assert sma.St.mask is np.False_

    mcrals_input = source.copy()
    mcrals_input[:, 4] = scp.MASKED
    guess = scp.NDDataset(
        np.abs(np.random.default_rng(1).normal(size=(source.shape[0], 2))),
        title="guess",
    )
    mcr = MCRALS(max_iter=5, tol=1.0e-10, constraints=[]).fit(mcrals_input, guess)
    assert mcr.C.mask is np.False_
    np.testing.assert_array_equal(
        np.all(np.asarray(mcr.St.mask), axis=-2),
        np.all(np.asarray(mcrals_input.mask), axis=-2),
    )


def test_pls_prediction_assembles_axes_from_xpredict_and_ytrain(
    pls_geometry_inputs,
):
    Xtrain, Ytrain = pls_geometry_inputs
    pls = PLSRegression(n_components=2).fit(Xtrain, Ytrain)

    Xpredict = scp.NDDataset(
        np.random.default_rng(1).normal(size=(4, 5)),
        coordset=[
            scp.Coord(np.arange(4.0) + 1000.0, title="predict observations"),
            scp.Coord(np.arange(5.0) + 500.0, title="predict x features"),
        ],
        title="X predict",
        name="xpredict",
    )
    prediction = pls.predict(Xpredict)

    assert prediction.shape == (4, 3)
    assert prediction.dims == ["y", "x"]
    np.testing.assert_allclose(prediction.y.data, Xpredict.y.data)
    np.testing.assert_allclose(prediction.x.data, Ytrain.x.data)
    assert prediction.mask is np.False_


def test_pls_prediction_combines_observation_and_target_axis_masks(
    pls_geometry_inputs,
):
    Xtrain, Ytrain = pls_geometry_inputs
    Ymasked = Ytrain.copy()
    Ymasked[:, 1] = scp.MASKED
    pls = PLSRegression(n_components=2).fit(Xtrain, Ymasked)

    Xpredict = scp.NDDataset(
        np.random.default_rng(2).normal(size=(4, 5)),
        coordset=[
            scp.Coord(np.arange(4.0) + 1000.0, title="predict observations"),
            scp.Coord(np.arange(5.0) + 500.0, title="predict x features"),
        ],
        title="X predict",
        name="xpredict",
    )
    Xpredict[1, :] = scp.MASKED

    prediction = pls.predict(Xpredict)

    expected = np.zeros((4, 3), dtype=bool)
    expected[1, :] = True
    expected[:, 1] = True
    np.testing.assert_array_equal(prediction.mask, expected)


def test_pls_prediction_arraylike_does_not_reuse_old_xpredict_geometry(
    pls_geometry_inputs,
):
    Xtrain, Ytrain = pls_geometry_inputs
    pls = PLSRegression(n_components=2).fit(Xtrain, Ytrain)

    Xpredict = scp.NDDataset(
        np.random.default_rng(3).normal(size=(4, 5)),
        coordset=[
            scp.Coord(np.arange(4.0) + 1000.0, title="predict observations"),
            scp.Coord(np.arange(5.0) + 500.0, title="predict x features"),
        ],
        title="X predict",
        name="xpredict",
    )
    Xpredict[1, :] = scp.MASKED
    _ = pls.predict(Xpredict)

    arraylike_prediction = pls.predict(Xpredict.data)

    assert arraylike_prediction.dims == ["y", "x"]
    assert arraylike_prediction.mask is np.False_
    assert arraylike_prediction.coordset is not None
    assert arraylike_prediction.y.data is None
    np.testing.assert_allclose(arraylike_prediction.x.data, Ytrain.x.data)


def test_pls_prediction_monotarget_uses_public_1d_observation_geometry():
    rng = np.random.default_rng(4)
    X = scp.NDDataset(
        rng.normal(size=(6, 5)),
        coordset=[scp.Coord(np.arange(6.0), title="obs"), scp.Coord(np.arange(5.0))],
    )
    Y = scp.NDDataset(
        rng.normal(size=(6,)),
        coordset=[scp.Coord(np.arange(6.0) + 10.0, title="target observations")],
    )
    pls = PLSRegression(n_components=2).fit(X, Y)
    Xpredict = scp.NDDataset(
        rng.normal(size=(4, 5)),
        coordset=[scp.Coord(np.arange(4.0) + 100.0, title="predict obs"), scp.Coord(np.arange(5.0) + 50.0)],
    )
    Xpredict[2, :] = scp.MASKED

    prediction = pls.predict(Xpredict)

    assert prediction.shape == (4,)
    assert prediction.dims == ["y"]
    np.testing.assert_allclose(prediction.y.data, Xpredict.y.data)
    np.testing.assert_array_equal(prediction.mask, [False, False, True, False])


def test_svd_diagnostics_stay_generated_and_unmasked(geometry_source_dataset):
    svd = SVD().fit(geometry_source_dataset)

    for diagnostic in (
        svd.singular_values,
        svd.explained_variance,
        svd.explained_variance_ratio,
        svd.cumulative_explained_variance,
    ):
        assert diagnostic.mask is np.False_
        assert diagnostic.dims == ["k"]
        assert diagnostic.coordset is not None
        assert diagnostic.coordset["k"].title == "components"


def test_svd_diagnostics_restore_public_k_axis_without_source_mask_transfer(
    geometry_source_dataset,
):
    masked = geometry_source_dataset.copy()
    masked[:, -1] = scp.MASKED
    masked[-1, :] = scp.MASKED

    svd = SVD().fit(masked)

    for diagnostic in (
        svd.singular_values,
        svd.explained_variance,
        svd.explained_variance_ratio,
        svd.cumulative_explained_variance,
    ):
        assert diagnostic.shape == (5,)
        assert diagnostic.mask is np.False_
        assert diagnostic.dims == ["k"]
        assert diagnostic.coordset is not None
        assert diagnostic.coordset["k"].title == "components"
