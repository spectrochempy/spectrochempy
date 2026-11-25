# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
"""
Tests for the SVD class
"""

from os import environ

import numpy as np
import pytest
from numpy.testing import assert_allclose

import spectrochempy as scp
from spectrochempy.analysis._base._analysisbase import NotFittedError
from spectrochempy.analysis.decomposition.svd import SVD
from spectrochempy.utils import docutils as chd


# test docstring
# but this is not intended to work with the debugger - use run instead of debug!
@pytest.mark.skipif(
    environ.get("PYDEVD_LOAD_VALUES_ASYNC", None),
    reason="debug mode cause error when checking docstrings",
)
def test_SVD_docstrings():
    chd.PRIVATE_CLASSES = []  # do not test private class docstring
    module = "spectrochempy.analysis.decomposition.svd"
    chd.check_docstrings(
        module,
        obj=scp.SVD,
        # exclude some errors - remove whatever you want to check
        exclude=["SA01", "EX01", "EX02", "ES01", "GL11", "GL08", "PR01"],
    )


@pytest.fixture()
def low_rank_dataset():
    y = scp.Coord.arange(4, title="sample")
    x = scp.Coord.arange(5, title="feature")
    data = np.zeros((4, 5))
    data[0, 0] = 5.0
    data[1, 1] = 3.0
    return scp.NDDataset(
        data,
        coordset=[y, x],
        units="absorbance",
        title="synthetic low-rank matrix",
    )


def test_svd(low_rank_dataset):
    dataset = low_rank_dataset
    svd = SVD()
    result = svd.fit(dataset)

    assert result is svd
    assert svd.U.shape == (4, 4)
    assert svd.VT.shape == (4, 5)
    assert svd.sv.shape == (4,)
    assert svd.sv.title == "singular values"
    assert svd.sv.dims == ["k"]
    assert_allclose(svd.s, [5.0, 3.0, 0.0, 0.0])
    assert_allclose(svd.ev_ratio.data, [2500.0 / 34.0, 900.0 / 34.0, 0.0, 0.0])
    assert_allclose(svd.ev_cum.data, [2500.0 / 34.0, 100.0, 100.0, 100.0])

    # Fully masked rows/columns are ignored by the SVD calculation while the
    # public input dataset keeps its original shape.
    masked = dataset.copy()
    masked[:, 4] = scp.MASKED
    masked[3] = scp.MASKED

    svd.fit(masked)
    assert svd.X.shape == dataset.shape
    assert svd.U.shape == (3, 3)
    assert svd.VT.shape == (3, 4)
    assert_allclose(svd.s, [5.0, 3.0, 0.0])
    assert_allclose(svd.ev_ratio.data, [2500.0 / 34.0, 900.0 / 34.0, 0.0])

    svd.full_matrices = True
    svd.fit(masked)

    assert svd.U.shape == (3, 3)
    assert svd.VT.shape == (4, 4)
    assert_allclose(svd.s, [5.0, 3.0, 0.0])


def test_svd_compute_uv_false(low_rank_dataset):
    dataset = low_rank_dataset
    svd = SVD(compute_uv=False)
    result = svd.fit(dataset)

    assert result is svd
    # _outfit is always a (U, s, VT) tuple after _fit normalisation
    assert isinstance(svd._outfit, tuple)
    assert len(svd._outfit) == 3
    # U and VT are None when compute_uv=False
    assert svd.U is None
    assert svd.VT is None
    # Singular values are correct
    assert_allclose(svd.s, [5.0, 3.0, 0.0, 0.0])
    assert svd.sv.shape == (4,)
    assert svd.sv.title == "singular values"
    # Diagnostics work correctly
    assert svd.ev.shape == (4,)
    assert svd.ev_ratio.shape == (4,)
    assert svd.ev_cum.shape == (4,)
    assert_allclose(svd.ev_ratio.data, [2500.0 / 34.0, 900.0 / 34.0, 0.0, 0.0])
    assert_allclose(svd.ev_cum.data, [2500.0 / 34.0, 100.0, 100.0, 100.0])
    # repr shows U and VT not computed
    assert "U and VT not computed" in repr(svd)

    # full_matrices=True + compute_uv=False
    svd2 = SVD(full_matrices=True, compute_uv=False)
    svd2.fit(dataset)
    assert svd2.U is None
    assert svd2.VT is None
    assert_allclose(svd2.s, [5.0, 3.0, 0.0, 0.0])


# --------------------------------------------------------------------------------------
# transform / fit_transform (issue #1972)
# --------------------------------------------------------------------------------------
@pytest.fixture()
def random_dataset():
    rng = np.random.default_rng(0)
    y = scp.Coord.linspace(0.0, 50.0, 6, title="time", units="s")
    x = scp.Coord.linspace(4000.0, 1000.0, 20, title="wavenumber", units="cm^-1")
    return scp.NDDataset(
        rng.random((6, 20)),
        coordset=[y, x],
        units="absorbance",
        title="absorbance",
        name="source",
    )


def test_svd_transform_fitted_data_equals_U_sigma(random_dataset):
    X = random_dataset
    svd = SVD().fit(X)
    expected = svd.U * svd.s

    scores = svd.transform()

    assert isinstance(scores, scp.NDDataset)
    assert scores.shape == (6, 6)
    assert scores.dims == ["y", "k"]
    assert_allclose(scores.data, expected, atol=1e-12)
    assert_allclose(scores.y.data, X.y.data)
    assert scores.y.title == X.y.title
    assert scores.y.units == X.y.units
    assert scores.k.title == "components"
    assert scores.k.size == 6
    assert scores.title == "scores"
    assert scores.units is None
    # explicit X gives the same scores
    assert_allclose(svd.transform(X).data, expected, atol=1e-12)
    # the input dataset is not modified
    assert X.shape == (6, 20)
    assert X.units == "absorbance"


def test_svd_transform_new_data_projects_on_V_without_centering(random_dataset):
    svd = SVD().fit(random_dataset)
    X_new = scp.NDDataset(np.random.default_rng(1).random((3, 20)) + 10.0)

    scores = svd.transform(X_new)

    assert scores.shape == (3, 6)
    assert_allclose(scores.data, X_new.data @ svd.VT.T)


def test_svd_transform_n_components(random_dataset):
    svd = SVD().fit(random_dataset)
    full = svd.transform()

    scores = svd.transform(n_components=2)

    assert scores.shape == (6, 2)
    assert scores.k.size == 2
    assert_allclose(scores.data, full.data[:, :2])

    with pytest.warns(UserWarning, match="cannot be greater"):
        too_many = svd.transform(n_components=10)
    assert too_many.shape == (6, 6)


def test_svd_fit_transform_equals_fit_then_transform(random_dataset):
    scores = SVD().fit_transform(random_dataset)
    expected = SVD().fit(random_dataset).transform()

    assert_allclose(scores.data, expected.data)
    assert scores.dims == expected.dims
    assert_allclose(scores.y.data, expected.y.data)

    truncated = SVD().fit_transform(random_dataset, n_components=3)
    assert truncated.shape == (6, 3)
    assert_allclose(truncated.data, expected.data[:, :3])


def test_svd_transform_full_matrices_keeps_min_dimension(random_dataset):
    svd = SVD(full_matrices=True).fit(random_dataset)
    assert svd.VT.shape == (20, 20)

    scores = svd.transform()

    assert scores.shape == (6, 6)
    assert_allclose(np.abs(scores.data), np.abs(svd.U * svd.s), atol=1e-12)
    assert_allclose(scores.data, random_dataset.data @ svd.VT[:6].T)


def test_svd_transform_masked_rows_and_columns(random_dataset):
    masked = random_dataset.copy()
    masked[:, 4] = scp.MASKED
    masked[2] = scp.MASKED
    svd = SVD().fit(masked)
    assert svd.U.shape == (5, 5)
    assert svd.VT.shape == (5, 19)

    scores = svd.transform()

    # the masked row is restored as a fully masked score row
    assert scores.shape == (6, 5)
    assert np.all(scores.mask[2])
    assert not np.any(scores.mask[[0, 1, 3, 4, 5]])
    assert_allclose(scores.data[[0, 1, 3, 4, 5]], svd.U * svd.s, atol=1e-12)
    assert_allclose(scores.y.data, random_dataset.y.data)

    # new data with the same masked feature are projected on the kept columns
    X_new = scp.NDDataset(np.random.default_rng(2).random((3, 20)))
    X_new[:, 4] = scp.MASKED
    keep = [i for i in range(20) if i != 4]
    assert_allclose(svd.transform(X_new).data, X_new.data[:, keep] @ svd.VT.T)

    # new data with a different number of features are rejected
    with pytest.raises(ValueError, match="features"):
        svd.transform(random_dataset)


def test_svd_components_are_VT_rows_with_feature_coords(random_dataset):
    X = random_dataset
    svd = SVD().fit(X)

    components = svd.components

    assert isinstance(components, scp.NDDataset)
    assert components.shape == (6, 20)
    assert components.dims == ["k", "x"]
    assert_allclose(components.data, svd.VT)
    assert_allclose(components.x.data, X.x.data)
    assert components.x.title == X.x.title
    assert components.x.units == X.x.units
    assert components.k.title == "components"
    assert components.title == "components"
    assert components.name == "source_SVD.components"
    assert components.units is None
    assert not np.any(components.mask)

    truncated = svd.get_components(n_components=2)
    assert truncated.shape == (2, 20)
    assert_allclose(truncated.data, svd.VT[:2])

    # with full_matrices=True only the min(M, N) singular vectors are components
    full = SVD(full_matrices=True).fit(X)
    assert full.VT.shape == (20, 20)
    assert full.components.shape == (6, 20)
    assert_allclose(full.components.data, full.VT[:6])


def test_svd_inverse_transform_full_rank_recovers_X(random_dataset):
    X = random_dataset
    svd = SVD().fit(X)

    X_hat = svd.inverse_transform(svd.transform())

    assert X_hat.shape == X.shape
    assert_allclose(X_hat.data, X.data, atol=1e-12)
    assert X_hat.units == X.units
    assert X_hat.title == "reconstruction"
    assert X_hat.dims == ["y", "x"]

    X_hat = svd.inverse_transform()
    assert_allclose(X_hat.data, X.data, atol=1e-12)
    assert X_hat.name == "source_SVD.reconstruction"
    assert X_hat.units == X.units
    assert_allclose(X_hat.x.data, X.x.data)
    assert_allclose(X_hat.y.data, X.y.data)
    assert X_hat.x.title == X.x.title
    assert X_hat.y.title == X.y.title


def test_svd_inverse_transform_truncated(random_dataset):
    svd = SVD().fit(random_dataset)
    U, s, VT = svd.U, svd.s, svd.VT
    k = 2
    expected = (U[:, :k] * s[:k]) @ VT[:k]

    # n_components is honoured when the scores are computed internally ...
    assert_allclose(svd.inverse_transform(n_components=k).data, expected)
    # ... from truncated scores ...
    assert_allclose(svd.inverse_transform(svd.transform(n_components=k)).data, expected)
    # ... and from full scores with an explicit n_components
    assert_allclose(
        svd.inverse_transform(svd.transform(), n_components=k).data, expected
    )
    # raw ndarray scores are accepted
    assert_allclose(svd.inverse_transform(U[:, :k] * s[:k]).data, expected)

    with pytest.warns(UserWarning, match="fitted model components"):
        X_hat = svd.inverse_transform(n_components=10)
    assert_allclose(X_hat.data, random_dataset.data, atol=1e-12)


def test_svd_components_and_inverse_transform_masked(random_dataset):
    masked = random_dataset.copy()
    masked[:, 4] = scp.MASKED
    masked[2] = scp.MASKED
    svd = SVD().fit(masked)

    components = svd.components
    assert components.shape == (5, 20)
    assert np.all(components.mask[:, 4])
    assert not np.any(np.delete(components.mask, 4, axis=1))
    assert_allclose(np.delete(components.data, 4, axis=1), svd.VT)

    X_hat = svd.inverse_transform()
    assert X_hat.shape == masked.shape
    assert_allclose(X_hat.mask, masked.mask)
    keep = ~masked.mask
    assert_allclose(X_hat.data[keep], masked.data[keep], atol=1e-12)


def test_svd_transform_errors(random_dataset):
    with pytest.raises(NotFittedError):
        SVD().transform(random_dataset)
    with pytest.raises(NotFittedError):
        SVD().inverse_transform()

    svd = SVD(compute_uv=False).fit(random_dataset)
    with pytest.raises(ValueError, match="compute_uv=True"):
        svd.transform()
    with pytest.raises(ValueError, match="compute_uv=True"):
        _ = svd.components
    with pytest.raises(ValueError, match="compute_uv=True"):
        svd.get_components(n_components=2)
    with pytest.raises(ValueError, match="compute_uv=True"):
        svd.inverse_transform(np.ones((6, 2)))


def test_svd_methods_follow_fitted_state_not_current_compute_uv(random_dataset):
    svd = SVD(compute_uv=False, full_matrices=False).fit(random_dataset)
    svd.compute_uv = True
    with pytest.raises(ValueError, match="compute_uv=True"):
        svd.transform()
    with pytest.raises(ValueError, match="compute_uv=True"):
        _ = svd.components

    svd = SVD(compute_uv=True, full_matrices=False).fit(random_dataset)
    expected = svd.U * svd.s
    svd.compute_uv = False
    assert_allclose(svd.transform().data, expected, atol=1e-12)


def test_svd_fit_transform_accepts_ignored_Y(random_dataset):
    scores = SVD(compute_uv=True, full_matrices=False).fit_transform(
        random_dataset, None
    )
    assert scores.shape == (6, 6)
    assert SVD().fit(random_dataset).n_components == 6


def test_svd_rejects_different_masked_feature_support(random_dataset):
    training = random_dataset.copy()
    training[:, 4] = scp.MASKED
    model = SVD().fit(training)
    new = random_dataset.copy()
    new[:, 7] = scp.MASKED
    with pytest.raises(ValueError, match="feature columns must match"):
        model.transform(new)
    assert_allclose(model.inverse_transform().mask, training.mask)


def test_svd_new_data_does_not_replace_fit_data(random_dataset):
    model = SVD().fit(random_dataset)
    expected = model.transform().copy()
    new = random_dataset[:3].copy()
    new.data = new.data + 10
    new.name = "new"
    new.y = scp.Coord([20, 21, 22], title="new observations")
    scores = model.transform(new)
    assert scores.shape == (3, 6)
    assert_allclose(scores.y.data, new.y.data)
    assert scores.name == "new_SVD.scores"
    assert_allclose(model.transform().data, expected.data)
    assert_allclose(model.inverse_transform().data, random_dataset.data, atol=1e-12)
    with pytest.raises(ValueError):
        model.transform(np.ones((3, 2)))
    assert_allclose(model.transform().data, expected.data)


def test_svd_explicit_masked_scores_roundtrip(random_dataset):
    training = random_dataset.copy()
    training[:, 4] = scp.MASKED
    training[2] = scp.MASKED
    model = SVD().fit(training)
    scores = model.transform()
    restored = model.inverse_transform(scores)
    assert restored.shape == training.shape
    assert_allclose(restored.mask, training.mask)
    assert_allclose(
        restored.data[~training.mask], training.data[~training.mask], atol=1e-12
    )
    assert_allclose(restored.x.data, training.x.data)
    assert_allclose(restored.y.data, training.y.data)
    assert restored.units == training.units


def test_svd_new_masked_scores_restore_their_observations(random_dataset):
    training = random_dataset.copy()
    training[:, 4] = scp.MASKED
    model = SVD().fit(training)
    new = training[:3].copy()
    new[1] = scp.MASKED
    new.y = scp.Coord([20, 21, 22])
    restored = model.inverse_transform(model.transform(new))
    assert restored.shape == new.shape
    assert_allclose(restored.mask, new.mask)
    assert_allclose(restored.data[~new.mask], new.data[~new.mask], atol=1e-12)
    assert_allclose(restored.y.data, new.y.data)
    assert_allclose(restored.x.data, training.x.data)
    assert_allclose(
        model.inverse_transform().data[:, :4], training.data[:, :4], atol=1e-12
    )


def test_svd_explicit_scores_without_coordinates():
    original = np.random.default_rng(4).random((6, 20))
    model = SVD().fit(original)
    restored = model.inverse_transform(model.transform())
    assert restored.shape == original.shape
    assert_allclose(restored.data, original, atol=1e-12)


@pytest.mark.parametrize("reference", ["x", "y"])
def test_svd_explicit_reconstruction_resolves_coordinate_references(reference):
    X = scp.NDDataset(
        np.random.default_rng(4).random((4, 4)),
        coordset=scp.CoordSet(
            **(
                {"y": scp.Coord(range(4)), "x": "y"}
                if reference == "x"
                else {"x": scp.Coord(range(4)), "y": "x"}
            )
        ),
        name="training",
    )
    X.author = "original author"
    model = SVD().fit(X)
    scores = model.transform()
    assert scores.author == X.author
    assert_allclose(model.components.data, model.VT)
    assert_allclose(model.get_components().data, model.VT)
    restored = model.inverse_transform(scores)
    assert restored.author == X.author
    assert model.inverse_transform().author == X.author
    assert_allclose(restored.data, X.data, atol=1e-12)
    assert_allclose(restored.x.data, np.arange(4))
    assert_allclose(restored.y.data, np.arange(4))
    assert restored.name == "training_SVD.reconstruction"
    assert "fit of training." in restored.description


def test_svd_reconstruction_disambiguates_axis_names():
    rng = np.random.default_rng(2)
    training = scp.NDDataset(
        rng.random((4, 5)),
        coordset=[scp.Coord(range(4)), scp.Coord(range(5))],
    )
    model = SVD().fit(training)
    new = scp.NDDataset(
        rng.random((3, 5)),
        dims=["x", "z"],
        coordset=scp.CoordSet(x=scp.Coord([10, 20, 30]), z=scp.Coord(range(5))),
    )
    new[1] = scp.MASKED
    scores = model.transform(new)
    restored = model.inverse_transform(scores)
    assert restored.dims == ["y", "x"]
    assert_allclose(restored.coord(0).data, [10, 20, 30])
    assert_allclose(restored.coord(1).data, np.arange(5))
    assert_allclose(restored.mask, new.mask)
    expected = scores.data @ model.VT
    assert_allclose(restored.data[~new.mask], expected[~new.mask])
