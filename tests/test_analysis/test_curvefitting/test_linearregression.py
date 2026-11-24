# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

"""Tests for the coefficient and intercept outputs of LSTSQ and NNLS."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import spectrochempy as scp

C_TRUE = np.array([[1.0, 0.5], [0.2, 0.8], [0.6, 0.6]])
OFFSETS = np.array([0.1, 0.2, 0.3])


@pytest.fixture()
def pure_and_mixtures():
    x = scp.Coord.linspace(1480.0, 1430.0, 101, title="wavenumbers", units="cm^-1")
    pure = [
        scp.gaussian(x, pos=p, width=8.0, normalized=False).data
        for p in (1452.0, 1458.0)
    ]
    S = scp.NDDataset(
        pure, coordset=[scp.Coord([0.0, 1.0], title="component"), x], title="S"
    )
    Y = scp.NDDataset(
        C_TRUE @ S.data,
        coordset=[scp.Coord([0.0, 1.0, 2.0], title="sample"), x],
        title="Y",
    )
    return S, Y


@pytest.mark.parametrize("model", [scp.NNLS, scp.LSTSQ])
def test_coef_multi_target_with_coordinates(model, pure_and_mixtures):
    S, Y = pure_and_mixtures

    coef = model(fit_intercept=False).fit(S.T, Y.T).coef

    assert isinstance(coef, scp.NDDataset)
    assert coef.shape == (3, 2)
    assert list(coef.dims) == ["y", "x"]
    assert coef.coord("y").title == "sample"
    assert_allclose(coef.coord("y").data, [0.0, 1.0, 2.0])
    assert coef.coord("x").title == "component"
    assert_allclose(coef.coord("x").data, [0.0, 1.0])
    assert_allclose(coef.data, C_TRUE, atol=1e-10)


@pytest.mark.parametrize("model", [scp.NNLS, scp.LSTSQ])
def test_coef_multi_target_without_coordinates(model, pure_and_mixtures):
    S, Y = pure_and_mixtures

    coef = model(fit_intercept=False).fit(scp.NDDataset(S.data.T), Y.data.T).coef

    assert isinstance(coef, scp.NDDataset)
    assert coef.shape == (3, 2)
    assert list(coef.dims) == ["y", "x"]
    assert coef.coordset is None
    assert_allclose(coef.data, C_TRUE, atol=1e-10)


@pytest.mark.parametrize("model", [scp.NNLS, scp.LSTSQ])
def test_coef_and_intercept_multi_target_with_ndarray(model, pure_and_mixtures):
    S, Y = pure_and_mixtures

    estimator = model(fit_intercept=True).fit(S.data.T, Y.data.T + OFFSETS)

    assert isinstance(estimator.coef, np.ndarray)
    assert_allclose(estimator.coef, C_TRUE, atol=1e-8)
    assert isinstance(estimator.intercept, np.ndarray)
    assert_allclose(estimator.intercept, OFFSETS, atol=1e-8)


def test_coef_keeps_distinct_source_dimension_names(pure_and_mixtures):
    S, Y = pure_and_mixtures
    X = scp.NDDataset(
        S.data.T, dims=["x", "u"], coordset=[None, scp.Coord([0.0, 1.0], title="c")]
    )

    coef = scp.LSTSQ(fit_intercept=False).fit(X, Y.T).coef

    assert list(coef.dims) == ["y", "u"]
    assert coef.coord("y").title == "sample"
    assert coef.coord("u").title == "c"


@pytest.mark.parametrize("model", [scp.NNLS, scp.LSTSQ])
def test_coef_single_target_multi_feature(model, pure_and_mixtures):
    S, Y = pure_and_mixtures

    coef = model(fit_intercept=False).fit(S.T, Y[0].squeeze()).coef

    assert coef.shape == (2,)
    assert coef.ndim == 1
    assert coef.coord(coef.dims[0]).title == "component"
    assert_allclose(coef.data, C_TRUE[0], atol=1e-10)


@pytest.mark.parametrize("model", [scp.NNLS, scp.LSTSQ])
def test_intercept_multi_target_with_coordinates(model, pure_and_mixtures):
    S, Y = pure_and_mixtures

    intercept = model(fit_intercept=True).fit(S.T, (Y + OFFSETS[:, None]).T).intercept

    assert isinstance(intercept, scp.NDDataset)
    assert intercept.shape == (3,)
    assert intercept.coord(intercept.dims[0]).title == "sample"
    assert_allclose(intercept.data, OFFSETS, atol=1e-8)


@pytest.mark.parametrize("model", [scp.NNLS, scp.LSTSQ])
def test_intercept_without_fit_intercept_is_zero(model, pure_and_mixtures):
    S, Y = pure_and_mixtures

    assert model(fit_intercept=False).fit(S.T, Y.T).intercept == 0.0


@pytest.mark.parametrize("model", [scp.NNLS, scp.LSTSQ])
@pytest.mark.parametrize("y_has_observation_coord", [False, True])
def test_coef_and_intercept_with_observation_coordinates_only(
    model, y_has_observation_coord
):
    observations = scp.Coord([1.0, 2.0, 3.0], title="sample")
    X = scp.NDDataset([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    X.set_coordset(y=observations)
    Y = scp.NDDataset([[1.0, 2.0], [3.0, 4.0], [4.0, 6.0]])
    if y_has_observation_coord:
        Y.set_coordset(y=observations.copy())

    fitted = model(fit_intercept=False).fit(X, Y)
    coef = fitted.coef

    assert list(coef.dims) == ["y", "x"]
    assert_allclose(coef.data, [[1.0, 3.0], [2.0, 4.0]], atol=1e-10)
    names = coef.coordset.names if coef.coordset is not None else []
    for dim in ("y", "x"):
        assert dim not in names or coef.coordset[dim].data is None

    intercept = model(fit_intercept=True).fit(X, Y).intercept
    assert list(intercept.dims) == ["x"]
    assert intercept.shape == (2,)
