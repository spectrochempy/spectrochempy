"""Regression for coefficient access with array-valued regression inputs."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import spectrochempy as scp


@pytest.mark.parametrize("estimator", [scp.LSTSQ, scp.NNLS])
@pytest.mark.parametrize("targets", [1, 2])
def test_array_coefficients_are_finite_copies_for_multiple_features(estimator, targets):
    x = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.5]])
    coefficients = (
        np.array([2.0, 3.0]) if targets == 1 else np.array([[2.0, 3.0], [4.0, 1.0]])
    )
    y = x @ coefficients.T
    fit = estimator(fit_intercept=False).fit(x, y)
    result = fit.coef
    assert isinstance(result, np.ndarray)
    assert result.shape == coefficients.shape
    assert_allclose(result, coefficients, atol=1e-12)
    result[:] = -1
    assert_allclose(fit.coef, coefficients, atol=1e-12)
    assert_allclose(fit.predict(x), y, atol=1e-12)


def test_nnls_dataset_coefficients_keep_units_and_single_feature_keeps_scalar():
    x = scp.NDDataset([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], units="s")
    y = scp.NDDataset([2.0, 3.0, 5.0], units="V")
    fit = scp.NNLS(fit_intercept=False).fit(x, y)
    assert isinstance(fit.coef, scp.NDDataset)
    assert fit.coef.units == scp.ur.V / scp.ur.s
    assert_allclose(fit.coef.data, [2, 3], atol=1e-12)
    scalar = scp.NNLS(fit_intercept=False).fit([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]).coef
    assert isinstance(scalar, float)
    assert_allclose(scalar, 2.0)
