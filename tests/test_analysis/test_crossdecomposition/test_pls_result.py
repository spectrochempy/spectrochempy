# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================

"""Tests for the PLSRegression result property."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from spectrochempy.analysis._base._result import AnalysisResult
from spectrochempy.analysis.crossdecomposition.pls import PLSRegression
from spectrochempy.utils.exceptions import NotFittedError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def pls_fitted():
    """Return a fitted PLSRegression with default parameters."""
    rng = np.random.default_rng(42)
    n_obs = 20
    n_features = 10
    n_components = 3
    n_targets = 2

    X = rng.normal(0, 1, (n_obs, n_features))
    Y = rng.normal(0, 1, (n_obs, n_targets))

    import spectrochempy as scp

    X_nd = scp.NDDataset(X)
    Y_nd = scp.NDDataset(Y)

    pls = PLSRegression(n_components=n_components)
    pls.fit(X_nd, Y_nd)
    return pls


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPLSRegressionResult:
    """Tests for PLSRegression.result returning AnalysisResult."""

    def test_result_is_analysis_result(self, pls_fitted):
        result = pls_fitted.result
        assert isinstance(result, AnalysisResult)

    def test_estimator_name(self, pls_fitted):
        assert pls_fitted.result.estimator == "PLSRegression"

    def test_outputs_contain_keys(self, pls_fitted):
        outputs = pls_fitted.result.outputs
        expected = {
            "x_scores",
            "x_loadings",
            "x_weights",
            "x_rotations",
            "y_scores",
            "y_loadings",
            "y_weights",
            "y_rotations",
            "coef",
            "intercept",
        }
        assert set(outputs.keys()) == expected

    def test_outputs_do_not_contain_components(self, pls_fitted):
        assert "components" not in pls_fitted.result.outputs

    def test_output_values_match_properties(self, pls_fitted):
        pls = pls_fitted
        outputs = pls.result.outputs
        assert_allclose(outputs["x_scores"].data, pls.x_scores.data)
        assert_allclose(outputs["x_loadings"].data, pls.x_loadings.data)
        assert_allclose(outputs["x_weights"].data, pls.x_weights.data)
        assert_allclose(outputs["x_rotations"].data, pls.x_rotations.data)
        assert_allclose(outputs["y_scores"].data, pls.y_scores.data)
        assert_allclose(outputs["y_loadings"].data, pls.y_loadings.data)
        assert_allclose(outputs["y_weights"].data, pls.y_weights.data)
        assert_allclose(outputs["y_rotations"].data, pls.y_rotations.data)
        assert_allclose(outputs["coef"].data, pls.coef.data)
        assert_allclose(outputs["intercept"].data, pls.intercept.data)

    def test_diagnostics_contain_keys(self, pls_fitted):
        diagnostics = pls_fitted.result.diagnostics
        assert "n_iter" in diagnostics
        assert "n_features_in" in diagnostics

    def test_diagnostics_values(self, pls_fitted):
        pls = pls_fitted
        diagnostics = pls.result.diagnostics
        # sklearn n_iter_ is a list of ints per component
        assert isinstance(diagnostics["n_iter"], (list, np.ndarray))
        assert len(diagnostics["n_iter"]) == pls.n_components
        assert diagnostics["n_features_in"] == 10

    def test_parameters_contain_expected_keys(self, pls_fitted):
        parameters = pls_fitted.result.parameters
        expected = {"n_components", "scale", "max_iter", "tol"}
        assert set(parameters.keys()) == expected

    def test_parameters_values(self, pls_fitted):
        parameters = pls_fitted.result.parameters
        assert parameters["n_components"] == 3
        assert parameters["scale"] is True
        assert parameters["max_iter"] == 500
        assert parameters["tol"] == 1.0e-6

    def test_parameters_match_estimator_config(self, pls_fitted):
        pls = PLSRegression(
            n_components=2,
            scale=False,
            max_iter=100,
            tol=1.0e-4,
        )
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (20, 10))
        Y = rng.normal(0, 1, (20, 2))
        import spectrochempy as scp

        pls.fit(scp.NDDataset(X), scp.NDDataset(Y))
        params = pls.result.parameters
        assert params["n_components"] == 2
        assert params["scale"] is False
        assert params["max_iter"] == 100
        assert params["tol"] == 1.0e-4

    def test_repr_contains_expected_fields(self, pls_fitted):
        text = repr(pls_fitted.result)
        assert "AnalysisResult" in text
        assert "PLSRegression" in text
        assert "x_scores" in text
        assert "x_loadings" in text
        assert "coef" in text
        assert "n_iter" in text
        assert "n_features_in" in text

    def test_repr_contains_parameters(self, pls_fitted):
        text = repr(pls_fitted.result)
        assert "parameters:" in text
        assert "n_components" in text

    def test_result_is_not_cached(self, pls_fitted):
        assert pls_fitted.result is not pls_fitted.result

    def test_raises_before_fit(self):
        pls = PLSRegression(n_components=3)
        with pytest.raises(NotFittedError):
            _ = pls.result

    def test_fit_still_returns_self(self, pls_fitted):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (20, 10))
        Y = rng.normal(0, 1, (20, 2))
        import spectrochempy as scp

        pls = PLSRegression(n_components=3)
        ret = pls.fit(scp.NDDataset(X), scp.NDDataset(Y))
        assert ret is pls

    def test_existing_properties_unchanged(self, pls_fitted):
        pls = pls_fitted
        assert pls._fitted
        assert pls.n_components == 3
        assert pls.x_loadings.shape == (3, 10)
        assert pls.x_scores.shape == (20, 3)
        assert pls.n_iter is not None

    def test_n_iter_works_after_fit(self, pls_fitted):
        n_iter = pls_fitted.n_iter
        assert n_iter is not None
        assert len(n_iter) == 3

    def test_components_raises_clear_error(self, pls_fitted):
        with pytest.raises(AttributeError, match="no single 'components' matrix"):
            _ = pls_fitted.components

    def test_intercept_in_result_is_nddataset(self, pls_fitted):
        assert hasattr(pls_fitted.result.outputs["intercept"], "shape")

    def test_intercept_property_handles_y_without_coordset(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (20, 10))
        Y = rng.normal(0, 1, (20, 2))
        import spectrochempy as scp

        pls = PLSRegression(n_components=3)
        pls.fit(scp.NDDataset(X), scp.NDDataset(Y))
        intercept = pls.intercept
        assert hasattr(intercept, "shape")
        assert intercept.shape == (2,)

    def test_result_intercept_matches_property(self, pls_fitted):
        pls = pls_fitted
        assert_allclose(
            pls.result.outputs["intercept"].data,
            pls.intercept.data,
        )

    def test_coef_is_nddataset(self, pls_fitted):
        assert hasattr(pls_fitted.coef, "shape")

    def test_all_outputs_are_nddataset(self, pls_fitted):
        for key, value in pls_fitted.result.outputs.items():
            assert hasattr(value, "shape"), f"{key} is not an NDDataset"
