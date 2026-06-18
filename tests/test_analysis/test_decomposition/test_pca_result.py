# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
"""
Tests for the PCA result object contract prototype.
"""

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.analysis._base._analysisbase import NotFittedError
from spectrochempy.analysis._base._result import AnalysisResult
from spectrochempy.analysis._base._result import FitResult
from spectrochempy.analysis._base._result import ResultBase
from spectrochempy.analysis.decomposition.pca import PCA


# ======================================================================================
# Fixtures
# ======================================================================================
@pytest.fixture()
def low_rank_dataset():
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


# ======================================================================================
# ResultBase and AnalysisResult creation tests
# ======================================================================================
class TestResultBase:
    def test_minimal_creation(self):
        result = ResultBase(estimator="TestEstimator")
        assert result.estimator == "TestEstimator"
        assert result.outputs == {}
        assert result.diagnostics == {}
        assert result.parameters == {}

    def test_with_outputs_and_diagnostics(self):
        result = ResultBase(
            estimator="Test",
            outputs={"a": np.ones(3), "b": np.ones((2, 3))},
            diagnostics={"d": np.array([1.0, 2.0])},
        )
        assert "a" in result.outputs
        assert "b" in result.outputs
        assert "d" in result.diagnostics

    def test_repr(self):
        result = ResultBase(
            estimator="Test",
            outputs={"x": np.ones((3, 4))},
            diagnostics={"y": np.ones(2)},
        )
        text = repr(result)
        assert "ResultBase" in text
        assert "Test" in text
        assert "x" in text
        assert "(3, 4)" in text
        assert "y" in text


class TestAnalysisResult:
    def test_creation(self):
        result = AnalysisResult(estimator="PCA")
        assert isinstance(result, AnalysisResult)
        assert isinstance(result, ResultBase)

    def test_repr(self):
        result = AnalysisResult(
            estimator="PCA",
            outputs={"scores": np.ones((6, 5))},
            diagnostics={"explained_variance": np.ones(5)},
        )
        text = repr(result)
        assert "AnalysisResult" in text
        assert "PCA" in text


class TestFitResult:
    def test_creation(self):
        result = FitResult(estimator="Optimize")
        assert isinstance(result, FitResult)
        assert isinstance(result, ResultBase)


# ======================================================================================
# PCA result integration tests
# ======================================================================================
class TestPCAResult:
    def test_result_is_analysis_result(self, low_rank_dataset):
        pca = PCA()
        pca.fit(low_rank_dataset)
        result = pca.result
        assert isinstance(result, AnalysisResult)

    def test_estimator_name(self, low_rank_dataset):
        pca = PCA()
        pca.fit(low_rank_dataset)
        assert pca.result.estimator == "PCA"

    def test_outputs_contain_keys(self, low_rank_dataset):
        pca = PCA()
        pca.fit(low_rank_dataset)
        result = pca.result
        for name in ("scores", "loadings", "components"):
            assert name in result.outputs, f"{name} missing from result.outputs"

    def test_diagnostics_contain_keys(self, low_rank_dataset):
        pca = PCA()
        pca.fit(low_rank_dataset)
        result = pca.result
        for name in ("explained_variance", "explained_variance_ratio"):
            assert name in result.diagnostics, f"{name} missing from result.diagnostics"

    def test_outputs_match_properties(self, low_rank_dataset):
        pca = PCA()
        pca.fit(low_rank_dataset)
        result = pca.result
        np.testing.assert_array_equal(
            result.outputs["scores"].data,
            pca.scores.data,
        )
        np.testing.assert_array_equal(
            result.outputs["loadings"].data,
            pca.loadings.data,
        )
        np.testing.assert_array_equal(
            result.outputs["components"].data,
            pca.components.data,
        )

    def test_diagnostics_match_properties(self, low_rank_dataset):
        pca = PCA()
        pca.fit(low_rank_dataset)
        result = pca.result
        np.testing.assert_array_equal(
            result.diagnostics["explained_variance"].data,
            pca.explained_variance.data,
        )
        np.testing.assert_array_equal(
            result.diagnostics["explained_variance_ratio"].data,
            pca.explained_variance_ratio.data,
        )

    def test_repr_contains_expected_fields(self, low_rank_dataset):
        pca = PCA()
        pca.fit(low_rank_dataset)
        result = pca.result
        text = repr(result)
        assert "AnalysisResult" in text
        assert "PCA" in text
        assert "scores" in text
        assert "loadings" in text
        assert "components" in text
        assert "explained_variance" in text

    def test_result_is_not_cached(self, low_rank_dataset):
        pca = PCA()
        pca.fit(low_rank_dataset)
        assert pca.result is not pca.result, (
            "AnalysisResult is recreated on every access; "
            "change this assertion if caching is added later"
        )

    def test_parameters_contain_expected_keys(self, low_rank_dataset):
        pca = PCA()
        pca.fit(low_rank_dataset)
        params = pca.result.parameters
        for name in ("n_components", "standardized", "scaled", "whiten", "svd_solver"):
            assert name in params, f"{name} missing from result.parameters"

    def test_parameters_values(self, low_rank_dataset):
        pca = PCA()
        pca.fit(low_rank_dataset)
        params = pca.result.parameters
        assert params["n_components"] == 5
        assert params["standardized"] is False
        assert params["scaled"] is False
        assert params["whiten"] is False
        assert params["svd_solver"] == "auto"

    def test_parameters_match_estimator_config(self, low_rank_dataset):
        pca = PCA(n_components=3, whiten=True, svd_solver="full")
        pca.fit(low_rank_dataset)
        params = pca.result.parameters
        assert params["n_components"] == 3
        assert params["whiten"] is True
        assert params["svd_solver"] == "full"

    def test_repr_contains_parameters(self, low_rank_dataset):
        pca = PCA()
        pca.fit(low_rank_dataset)
        text = repr(pca.result)
        assert "parameters:" in text
        assert "n_components" in text
        assert "standardized" in text

    def test_raises_before_fit(self):
        pca = PCA()
        with pytest.raises(NotFittedError):
            _ = pca.result

    def test_fit_still_returns_self(self, low_rank_dataset):
        pca = PCA()
        ret = pca.fit(low_rank_dataset)
        assert ret is pca

    def test_existing_properties_unchanged(self, low_rank_dataset):
        pca = PCA()
        pca.fit(low_rank_dataset)
        assert pca.loadings.shape == (5, 5)
        assert pca.scores.shape == (6, 5)
        assert pca.components.shape == (5, 5)
        assert pca.n_components == 5
