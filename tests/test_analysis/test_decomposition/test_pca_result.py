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

from spectrochempy.analysis._base._analysisbase import NotFittedError
from spectrochempy.analysis._base._result import AnalysisResult
from spectrochempy.analysis.decomposition.pca import PCA
from tests.test_analysis.result_test_helpers import assert_fit_returns_self
from tests.test_analysis.result_test_helpers import assert_result_basics
from tests.test_analysis.result_test_helpers import assert_result_raises_before_fit


# ======================================================================================
# PCA result integration tests
# ======================================================================================
class TestPCAResult:
    def test_result_is_analysis_result(self, low_rank_pca_dataset):
        pca = PCA()
        pca.fit(low_rank_pca_dataset)
        assert_result_basics(pca, AnalysisResult, "PCA")

    def test_outputs_contain_keys(self, low_rank_pca_dataset):
        pca = PCA()
        pca.fit(low_rank_pca_dataset)
        result = pca.result
        for name in ("scores", "loadings", "components"):
            assert name in result.outputs, f"{name} missing from result.outputs"

    def test_diagnostics_contain_keys(self, low_rank_pca_dataset):
        pca = PCA()
        pca.fit(low_rank_pca_dataset)
        result = pca.result
        for name in ("explained_variance", "explained_variance_ratio"):
            assert name in result.diagnostics, f"{name} missing from result.diagnostics"

    def test_outputs_match_properties(self, low_rank_pca_dataset):
        pca = PCA()
        pca.fit(low_rank_pca_dataset)
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

    def test_diagnostics_match_properties(self, low_rank_pca_dataset):
        pca = PCA()
        pca.fit(low_rank_pca_dataset)
        result = pca.result
        np.testing.assert_array_equal(
            result.diagnostics["explained_variance"].data,
            pca.explained_variance.data,
        )
        np.testing.assert_array_equal(
            result.diagnostics["explained_variance_ratio"].data,
            pca.explained_variance_ratio.data,
        )

    def test_result_attribute_access(self, low_rank_pca_dataset):
        pca = PCA(n_components=2).fit(low_rank_pca_dataset)
        result = pca.result
        assert result.scores is result.outputs["scores"]
        assert result.loadings is result.outputs["loadings"]
        assert result.explained_variance is result.diagnostics["explained_variance"]
        assert {"scores", "loadings", "explained_variance"} <= set(dir(result))

    def test_repr_contains_expected_fields(self, low_rank_pca_dataset):
        pca = PCA()
        pca.fit(low_rank_pca_dataset)
        result = pca.result
        text = repr(result)
        assert "AnalysisResult" in text
        assert "PCA" in text
        assert "scores" in text
        assert "loadings" in text
        assert "components" in text
        assert "explained_variance" in text

    def test_parameters_contain_expected_keys(self, low_rank_pca_dataset):
        pca = PCA()
        pca.fit(low_rank_pca_dataset)
        params = pca.result.parameters
        for name in ("n_components", "standardized", "scaled", "whiten", "svd_solver"):
            assert name in params, f"{name} missing from result.parameters"

    def test_parameters_values(self, low_rank_pca_dataset):
        pca = PCA()
        pca.fit(low_rank_pca_dataset)
        params = pca.result.parameters
        assert params["n_components"] == 5
        assert params["standardized"] is False
        assert params["scaled"] is False
        assert params["whiten"] is False
        assert params["svd_solver"] == "auto"

    def test_parameters_match_estimator_config(self, low_rank_pca_dataset):
        pca = PCA(n_components=3, whiten=True, svd_solver="full")
        pca.fit(low_rank_pca_dataset)
        params = pca.result.parameters
        assert params["n_components"] == 3
        assert params["whiten"] is True
        assert params["svd_solver"] == "full"

    def test_repr_contains_parameters(self, low_rank_pca_dataset):
        pca = PCA()
        pca.fit(low_rank_pca_dataset)
        text = repr(pca.result)
        assert "parameters:" in text
        assert "n_components" in text
        assert "standardized" in text

    def test_raises_before_fit(self):
        pca = PCA()
        assert_result_raises_before_fit(pca, NotFittedError)

    def test_fit_still_returns_self(self, low_rank_pca_dataset):
        pca = PCA()
        assert_fit_returns_self(pca, low_rank_pca_dataset)

    def test_existing_properties_unchanged(self, low_rank_pca_dataset):
        pca = PCA()
        pca.fit(low_rank_pca_dataset)
        assert pca.loadings.shape == (5, 5)
        assert pca.scores.shape == (6, 5)
        assert pca.components.shape == (5, 5)
        assert pca.n_components == 5
