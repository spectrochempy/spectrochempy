# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
"""
Tests for the SIMPLISMA result object.
"""

import numpy as np
import pytest

from spectrochempy.analysis._base._analysisbase import NotFittedError
from spectrochempy.analysis._base._result import AnalysisResult
from spectrochempy.analysis._base._result import ResultBase
from spectrochempy.analysis.decomposition.simplisma import SIMPLISMA


# ======================================================================================
# SIMPLISMA result integration tests
# ======================================================================================
class TestSIMPLISMAResult:
    def test_result_is_analysis_result(self, simplisma_dataset):
        sma = SIMPLISMA(n_components=2, log_level="WARNING")
        sma.fit(simplisma_dataset)
        result = sma.result
        assert isinstance(result, AnalysisResult)
        assert isinstance(result, ResultBase)

    def test_estimator_name(self, simplisma_dataset):
        sma = SIMPLISMA(n_components=2, log_level="WARNING")
        sma.fit(simplisma_dataset)
        assert sma.result.estimator == "SIMPLISMA"

    def test_outputs_contain_keys(self, simplisma_dataset):
        sma = SIMPLISMA(n_components=2, log_level="WARNING")
        sma.fit(simplisma_dataset)
        result = sma.result
        for name in ("C", "components", "Pt", "s"):
            assert name in result.outputs, f"{name} missing from result.outputs"

    def test_diagnostics_contain_keys(self, simplisma_dataset):
        sma = SIMPLISMA(n_components=2, log_level="WARNING")
        sma.fit(simplisma_dataset)
        result = sma.result
        for name in ("r_squared", "residual_std", "n_components_selected"):
            assert name in result.diagnostics, f"{name} missing from result.diagnostics"

    def test_output_values_match_properties(self, simplisma_dataset):
        sma = SIMPLISMA(n_components=2, log_level="WARNING")
        sma.fit(simplisma_dataset)
        result = sma.result
        np.testing.assert_array_equal(
            result.outputs["C"].data,
            sma.C.data,
        )
        np.testing.assert_array_equal(
            result.outputs["components"].data,
            sma.components.data,
        )
        np.testing.assert_array_equal(
            result.outputs["Pt"].data,
            sma.Pt.data,
        )
        np.testing.assert_array_equal(
            result.outputs["s"].data,
            sma.s.data,
        )

    def test_diagnostic_values(self, simplisma_dataset):
        sma = SIMPLISMA(n_components=2, log_level="WARNING")
        sma.fit(simplisma_dataset)
        result = sma.result
        assert isinstance(result.diagnostics["r_squared"], float)
        assert isinstance(result.diagnostics["residual_std"], float)
        assert result.diagnostics["n_components_selected"] == 2
        assert 0.0 < result.diagnostics["r_squared"] <= 1.0
        assert result.diagnostics["residual_std"] >= 0.0

    def test_repr_contains_expected_fields(self, simplisma_dataset):
        sma = SIMPLISMA(n_components=2, log_level="WARNING")
        sma.fit(simplisma_dataset)
        text = repr(sma.result)
        assert "AnalysisResult" in text
        assert "SIMPLISMA" in text
        assert "C" in text
        assert "components" in text
        assert "Pt" in text
        assert "s" in text
        assert "r_squared" in text
        assert "residual_std" in text
        assert "n_components_selected" in text

    def test_result_is_not_cached(self, simplisma_dataset):
        sma = SIMPLISMA(n_components=2, log_level="WARNING")
        sma.fit(simplisma_dataset)
        assert sma.result is not sma.result, (
            "AnalysisResult is recreated on every access; "
            "change this assertion if caching is added later"
        )

    def test_parameters_contain_expected_keys(self, simplisma_dataset):
        sma = SIMPLISMA(n_components=2, log_level="WARNING")
        sma.fit(simplisma_dataset)
        params = sma.result.parameters
        for name in ("interactive", "n_components", "tol", "noise"):
            assert name in params, f"{name} missing from result.parameters"

    def test_parameters_values(self, simplisma_dataset):
        sma = SIMPLISMA(n_components=2, log_level="WARNING")
        sma.fit(simplisma_dataset)
        params = sma.result.parameters
        assert params["interactive"] is False
        assert params["n_components"] == 2
        assert params["tol"] == 0.1
        assert params["noise"] == 3

    def test_parameters_match_estimator_config(self, simplisma_dataset):
        sma = SIMPLISMA(n_components=3, tol=0.5, noise=5, log_level="WARNING")
        sma.fit(simplisma_dataset)
        params = sma.result.parameters
        assert params["n_components"] == 3
        assert params["tol"] == 0.5
        assert params["noise"] == 5

    def test_repr_contains_parameters(self, simplisma_dataset):
        sma = SIMPLISMA(n_components=2, log_level="WARNING")
        sma.fit(simplisma_dataset)
        text = repr(sma.result)
        assert "parameters:" in text
        assert "n_components" in text
        assert "tol" in text
        assert "noise" in text

    def test_raises_before_fit(self):
        sma = SIMPLISMA()
        with pytest.raises(NotFittedError):
            _ = sma.result

    def test_fit_still_returns_self(self, simplisma_dataset):
        sma = SIMPLISMA(n_components=2, log_level="WARNING")
        ret = sma.fit(simplisma_dataset)
        assert ret is sma

    def test_existing_properties_unchanged(self, simplisma_dataset):
        sma = SIMPLISMA(n_components=2, log_level="WARNING")
        sma.fit(simplisma_dataset)
        assert sma.C.shape == (20, 2)
        assert sma.St.shape == (2, 100)
        assert sma.Pt.shape == (2, 100)
        assert sma.s.shape == (2, 100)
        assert sma.n_components == 2
