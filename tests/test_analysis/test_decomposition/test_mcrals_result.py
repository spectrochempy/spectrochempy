# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
"""
Tests for the MCRALS result object.
"""

import numpy as np
import pytest

from spectrochempy.analysis._base._analysisbase import NotFittedError
from spectrochempy.analysis._base._result import AnalysisResult
from spectrochempy.analysis._base._result import ResultBase
from spectrochempy.analysis.decomposition.mcrals import MCRALS
from tests.test_analysis.result_test_helpers import assert_fit_returns_self
from tests.test_analysis.result_test_helpers import assert_result_basics
from tests.test_analysis.result_test_helpers import assert_result_raises_before_fit


# ======================================================================================
# Fixtures
# ======================================================================================
@pytest.fixture()
def mcrals_dataset():
    """Small synthetic non-negative dataset and initial guess for MCRALS."""
    rng = np.random.RandomState(42)
    C_true = rng.rand(10, 2)
    St_true = rng.rand(2, 6)
    D = C_true @ St_true
    X = D + 0.01 * rng.randn(10, 6)
    C0 = np.abs(C_true + 0.1 * rng.randn(10, 2))
    return X, C0


@pytest.fixture()
def fitted_mcrals(mcrals_dataset):
    X, C0 = mcrals_dataset
    mcr = MCRALS()
    mcr.fit(X, C0)
    return mcr


# ======================================================================================
# MCRALS result integration tests
# ======================================================================================
class TestMCRALSResult:
    def test_result_is_analysis_result(self, fitted_mcrals):
        result = assert_result_basics(fitted_mcrals, AnalysisResult, "MCRALS")
        assert isinstance(result, ResultBase)

    def test_outputs_contain_keys(self, fitted_mcrals):
        result = fitted_mcrals.result
        for name in ("C", "components"):
            assert name in result.outputs, f"{name} missing from result.outputs"
        assert "St" not in result.outputs

    def test_diagnostics_contain_keys(self, fitted_mcrals):
        result = fitted_mcrals.result
        for name in ("n_iter", "change", "residual_std", "converged"):
            assert name in result.diagnostics, f"{name} missing from result.diagnostics"

    def test_output_values_match_properties(self, fitted_mcrals):
        result = fitted_mcrals.result
        np.testing.assert_array_equal(
            result.outputs["C"].data,
            fitted_mcrals.C.data,
        )
        np.testing.assert_array_equal(
            result.outputs["components"].data,
            fitted_mcrals.components.data,
        )

    def test_diagnostic_values(self, fitted_mcrals):
        result = fitted_mcrals.result
        assert result.diagnostics["n_iter"] >= 1
        assert isinstance(result.diagnostics["converged"], (bool, np.bool_))
        assert isinstance(result.diagnostics["change"], float)
        assert isinstance(result.diagnostics["residual_std"], float)
        assert result.diagnostics["residual_std"] >= 0.0

    def test_repr_contains_expected_fields(self, fitted_mcrals):
        text = repr(fitted_mcrals.result)
        assert "AnalysisResult" in text
        assert "MCRALS" in text
        assert "C" in text
        assert "components" in text
        assert "n_iter" in text
        assert "converged" in text

    def test_parameters_contain_expected_keys(self, fitted_mcrals):
        params = fitted_mcrals.result.parameters
        for name in (
            "n_components",
            "max_iter",
            "tol",
            "maxdiv",
            "solverConc",
            "solverSpec",
            "nonnegConc",
            "nonnegSpec",
            "unimodConc",
            "unimodSpec",
            "unimodConcMod",
            "unimodSpecMod",
            "unimodConcTol",
            "unimodSpecTol",
            "closureConc",
            "closureTarget",
            "closureMethod",
            "monoDecConc",
            "monoDecTol",
            "monoIncConc",
            "monoIncTol",
            "hardConc",
            "hardSpec",
            "normSpec",
            "storeIterations",
        ):
            assert name in params, f"{name} missing from result.parameters"

    def test_parameters_values(self, fitted_mcrals):
        params = fitted_mcrals.result.parameters
        assert params["n_components"] == 2
        assert params["max_iter"] == 50
        assert params["tol"] == 0.1
        assert params["maxdiv"] == 5
        assert params["solverConc"] == "lstsq"
        assert params["solverSpec"] == "lstsq"
        assert params["nonnegConc"] == [0, 1]
        assert params["nonnegSpec"] == [0, 1]
        assert params["unimodConc"] == [0, 1]
        assert params["unimodSpec"] == []
        assert params["unimodConcMod"] == "strict"
        assert params["unimodSpecMod"] == "strict"
        assert params["unimodConcTol"] == 1.1
        assert params["unimodSpecTol"] == 1.1
        assert params["closureConc"] == []
        assert params["closureTarget"].startswith("<array shape=")
        assert params["closureMethod"] == "scaling"
        assert params["monoDecConc"] == []
        assert params["monoDecTol"] == 1.1
        assert params["monoIncConc"] == []
        assert params["monoIncTol"] == 1.1
        assert params["hardConc"] == []
        assert params["hardSpec"] == []
        assert params["normSpec"] is None
        assert params["storeIterations"] is False

    def test_parameters_match_estimator_config(self):
        X, C0 = _synthetic_data(rng=np.random.RandomState(0), n_rows=10, n_cols=6)
        mcr = MCRALS(
            max_iter=100,
            tol=1e-4,
            maxdiv=10,
            nonnegConc=[],
            unimodSpec=[0],
            normSpec="max",
            closureConc=[0, 1],
            closureMethod="constantSum",
        )
        mcr.fit(X, C0)
        params = mcr.result.parameters
        assert params["n_components"] == 2
        assert params["max_iter"] == 100
        assert params["tol"] == 1e-4
        assert params["maxdiv"] == 10
        assert params["nonnegConc"] == []
        assert params["unimodSpec"] == [0]
        assert params["normSpec"] == "max"
        assert params["closureConc"] == [0, 1]
        assert params["closureMethod"] == "constantSum"

    def test_repr_contains_parameters(self, fitted_mcrals):
        text = repr(fitted_mcrals.result)
        assert "parameters:" in text
        assert "n_components" in text
        assert "max_iter" in text

    def test_raises_before_fit(self):
        assert_result_raises_before_fit(MCRALS(), NotFittedError)

    def test_fit_still_returns_self(self, mcrals_dataset):
        X, C0 = mcrals_dataset
        mcr = MCRALS()
        assert_fit_returns_self(mcr, X, C0)

    def test_existing_properties_unchanged(self, fitted_mcrals):
        assert fitted_mcrals.components.shape == (2, 6)
        c_result = fitted_mcrals.C
        assert c_result.shape == (10, 2)
        assert fitted_mcrals.n_components == 2


# ======================================================================================
# Helpers
# ======================================================================================
def _synthetic_data(rng, n_rows=10, n_cols=6):
    """Create synthetic non-negative dataset and initial guess C0."""
    C_true = rng.rand(n_rows, 2)
    St_true = rng.rand(2, n_cols)
    X = C_true @ St_true + 0.01 * rng.randn(n_rows, n_cols)
    C0 = np.abs(C_true + 0.1 * rng.randn(n_rows, 2))
    return X, C0
