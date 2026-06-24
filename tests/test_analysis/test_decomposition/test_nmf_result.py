# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
"""
Tests for the NMF result object.
"""

import numpy as np
import pytest

from spectrochempy.analysis._base._analysisbase import NotFittedError
from spectrochempy.analysis._base._result import AnalysisResult
from spectrochempy.analysis._base._result import ResultBase
from spectrochempy.analysis.decomposition.nmf import NMF
from tests.test_analysis.result_test_helpers import assert_fit_returns_self
from tests.test_analysis.result_test_helpers import assert_result_basics
from tests.test_analysis.result_test_helpers import assert_result_raises_before_fit


# ======================================================================================
# Fixtures
# ======================================================================================
@pytest.fixture()
def nmf_dataset():
    """Small synthetic non-negative dataset for NMF testing."""
    rng = np.random.RandomState(42)
    W_true = rng.rand(10, 3)
    H_true = rng.rand(3, 6)
    data = W_true @ H_true
    return data


@pytest.fixture()
def fitted_nmf(nmf_dataset):
    nmf = NMF(n_components=3, init="random", max_iter=500, random_state=42, tol=1e-8)
    nmf.fit(nmf_dataset)
    return nmf


# ======================================================================================
# NMF result integration tests
# ======================================================================================
class TestNMFResult:
    def test_result_is_analysis_result(self, fitted_nmf):
        result = assert_result_basics(fitted_nmf, AnalysisResult, "NMF")
        assert isinstance(result, ResultBase)

    def test_outputs_contain_keys(self, fitted_nmf):
        result = fitted_nmf.result
        for name in ("components", "W"):
            assert name in result.outputs, f"{name} missing from result.outputs"

    def test_diagnostics_contain_keys(self, fitted_nmf):
        result = fitted_nmf.result
        for name in ("reconstruction_error", "n_iter"):
            assert name in result.diagnostics, f"{name} missing from result.diagnostics"

    def test_output_values_match_properties(self, fitted_nmf):
        result = fitted_nmf.result
        np.testing.assert_array_equal(
            result.outputs["components"].data,
            fitted_nmf.components.data,
        )
        np.testing.assert_array_equal(
            result.outputs["W"].data,
            fitted_nmf.transform().data,
        )

    def test_diagnostic_values(self, fitted_nmf):
        result = fitted_nmf.result
        assert (
            result.diagnostics["reconstruction_error"]
            == fitted_nmf._nmf.reconstruction_err_
        )
        assert result.diagnostics["n_iter"] == fitted_nmf._nmf.n_iter_

    def test_repr_contains_expected_fields(self, fitted_nmf):
        text = repr(fitted_nmf.result)
        assert "AnalysisResult" in text
        assert "NMF" in text
        assert "components" in text
        assert "W" in text
        assert "reconstruction_error" in text
        assert "n_iter" in text

    def test_parameters_contain_expected_keys(self, fitted_nmf):
        params = fitted_nmf.result.parameters
        for name in (
            "n_components",
            "init",
            "solver",
            "beta_loss",
            "tol",
            "max_iter",
            "random_state",
            "alpha_W",
            "alpha_H",
            "l1_ratio",
            "shuffle",
        ):
            assert name in params, f"{name} missing from result.parameters"

    def test_parameters_values(self, fitted_nmf):
        params = fitted_nmf.result.parameters
        assert params["n_components"] == 3
        assert params["init"] == "random"
        assert params["solver"] == "cd"
        assert params["beta_loss"] == "frobenius"
        assert params["tol"] == 1e-8
        assert params["max_iter"] == 500
        assert params["random_state"] == 42
        assert params["alpha_W"] == 0.0
        assert params["alpha_H"] == "same"
        assert params["l1_ratio"] == 0.0
        assert params["shuffle"] is False

    def test_parameters_match_estimator_config(self):
        nmf = NMF(n_components=2, l1_ratio=0.5, tol=1e-6, max_iter=300)
        data = np.random.RandomState(0).rand(10, 6)
        nmf.fit(data)
        params = nmf.result.parameters
        assert params["n_components"] == 2
        assert params["l1_ratio"] == 0.5
        assert params["tol"] == 1e-6
        assert params["max_iter"] == 300

    def test_repr_contains_parameters(self, fitted_nmf):
        text = repr(fitted_nmf.result)
        assert "parameters:" in text
        assert "n_components" in text
        assert "solver" in text

    def test_raises_before_fit(self):
        assert_result_raises_before_fit(NMF(), NotFittedError)

    def test_fit_still_returns_self(self, nmf_dataset):
        nmf = NMF(
            n_components=3, init="random", max_iter=500, random_state=42, tol=1e-8
        )
        assert_fit_returns_self(nmf, nmf_dataset)

    def test_existing_properties_unchanged(self, fitted_nmf):
        assert fitted_nmf.components.shape == (3, 6)
        assert fitted_nmf.transform().shape == (10, 3)
        assert fitted_nmf.n_components == 3
