# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
"""
Tests for the FastICA result object.
"""

import numpy as np
import pytest

from spectrochempy.analysis._base._analysisbase import NotFittedError
from spectrochempy.analysis._base._result import AnalysisResult
from spectrochempy.analysis._base._result import ResultBase
from spectrochempy.analysis.decomposition.fast_ica import FastICA


# ======================================================================================
# FastICA result integration tests
# ======================================================================================
class TestFastICAResult:
    def test_result_is_analysis_result(self, fastica_dataset):
        ica = FastICA(n_components=4, random_state=123, whiten="unit-variance")
        ica.fit(fastica_dataset)
        result = ica.result
        assert isinstance(result, AnalysisResult)
        assert isinstance(result, ResultBase)

    def test_estimator_name(self, fastica_dataset):
        ica = FastICA(n_components=4, random_state=123, whiten="unit-variance")
        ica.fit(fastica_dataset)
        assert ica.result.estimator == "FastICA"

    def test_outputs_contain_keys(self, fastica_dataset):
        ica = FastICA(n_components=4, random_state=123, whiten="unit-variance")
        ica.fit(fastica_dataset)
        result = ica.result
        for name in ("components", "mixing", "St", "A"):
            assert name in result.outputs, f"{name} missing from result.outputs"

    def test_output_values_match_properties(self, fastica_dataset):
        ica = FastICA(n_components=4, random_state=123, whiten="unit-variance")
        ica.fit(fastica_dataset)
        result = ica.result
        np.testing.assert_array_equal(
            result.outputs["components"].data,
            ica.components.data,
        )
        np.testing.assert_array_equal(
            result.outputs["mixing"].data,
            ica.mixing.data,
        )
        np.testing.assert_array_equal(
            result.outputs["St"].data,
            ica.St.data,
        )
        np.testing.assert_array_equal(
            result.outputs["A"].data,
            ica.A.data,
        )

    def test_diagnostics_contain_keys(self, fastica_dataset):
        ica = FastICA(n_components=4, random_state=123, whiten="unit-variance")
        ica.fit(fastica_dataset)
        result = ica.result
        for name in ("n_iter",):
            assert name in result.diagnostics, f"{name} missing from result.diagnostics"

    def test_diagnostic_values(self, fastica_dataset):
        ica = FastICA(n_components=4, random_state=123, whiten="unit-variance")
        ica.fit(fastica_dataset)
        result = ica.result
        assert result.diagnostics["n_iter"] == ica.n_iter
        assert isinstance(result.diagnostics["n_iter"], int)
        assert result.diagnostics["n_iter"] > 0

    def test_repr_contains_expected_fields(self, fastica_dataset):
        ica = FastICA(n_components=4, random_state=123, whiten="unit-variance")
        ica.fit(fastica_dataset)
        text = repr(ica.result)
        assert "AnalysisResult" in text
        assert "FastICA" in text
        assert "components" in text
        assert "mixing" in text
        assert "n_iter" in text

    def test_result_is_not_cached(self, fastica_dataset):
        ica = FastICA(n_components=4, random_state=123, whiten="unit-variance")
        ica.fit(fastica_dataset)
        assert ica.result is not ica.result, (
            "AnalysisResult is recreated on every access; "
            "change this assertion if caching is added later"
        )

    def test_parameters_contain_expected_keys(self, fastica_dataset):
        ica = FastICA(n_components=4, random_state=123, whiten="unit-variance")
        ica.fit(fastica_dataset)
        params = ica.result.parameters
        for name in (
            "n_components",
            "algorithm",
            "whiten",
            "fun",
            "fun_args",
            "tol",
            "max_iter",
            "w_init",
            "whiten_solver",
            "random_state",
        ):
            assert name in params, f"{name} missing from result.parameters"

    def test_parameters_values(self, fastica_dataset):
        ica = FastICA(n_components=4, random_state=123, whiten="unit-variance")
        ica.fit(fastica_dataset)
        params = ica.result.parameters
        assert params["n_components"] == 4
        assert params["algorithm"] == "parallel"
        assert params["whiten"] == "unit-variance"
        assert params["fun"] == "logcosh"
        assert params["fun_args"] is None
        assert params["tol"] == 1e-4
        assert params["max_iter"] == 200
        assert params["w_init"] is None
        assert params["whiten_solver"] == "svd"
        assert params["random_state"] == 123

    def test_parameters_match_estimator_config(self, fastica_dataset):
        ica = FastICA(
            n_components=3,
            algorithm="deflation",
            tol=1e-6,
            max_iter=500,
            random_state=42,
        )
        ica.fit(fastica_dataset)
        params = ica.result.parameters
        assert params["n_components"] == 3
        assert params["algorithm"] == "deflation"
        assert params["tol"] == 1e-6
        assert params["max_iter"] == 500
        assert params["random_state"] == 42

    def test_repr_contains_parameters(self, fastica_dataset):
        ica = FastICA(n_components=4, random_state=123, whiten="unit-variance")
        ica.fit(fastica_dataset)
        text = repr(ica.result)
        assert "parameters:" in text
        assert "n_components" in text
        assert "algorithm" in text
        assert "whiten" in text

    def test_raises_before_fit(self):
        ica = FastICA()
        with pytest.raises(NotFittedError):
            _ = ica.result

    def test_fit_still_returns_self(self, fastica_dataset):
        ica = FastICA(n_components=4, random_state=123, whiten="unit-variance")
        ret = ica.fit(fastica_dataset)
        assert ret is ica

    def test_existing_properties_unchanged(self, fastica_dataset):
        ica = FastICA(n_components=4, random_state=123, whiten="unit-variance")
        ica.fit(fastica_dataset)
        assert ica.components.shape == (4, 8)
        assert ica.mixing.shape == (8, 4)
        assert ica.St.shape == (4, 8)
        assert ica.n_iter > 0
        assert np.all(np.isfinite(ica.components.data))
        assert np.all(np.isfinite(ica.mixing.data))

    # whitening when whiten=False covered in test_fast_ica.py
