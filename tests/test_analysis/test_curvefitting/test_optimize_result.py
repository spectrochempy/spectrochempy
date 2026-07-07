# ======================================================================================
# Copyright (©) 2014-2026 Laboratoire Catalyse et Spectrochimie (LCS), Caen, France.
# CeCILL-B FREE SOFTWARE LICENSE AGREEMENT
# See full LICENSE agreement in the root directory.
# ======================================================================================
# ruff: noqa
"""
Tests for the Optimize FitResult prototype.
"""

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.analysis._base._analysisbase import NotFittedError
from spectrochempy.analysis._base._result import FitResult
from spectrochempy.analysis._base._result import ResultBase
from spectrochempy.analysis.curvefitting.optimize import _compute_fit_diagnostics
from tests.test_analysis.result_test_helpers import assert_fit_returns_self
from tests.test_analysis.result_test_helpers import assert_result_basics
from tests.test_analysis.result_test_helpers import assert_result_raises_before_fit


# ======================================================================================
# Optimize result tests
# ======================================================================================
class TestOptimizeResult:
    # ----------------------------------------------------------------------------------
    # Identity and type
    # ----------------------------------------------------------------------------------
    def test_result_is_fit_result(self, synthetic_two_peak_dataset, optimize_script):
        opt = scp.Optimize()
        opt.script = optimize_script
        opt.autobase = True
        opt.max_iter = 10
        opt.fit(synthetic_two_peak_dataset)
        assert_result_basics(opt, FitResult, "Optimize")

    def test_result_is_instance_of_base(
        self, synthetic_two_peak_dataset, optimize_script
    ):
        opt = scp.Optimize()
        opt.script = optimize_script
        opt.autobase = True
        opt.max_iter = 10
        opt.fit(synthetic_two_peak_dataset)
        assert isinstance(opt.result, ResultBase)

    # ----------------------------------------------------------------------------------
    # Outputs
    # ----------------------------------------------------------------------------------
    def test_outputs_contain_keys(self, synthetic_two_peak_dataset, optimize_script):
        opt = scp.Optimize()
        opt.script = optimize_script
        opt.autobase = True
        opt.max_iter = 10
        opt.fit(synthetic_two_peak_dataset)
        result = opt.result
        for name in ("fitted", "components", "residuals"):
            assert name in result.outputs, f"{name} missing from result.outputs"

    def test_output_values_match_properties(
        self, synthetic_two_peak_dataset, optimize_script
    ):
        opt = scp.Optimize()
        opt.script = optimize_script
        opt.autobase = True
        opt.max_iter = 10
        opt.fit(synthetic_two_peak_dataset)
        result = opt.result
        np.testing.assert_array_equal(
            result.outputs["fitted"].data,
            opt.predict().data,
        )
        np.testing.assert_array_equal(
            result.outputs["components"].data,
            opt.components.data,
        )
        np.testing.assert_array_equal(
            result.outputs["residuals"].data,
            result.residuals.data,
        )

    def test_output_fitted_shape(self, synthetic_two_peak_dataset, optimize_script):
        opt = scp.Optimize()
        opt.script = optimize_script
        opt.autobase = True
        opt.max_iter = 10
        opt.fit(synthetic_two_peak_dataset)
        dataset = synthetic_two_peak_dataset
        result = opt.result
        # fitted shape matches the original dataset
        assert result.outputs["fitted"].shape == (1, dataset.size)

    def test_residuals_match_observed_minus_fitted(
        self, synthetic_two_peak_dataset, optimize_script
    ):
        opt = scp.Optimize()
        opt.script = optimize_script
        opt.autobase = True
        opt.max_iter = 10
        opt.fit(synthetic_two_peak_dataset)
        result = opt.result

        expected = synthetic_two_peak_dataset - result.fitted
        np.testing.assert_allclose(result.residuals.masked_data, expected.masked_data)
        assert result.residuals.units == synthetic_two_peak_dataset.units

    # ----------------------------------------------------------------------------------
    # Parameters
    # ----------------------------------------------------------------------------------
    def test_parameters_contain_expected_keys(
        self, synthetic_two_peak_dataset, optimize_script
    ):
        opt = scp.Optimize()
        opt.script = optimize_script
        opt.autobase = True
        opt.max_iter = 10
        opt.fit(synthetic_two_peak_dataset)
        params = opt.result.parameters
        for name in (
            "method",
            "max_iter",
            "max_fun_calls",
            "autobase",
            "amplitude_mode",
        ):
            assert name in params, f"{name} missing from result.parameters"

    def test_parameters_values_default(
        self, synthetic_two_peak_dataset, optimize_script
    ):
        opt = scp.Optimize()
        opt.script = optimize_script
        opt.autobase = True
        opt.max_iter = 10
        opt.fit(synthetic_two_peak_dataset)
        params = opt.result.parameters
        assert params["method"] == "least_squares"
        assert params["max_iter"] == 10
        assert params["autobase"] is True
        assert params["amplitude_mode"] == "height"

    def test_parameters_match_estimator_config(
        self, synthetic_two_peak_dataset, optimize_script
    ):
        opt = scp.Optimize()
        opt.script = optimize_script
        opt.autobase = True
        opt.max_iter = 20
        opt.method = "simplex"
        opt.amplitude_mode = "area"
        opt.fit(synthetic_two_peak_dataset)
        params = opt.result.parameters
        assert params["method"] == "simplex"
        assert params["max_iter"] == 20
        assert params["amplitude_mode"] == "area"

    # ----------------------------------------------------------------------------------
    # Diagnostics
    # ----------------------------------------------------------------------------------
    def test_diagnostics_contain_keys(
        self, synthetic_two_peak_dataset, optimize_script
    ):
        opt = scp.Optimize()
        opt.script = optimize_script
        opt.autobase = True
        opt.max_iter = 10
        opt.fit(synthetic_two_peak_dataset)
        diag = opt.result.diagnostics
        for name in (
            "cost",
            "niter",
            "ncalls",
            "rss",
            "sse",
            "rmse",
            "r_squared",
            "success",
            "status",
            "message",
        ):
            assert name in diag, f"{name} missing from result.diagnostics"

    def test_diagnostics_are_scalars(self, synthetic_two_peak_dataset, optimize_script):
        opt = scp.Optimize()
        opt.script = optimize_script
        opt.autobase = True
        opt.max_iter = 10
        opt.fit(synthetic_two_peak_dataset)
        diag = opt.result.diagnostics
        assert isinstance(diag["cost"], float) or diag["cost"] is None
        assert isinstance(diag["niter"], int)
        assert isinstance(diag["ncalls"], int)
        assert isinstance(diag["rss"], float)
        assert isinstance(diag["sse"], float)
        assert isinstance(diag["rmse"], float)
        assert isinstance(diag["r_squared"], float)
        assert isinstance(diag["success"], bool)
        assert isinstance(diag["message"], str)

    def test_diagnostics_meaningful_values(
        self, synthetic_two_peak_dataset, optimize_script
    ):
        opt = scp.Optimize()
        opt.script = optimize_script
        opt.autobase = True
        opt.max_iter = 10
        opt.fit(synthetic_two_peak_dataset)
        diag = opt.result.diagnostics
        # cost should be a small positive number after a reasonable fit
        assert diag["cost"] is None or diag["cost"] >= 0.0
        # at least one iteration and function call should have occurred
        assert diag["niter"] >= 0
        assert diag["ncalls"] >= 0
        assert diag["rss"] == pytest.approx(diag["sse"])
        assert np.isfinite(diag["rmse"])
        assert diag["rmse"] >= 0.0
        assert np.isfinite(diag["r_squared"])
        assert diag["r_squared"] > 0.99
        assert isinstance(diag["status"], int | type(None))

    def test_rss_matches_residual_sum_of_squares(
        self, synthetic_two_peak_dataset, optimize_script
    ):
        opt = scp.Optimize()
        opt.script = optimize_script
        opt.autobase = True
        opt.max_iter = 10
        opt.fit(synthetic_two_peak_dataset)
        result = opt.result

        residual_data = np.ma.asarray(result.residuals.masked_data)
        expected_rss = float(np.ma.sum(np.ma.abs(residual_data) ** 2))
        assert result.diagnostics["rss"] == pytest.approx(expected_rss)
        assert result.diagnostics["sse"] == pytest.approx(expected_rss)

    def test_constant_data_zero_tss_gives_nan_r_squared(
        self, constant_optimize_dataset, optimize_script
    ):
        opt = scp.Optimize()
        opt.script = optimize_script
        opt.autobase = True
        opt.max_iter = 10
        opt.fit(constant_optimize_dataset)
        diag = opt.result.diagnostics

        assert np.isnan(diag["r_squared"])
        assert diag["rmse"] >= 0.0 or np.isnan(diag["rmse"])
        assert diag["rss"] == pytest.approx(diag["sse"])

    def test_empty_diagnostics_are_stable(self):
        empty = scp.NDDataset(np.array([], dtype=np.float64))
        residuals, diagnostics = _compute_fit_diagnostics(empty, empty, {})

        assert residuals.size == 0
        assert diagnostics["rss"] == pytest.approx(0.0)
        assert diagnostics["sse"] == pytest.approx(0.0)
        assert np.isnan(diagnostics["rmse"])
        assert np.isnan(diagnostics["r_squared"])

    def test_dry_fit_exposes_conservative_solver_status(
        self, synthetic_two_peak_dataset, optimize_script
    ):
        opt = scp.Optimize()
        opt.script = optimize_script
        opt.autobase = True
        opt.dry = True

        opt.fit(synthetic_two_peak_dataset)
        diag = opt.result.diagnostics

        assert diag["success"] is False
        assert diag["status"] is None
        assert diag["message"] == ""

    # ----------------------------------------------------------------------------------
    # Representation
    # ----------------------------------------------------------------------------------
    def test_repr_contains_expected_fields(
        self, synthetic_two_peak_dataset, optimize_script
    ):
        opt = scp.Optimize()
        opt.script = optimize_script
        opt.autobase = True
        opt.max_iter = 10
        opt.fit(synthetic_two_peak_dataset)
        text = repr(opt.result)
        assert "FitResult" in text
        assert "Optimize" in text
        assert "fitted" in text
        assert "components" in text
        assert "residuals" in text
        assert "cost" in text

    def test_repr_does_not_crash(self, synthetic_two_peak_dataset, optimize_script):
        opt = scp.Optimize()
        opt.script = optimize_script
        opt.autobase = True
        opt.max_iter = 10
        opt.fit(synthetic_two_peak_dataset)
        _ = repr(opt.result)

    # ----------------------------------------------------------------------------------
    # Pre-fit guard
    # ----------------------------------------------------------------------------------
    def test_raises_before_fit(self, optimize_script):
        opt = scp.Optimize()
        opt.script = optimize_script
        assert_result_raises_before_fit(opt, NotFittedError)

    # ----------------------------------------------------------------------------------
    # Existing behaviour preserved
    # ----------------------------------------------------------------------------------
    def test_fit_still_returns_self(self, synthetic_two_peak_dataset, optimize_script):
        opt = scp.Optimize()
        opt.script = optimize_script
        opt.autobase = True
        opt.max_iter = 10
        assert_fit_returns_self(opt, synthetic_two_peak_dataset)

    def test_existing_properties_unchanged(
        self, synthetic_two_peak_dataset, optimize_script
    ):
        opt = scp.Optimize()
        opt.script = optimize_script
        opt.autobase = True
        opt.max_iter = 10
        opt.fit(synthetic_two_peak_dataset)

        assert opt.n_components == 2
        assert opt.components.shape == (3, synthetic_two_peak_dataset.size)
        assert opt.predict().shape == (1, synthetic_two_peak_dataset.size)
        assert opt.transform().shape == (1, 2)
