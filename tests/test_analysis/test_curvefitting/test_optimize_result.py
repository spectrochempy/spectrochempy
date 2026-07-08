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
from spectrochempy.analysis.curvefitting._parameters import FitParameters
from spectrochempy.analysis.curvefitting import optimize as optimize_module
from spectrochempy.analysis.curvefitting.optimize import _compute_covariance_matrix
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
            "n_observations",
            "n_varying_parameters",
            "degrees_of_freedom",
            "rss",
            "sse",
            "rmse",
            "r_squared",
            "reduced_chi_square",
            "adjusted_r_squared",
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
        assert isinstance(diag["n_observations"], int)
        assert isinstance(diag["n_varying_parameters"], int)
        assert isinstance(diag["degrees_of_freedom"], int)
        assert isinstance(diag["rss"], float)
        assert isinstance(diag["sse"], float)
        assert isinstance(diag["rmse"], float)
        assert isinstance(diag["r_squared"], float)
        assert isinstance(diag["reduced_chi_square"], float)
        assert isinstance(diag["adjusted_r_squared"], float)
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
        assert diag["n_observations"] == synthetic_two_peak_dataset.size
        assert diag["n_varying_parameters"] == 9
        assert diag["degrees_of_freedom"] == (
            diag["n_observations"] - diag["n_varying_parameters"]
        )
        assert diag["rss"] == pytest.approx(diag["sse"])
        assert np.isfinite(diag["rmse"])
        assert diag["rmse"] >= 0.0
        assert np.isfinite(diag["r_squared"])
        assert diag["r_squared"] > 0.99
        assert np.isfinite(diag["reduced_chi_square"])
        assert diag["reduced_chi_square"] == pytest.approx(
            diag["rss"] / diag["degrees_of_freedom"]
        )
        assert np.isfinite(diag["adjusted_r_squared"])
        assert diag["adjusted_r_squared"] <= diag["r_squared"]
        assert isinstance(diag["status"], int | type(None))
        assert opt.result.covariance is not None

    def test_covariance_available_for_least_squares_fit(
        self, synthetic_two_peak_dataset, optimize_script
    ):
        opt = scp.Optimize()
        opt.script = optimize_script
        opt.autobase = True
        opt.max_iter = 10
        opt.fit(synthetic_two_peak_dataset)

        covariance = opt.result.covariance

        assert covariance is not None
        assert covariance.flags.writeable is False
        assert covariance.shape == (
            opt.result.diagnostics["n_varying_parameters"],
            opt.result.diagnostics["n_varying_parameters"],
        )
        np.testing.assert_allclose(covariance, covariance.T)
        assert np.all(np.isfinite(np.diag(covariance)))
        assert np.all(np.diag(covariance) >= 0.0)
        np.testing.assert_allclose(opt.result.variance, np.diag(covariance))
        np.testing.assert_allclose(opt.result.stderr**2, opt.result.variance)
        assert np.all(np.isfinite(opt.result.stderr))
        assert np.all(opt.result.stderr >= 0.0)

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
        assert np.isnan(diag["adjusted_r_squared"])
        assert diag["rmse"] >= 0.0 or np.isnan(diag["rmse"])
        assert diag["rss"] == pytest.approx(diag["sse"])
        assert diag["degrees_of_freedom"] == (
            diag["n_observations"] - diag["n_varying_parameters"]
        )

    def test_fixed_parameters_are_not_counted_as_varying(
        self, synthetic_two_peak_dataset, optimize_script
    ):
        opt = scp.Optimize()
        opt.script = optimize_script
        opt.autobase = True
        opt.max_iter = 10
        opt.fit(synthetic_two_peak_dataset)

        assert opt.result.diagnostics["n_varying_parameters"] == 9

    def test_empty_diagnostics_are_stable(self):
        empty = scp.NDDataset(np.array([], dtype=np.float64))
        residuals, diagnostics = _compute_fit_diagnostics(empty, empty, {}, None)

        assert residuals.size == 0
        assert diagnostics["n_observations"] == 0
        assert diagnostics["n_varying_parameters"] == 0
        assert diagnostics["degrees_of_freedom"] == 0
        assert diagnostics["rss"] == pytest.approx(0.0)
        assert diagnostics["sse"] == pytest.approx(0.0)
        assert np.isnan(diagnostics["rmse"])
        assert np.isnan(diagnostics["r_squared"])
        assert np.isnan(diagnostics["reduced_chi_square"])
        assert np.isnan(diagnostics["adjusted_r_squared"])

    def test_non_positive_degrees_of_freedom_are_stable(self):
        observed = scp.NDDataset(np.array([1.0, 2.0, 3.0], dtype=np.float64))
        fitted = observed.copy()
        fit_parameters = FitParameters()
        fit_parameters["a"] = (1.0, None, None, False)
        fit_parameters["b"] = (1.0, None, None, False)
        fit_parameters["c"] = (1.0, None, None, False)

        _, diagnostics = _compute_fit_diagnostics(observed, fitted, {}, fit_parameters)

        assert diagnostics["n_observations"] == 3
        assert diagnostics["n_varying_parameters"] == 3
        assert diagnostics["degrees_of_freedom"] == 0
        assert np.isnan(diagnostics["reduced_chi_square"])
        assert np.isnan(diagnostics["adjusted_r_squared"])

        jacobian = np.eye(3)
        covariance = _compute_covariance_matrix(observed, fitted, jacobian, diagnostics)
        assert covariance is None

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
        assert diag["n_varying_parameters"] == 9
        assert diag["degrees_of_freedom"] == (
            diag["n_observations"] - diag["n_varying_parameters"]
        )
        assert opt.result.covariance is None
        assert opt.result.variance is None
        assert opt.result.stderr is None

    # ----------------------------------------------------------------------------------
    # Solver artifacts
    # ----------------------------------------------------------------------------------
    def test_jacobian_raises_before_fit(self, optimize_script):
        opt = scp.Optimize()
        opt.script = optimize_script
        with pytest.raises(NotFittedError):
            _ = opt.jacobian

    def test_jacobian_available_for_least_squares_backend(
        self, synthetic_two_peak_dataset, optimize_script
    ):
        opt = scp.Optimize()
        opt.script = optimize_script
        opt.autobase = True
        opt.max_iter = 10
        opt.fit(synthetic_two_peak_dataset)

        jacobian = opt.jacobian

        assert jacobian is not None
        assert jacobian.flags.writeable is False
        assert jacobian.shape == (
            opt.result.diagnostics["n_observations"],
            opt.result.diagnostics["n_varying_parameters"],
        )

    def test_jacobian_available_for_leastsq_alias(
        self, synthetic_two_peak_dataset, optimize_script
    ):
        opt = scp.Optimize()
        opt.script = optimize_script
        opt.method = "leastsq"
        opt.autobase = True
        opt.max_iter = 10
        opt.fit(synthetic_two_peak_dataset)

        jacobian = opt.jacobian

        assert jacobian is not None
        assert jacobian.flags.writeable is False
        assert jacobian.shape == (
            opt.result.diagnostics["n_observations"],
            opt.result.diagnostics["n_varying_parameters"],
        )

    def test_jacobian_absent_for_simplex_backend(
        self, synthetic_two_peak_dataset, optimize_script
    ):
        opt = scp.Optimize()
        opt.script = optimize_script
        opt.method = "simplex"
        opt.autobase = True
        opt.max_iter = 10
        opt.fit(synthetic_two_peak_dataset)

        assert opt.jacobian is None
        assert opt.result.covariance is None
        assert opt.result.variance is None
        assert opt.result.stderr is None

    def test_jacobian_absent_for_basinhopping_backend(
        self, synthetic_two_peak_dataset, optimize_script, monkeypatch
    ):
        class _FakeLocalResult:
            success = True
            status = 0
            message = "ok"
            jac = np.ones((synthetic_two_peak_dataset.size, 9))

        class _FakeBasinhoppingResult:
            x = np.zeros(9, dtype=np.float64)
            fun = 0.0
            message = "ok"
            success = True
            status = 0
            lowest_optimization_result = _FakeLocalResult()

        monkeypatch.setattr(
            optimize_module.optimize,
            "basinhopping",
            lambda *args, **kwargs: _FakeBasinhoppingResult(),
        )

        opt = scp.Optimize()
        opt.script = optimize_script
        opt.method = "basinhopping"
        opt.autobase = True
        opt.max_iter = 10
        opt.fit(synthetic_two_peak_dataset)

        assert opt.jacobian is None
        assert opt.result.covariance is None
        assert opt.result.variance is None
        assert opt.result.stderr is None

    def test_jacobian_absent_for_dry_fit(
        self, synthetic_two_peak_dataset, optimize_script
    ):
        opt = scp.Optimize()
        opt.script = optimize_script
        opt.autobase = True
        opt.dry = True
        opt.fit(synthetic_two_peak_dataset)

        assert opt.jacobian is None
        assert opt.result.covariance is None
        assert opt.result.variance is None
        assert opt.result.stderr is None

    def test_fit_result_does_not_expose_jacobian(
        self, synthetic_two_peak_dataset, optimize_script
    ):
        opt = scp.Optimize()
        opt.script = optimize_script
        opt.autobase = True
        opt.max_iter = 10
        opt.fit(synthetic_two_peak_dataset)

        assert "jacobian" not in opt.result.outputs
        assert "jacobian" not in opt.result.diagnostics
        with pytest.raises(AttributeError):
            _ = opt.result.jacobian

    def test_covariance_available_for_leastsq_alias(
        self, synthetic_two_peak_dataset, optimize_script
    ):
        opt = scp.Optimize()
        opt.script = optimize_script
        opt.method = "leastsq"
        opt.autobase = True
        opt.max_iter = 10
        opt.fit(synthetic_two_peak_dataset)

        covariance = opt.result.covariance

        assert covariance is not None
        assert covariance.shape == (
            opt.result.diagnostics["n_varying_parameters"],
            opt.result.diagnostics["n_varying_parameters"],
        )
        assert opt.result.variance.shape == (
            opt.result.diagnostics["n_varying_parameters"],
        )
        assert opt.result.stderr.shape == (
            opt.result.diagnostics["n_varying_parameters"],
        )

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
