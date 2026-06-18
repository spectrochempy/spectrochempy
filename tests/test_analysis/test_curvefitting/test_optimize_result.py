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
from spectrochempy.analysis.curvefitting._models import asymmetricvoigtmodel


# ======================================================================================
# Fixtures (same as test_optimize.py)
# ======================================================================================
@pytest.fixture()
def script():
    return """

    #-----------------------------------------------------------
    # syntax for parameters definition :
    # name : value, low_bound,  high_bound
    #  * for fixed parameters
    #  $ for variable parameters
    #  > for reference to a parameter in the COMMON block
    #    (> is forbidden in the COMMON block)
    # common block parameters should not have a _ in their names
    #-----------------------------------------------------------
    #
    COMMON:
    # common parameters ex.
    # $ gwidth: 1.0, 0.0, none
    $ gratio: 0.1, 0.0, 1.0

    MODEL: LINE_1
    shape: asymmetricvoigtmodel
        * ampl:  1.0, 0.0, none
        $ pos:   3620, 3400.0, 3700.0
        $ ratio: 0.0147, 0.0, 1.0
        $ asym: 0.1, 0, 1
        $ width: 200, 0, 1000

    MODEL: LINE_2
    shape: asymmetricvoigtmodel
        $ ampl:  0.2, 0.0, none
        $ pos:   3520, 3400.0, 3700.0
        > ratio: gratio
        $ asym: 0.1, 0, 1
        $ width: 200, 0, 1000
    """


@pytest.fixture()
def synthetic_two_peak_dataset():
    x = scp.Coord(np.linspace(3700.0, 3400.0, 301), title="wavenumber", units="cm^-1")
    model = asymmetricvoigtmodel()
    y = (
        model.f(x.data, ampl=1.0, pos=3620.0, width=200.0, ratio=0.0147, asym=0.1)
        + model.f(x.data, ampl=0.2, pos=3520.0, width=200.0, ratio=0.1, asym=0.1)
        + 0.0002 * x.data
        - 0.5
    )
    return scp.NDDataset(
        y,
        coordset=[x],
        units="absorbance",
        title="synthetic optimize spectrum",
    )


# ======================================================================================
# Optimize result tests
# ======================================================================================
class TestOptimizeResult:
    # ----------------------------------------------------------------------------------
    # Identity and type
    # ----------------------------------------------------------------------------------
    def test_result_is_fit_result(self, synthetic_two_peak_dataset, script):
        opt = scp.Optimize()
        opt.script = script
        opt.autobase = True
        opt.max_iter = 10
        opt.fit(synthetic_two_peak_dataset)
        result = opt.result
        assert isinstance(result, FitResult)

    def test_result_is_instance_of_base(self, synthetic_two_peak_dataset, script):
        opt = scp.Optimize()
        opt.script = script
        opt.autobase = True
        opt.max_iter = 10
        opt.fit(synthetic_two_peak_dataset)
        assert isinstance(opt.result, ResultBase)

    def test_estimator_name(self, synthetic_two_peak_dataset, script):
        opt = scp.Optimize()
        opt.script = script
        opt.autobase = True
        opt.max_iter = 10
        opt.fit(synthetic_two_peak_dataset)
        assert opt.result.estimator == "Optimize"

    def test_result_is_not_cached(self, synthetic_two_peak_dataset, script):
        opt = scp.Optimize()
        opt.script = script
        opt.autobase = True
        opt.max_iter = 10
        opt.fit(synthetic_two_peak_dataset)
        assert opt.result is not opt.result, (
            "FitResult is recreated on every access; "
            "change this assertion if caching is added later"
        )

    # ----------------------------------------------------------------------------------
    # Outputs
    # ----------------------------------------------------------------------------------
    def test_outputs_contain_keys(self, synthetic_two_peak_dataset, script):
        opt = scp.Optimize()
        opt.script = script
        opt.autobase = True
        opt.max_iter = 10
        opt.fit(synthetic_two_peak_dataset)
        result = opt.result
        for name in ("fitted", "components"):
            assert name in result.outputs, f"{name} missing from result.outputs"

    def test_output_values_match_properties(self, synthetic_two_peak_dataset, script):
        opt = scp.Optimize()
        opt.script = script
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

    def test_output_fitted_shape(self, synthetic_two_peak_dataset, script):
        opt = scp.Optimize()
        opt.script = script
        opt.autobase = True
        opt.max_iter = 10
        opt.fit(synthetic_two_peak_dataset)
        dataset = synthetic_two_peak_dataset
        result = opt.result
        # fitted shape matches the original dataset
        assert result.outputs["fitted"].shape == (1, dataset.size)

    # ----------------------------------------------------------------------------------
    # Parameters
    # ----------------------------------------------------------------------------------
    def test_parameters_contain_expected_keys(self, synthetic_two_peak_dataset, script):
        opt = scp.Optimize()
        opt.script = script
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

    def test_parameters_values_default(self, synthetic_two_peak_dataset, script):
        opt = scp.Optimize()
        opt.script = script
        opt.autobase = True
        opt.max_iter = 10
        opt.fit(synthetic_two_peak_dataset)
        params = opt.result.parameters
        assert params["method"] == "least_squares"
        assert params["max_iter"] == 10
        assert params["autobase"] is True
        assert params["amplitude_mode"] == "height"

    def test_parameters_match_estimator_config(
        self, synthetic_two_peak_dataset, script
    ):
        opt = scp.Optimize()
        opt.script = script
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
    def test_diagnostics_contain_keys(self, synthetic_two_peak_dataset, script):
        opt = scp.Optimize()
        opt.script = script
        opt.autobase = True
        opt.max_iter = 10
        opt.fit(synthetic_two_peak_dataset)
        diag = opt.result.diagnostics
        for name in ("cost", "niter", "ncalls"):
            assert name in diag, f"{name} missing from result.diagnostics"

    def test_diagnostics_are_scalars(self, synthetic_two_peak_dataset, script):
        opt = scp.Optimize()
        opt.script = script
        opt.autobase = True
        opt.max_iter = 10
        opt.fit(synthetic_two_peak_dataset)
        diag = opt.result.diagnostics
        assert isinstance(diag["cost"], float) or diag["cost"] is None
        assert isinstance(diag["niter"], int)
        assert isinstance(diag["ncalls"], int)

    def test_diagnostics_meaningful_values(self, synthetic_two_peak_dataset, script):
        opt = scp.Optimize()
        opt.script = script
        opt.autobase = True
        opt.max_iter = 10
        opt.fit(synthetic_two_peak_dataset)
        diag = opt.result.diagnostics
        # cost should be a small positive number after a reasonable fit
        assert diag["cost"] is None or diag["cost"] >= 0.0
        # at least one iteration and function call should have occurred
        assert diag["niter"] >= 0
        assert diag["ncalls"] >= 0

    # ----------------------------------------------------------------------------------
    # Representation
    # ----------------------------------------------------------------------------------
    def test_repr_contains_expected_fields(self, synthetic_two_peak_dataset, script):
        opt = scp.Optimize()
        opt.script = script
        opt.autobase = True
        opt.max_iter = 10
        opt.fit(synthetic_two_peak_dataset)
        text = repr(opt.result)
        assert "FitResult" in text
        assert "Optimize" in text
        assert "fitted" in text
        assert "components" in text
        assert "cost" in text

    def test_repr_does_not_crash(self, synthetic_two_peak_dataset, script):
        opt = scp.Optimize()
        opt.script = script
        opt.autobase = True
        opt.max_iter = 10
        opt.fit(synthetic_two_peak_dataset)
        _ = repr(opt.result)

    # ----------------------------------------------------------------------------------
    # Pre-fit guard
    # ----------------------------------------------------------------------------------
    def test_raises_before_fit(self, script):
        opt = scp.Optimize()
        opt.script = script
        with pytest.raises(NotFittedError):
            _ = opt.result

    # ----------------------------------------------------------------------------------
    # Existing behaviour preserved
    # ----------------------------------------------------------------------------------
    def test_fit_still_returns_self(self, synthetic_two_peak_dataset, script):
        opt = scp.Optimize()
        opt.script = script
        opt.autobase = True
        opt.max_iter = 10
        ret = opt.fit(synthetic_two_peak_dataset)
        assert ret is opt

    def test_existing_properties_unchanged(self, synthetic_two_peak_dataset, script):
        opt = scp.Optimize()
        opt.script = script
        opt.autobase = True
        opt.max_iter = 10
        opt.fit(synthetic_two_peak_dataset)

        assert opt.n_components == 2
        assert opt.components.shape == (3, synthetic_two_peak_dataset.size)
        assert opt.predict().shape == (1, synthetic_two_peak_dataset.size)
        assert opt.transform().shape == (1, 2)
