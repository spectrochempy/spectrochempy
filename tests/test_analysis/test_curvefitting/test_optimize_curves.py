"""Named recovery and decay preserve a native one-dimensional physical grid."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal
from pint.errors import DimensionalityError
from scipy.optimize import least_squares

import spectrochempy as scp
from spectrochempy.analysis._base._analysisbase import NotFittedError


@pytest.mark.parametrize("model", ["saturation_recovery", "exponential_decay"])
@pytest.mark.parametrize("dim", ["x", "y"])
@pytest.mark.parametrize("reverse", [False, True])
def test_named_curve_matches_callable_reference(model, dim, reverse):
    times = np.array([0, 0.01, 0.04, 0.15, 0.5, 1.2, 3, 7, 20.0])
    if reverse:
        times = times[::-1]

    def law(tau):
        if model == "saturation_recovery":
            return -np.expm1(-times / tau)
        return np.exp(-times / tau)

    data = 4 * law(2) + np.random.default_rng(493).normal(0, 0.002, times.size)
    observed = scp.NDDataset(
        data,
        dims=[dim],
        coordset=[scp.Coord(times, units="s", rounding=False)],
        units="V",
    )
    mask = np.zeros(observed.shape, dtype=bool)
    mask[2] = True
    observed.mask = mask
    observed.data[4] = np.nan
    valid = ~observed.mask & np.isfinite(observed.data)
    before = observed.copy()
    spec = {"amplitude": 3 * scp.ur.V, "tau": 1300 * scp.ur.ms}
    fit = scp.Optimize(model=model, parameters=spec).fit(observed)
    result = fit.result
    oracle = least_squares(
        lambda p: (data - p[0] * law(p[1]))[valid],
        [3, 1.3],
        bounds=([-np.inf, 1e-10], [np.inf, np.inf]),
    )
    assert_allclose(
        result.fitted_parameters["amplitude"].to("V").magnitude,
        oracle.x[0],
        rtol=1e-5,
    )
    assert_allclose(
        result.fitted_parameters["tau"].to("s").magnitude, oracle.x[1], rtol=1e-5
    )
    assert abs(result.fitted_parameters["tau"].to("s").magnitude - 2) < 0.01
    assert result.parameter_names == ("amplitude", "tau")
    assert result.parameter_units == (scp.ur.V, scp.ur.s)
    for output in (fit.predict(), result.fitted, result.residuals, *result.components):
        assert output.dims == [dim]
        assert output.shape == observed.shape
        assert output.units == observed.units
        assert_array_equal(output.coord(dim).data, times)
        assert_array_equal(output.mask, ~valid)
        assert output.history_entries[-1]["parameters"]["model"] == model
    assert_allclose(
        result.residuals.data[valid],
        (data - oracle.x[0] * law(oracle.x[1]))[valid],
        atol=1e-7,
    )
    assert_array_equal(observed.data, before.data)
    assert_array_equal(observed.mask, before.mask)
    assert spec["tau"] == 1300 * scp.ur.ms


def test_fixed_parameters_and_unit_conversion():
    times = np.array([0, 100, 700, 2000, 9000.0])
    curve = scp.NDDataset(
        2 * np.exp(-times / 700),
        coordset=[scp.Coord(times, units="ms")],
        units="mV",
    )
    fit = scp.Optimize(
        model="exponential_decay",
        parameters={
            "amplitude": {"value": 2 * scp.ur.mV, "fixed": True},
            "tau": {"value": 0.7 * scp.ur.s, "fixed": True},
        },
    ).fit(curve)
    assert fit.result.fitted_parameters["tau"] == 700 * scp.ur.ms
    assert fit.result.parameter_names == ()
    assert_allclose(fit.predict().data, curve.data)
    assert fit.result.diagnostics["covariance_reason"] == "no_free_parameters"


@pytest.mark.parametrize("tau", [0, -1, {"value": 1, "bounds": (0, 2)}, 2 * scp.ur.Hz])
def test_invalid_tau_is_rejected(tau):
    curve = scp.NDDataset([0, 1, 2], coordset=[scp.Coord([0, 1, 3], units="s")])
    with pytest.raises((ValueError, DimensionalityError)):
        scp.Optimize(
            model="saturation_recovery", parameters={"amplitude": 3, "tau": tau}
        ).fit(curve)


def test_failed_refit_invalidates_result_and_preserves_nd_boundary():
    curve = scp.NDDataset([0, 0.4, 0.9], coordset=[scp.Coord([0, 1, 3], units="s")])
    fit = scp.Optimize(
        model="saturation_recovery", parameters={"amplitude": 1, "tau": 1}
    ).fit(curve)
    with pytest.raises(ValueError, match="1D"):
        fit.fit(scp.zeros((3, 3)))
    with pytest.raises(NotFittedError):
        _ = fit.result
    with pytest.raises(ValueError, match="at least two"):
        scp.Optimize(model=lambda template: np.zeros(template.shape)).fit(curve)


@pytest.mark.parametrize("times", [[0, 1, 1], [0, np.nan, 3], [-1, 0, 3], [0, 3, 1]])
def test_invalid_delay_grid_is_rejected(times):
    curve = scp.NDDataset([0, 1, 2], coordset=[scp.Coord(times, units="s")])
    with pytest.raises(ValueError, match="coordinate"):
        scp.Optimize(
            model="saturation_recovery", parameters={"amplitude": 3, "tau": 1}
        ).fit(curve)


@pytest.mark.parametrize(
    "parameters", [{}, {"tau": 1}, {"amplitude": 2, "tau": 1, "offset": 0}]
)
def test_unknown_or_missing_parameters_are_rejected(parameters):
    curve = scp.NDDataset([0, 1, 2], coordset=[scp.Coord([0, 1, 3], units="s")])
    with pytest.raises(ValueError, match="amplitude and tau"):
        scp.Optimize(model="saturation_recovery", parameters=parameters).fit(curve)


def test_bound_and_partial_fixed_parameter_preserve_parameter_order():
    t = np.array([0, 0.1, 0.5, 1, 3, 10.0])
    curve = scp.NDDataset(2 * np.exp(-t / 3), coordset=[scp.Coord(t, units="s")])
    fit = scp.Optimize(
        model="exponential_decay",
        parameters={
            "amplitude": {"value": 2, "fixed": True},
            "tau": {"value": 1, "bounds": (0.1, 2)},
        },
    ).fit(curve)
    assert_allclose(fit.result.fitted_parameters["tau"].magnitude, 2, atol=1e-6)
    assert fit.result.parameter_names == ("tau",)
    assert fit.result.parameter_units == (scp.ur.s,)
    assert fit.result.diagnostics["covariance_reason"] == "active_bounds"
