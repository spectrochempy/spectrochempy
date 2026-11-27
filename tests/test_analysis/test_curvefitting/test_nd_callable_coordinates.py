"""Private ND coordinate wiring with physical output and uncertainty oracles."""

from copy import deepcopy
from dataclasses import replace
from functools import partial
from importlib import import_module

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.analysis.curvefitting._optimize_model import _parameter_layout
from spectrochempy.analysis.curvefitting._optimize_model import _rescale_jacobian
from spectrochempy.analysis.curvefitting._optimize_model import fit_callable_model
from spectrochempy.analysis.curvefitting._parameter_coordinates import (
    _ParameterCoordinates,
)
from spectrochempy.analysis.curvefitting._prepared_parameter import _ParameterLayout


def problem(gain_scale=1, unit="V", normalization=2):
    factor = (1 * scp.ur.V).to(unit).magnitude
    specs = {
        "z_gain": {
            "value": 0.3 / gain_scale * scp.ur.V,
            "unit": unit,
            "bounds": (0.001 / gain_scale * scp.ur.V, 10 / gain_scale * scp.ur.V),
            "scale": 1 / gain_scale * scp.ur.V,
        },
        "floor": {"value": 0.2 * scp.ur.V, "unit": unit, "fixed": True},
        "a_tilt": {
            "value": -0.1 * scp.ur.V,
            "unit": unit,
            "bounds": (-2 * scp.ur.V, 2 * scp.ur.V),
            "scale": 0.5 * scp.ur.V,
        },
    }
    layout = _parameter_layout(specs)
    space = _ParameterCoordinates(
        _ParameterLayout(layout.parameters, (normalization, 0.5 * factor)),
        ("log", "lin"),
        (0.5 / gain_scale * factor, None),
    )
    x = scp.Coord.linspace(1, -1, 17, rounding=False)
    y = scp.Coord.linspace(-0.6, 1.2, 13, rounding=False)
    xx, yy = np.meshgrid(x.data, y.data)
    data = scp.NDDataset(
        2 * xx - 0.3 * (yy + 1.2 * xx) + 0.2 + 0.01 * np.cos(3 * xx + 7 * yy),
        coordset=[y, x],
        units="V",
    )
    data[2:5, 1:8] = scp.MASKED
    data.data[0, 0] = np.nan
    valid = ~data.mask & np.isfinite(data.data)
    design = np.column_stack((gain_scale * xx[valid], (yy + 1.2 * xx)[valid])) / factor

    def model(template, z_gain, floor, a_tilt):
        tx, ty = template.x.data[None, :], template.y.data[:, None]
        return (gain_scale * z_gain * tx + a_tilt * (ty + 1.2 * tx) + floor) / factor

    return data, model, specs, space, valid, design


@pytest.mark.parametrize("gain_scale", [1e-6, 1, 1e6])
@pytest.mark.parametrize("unit", ["V", "mV"])
@pytest.mark.parametrize("normalization", [0.25, 2])
def test_private_backend_physical_solution_jacobian_and_covariance(
    gain_scale, unit, normalization
):
    data, model, specs, space, valid, design = problem(gain_scale, unit, normalization)
    before = deepcopy(specs)
    original = data.copy()
    options = {"ftol": 1e-12, "xtol": 1e-12, "gtol": 1e-12}
    fit, solution = fit_callable_model(
        data, model, specs, 300, False, options, parameter_coordinates=space
    )
    observed = data.data[valid] - 0.2
    expected = np.linalg.lstsq(design, observed, rcond=None)[0]
    residual = observed - design @ expected
    covariance = (
        np.linalg.inv(design.T @ design) * (residual @ residual) / (valid.sum() - 2)
    )
    assert solution.success
    assert fit["names"] == ("z_gain", "a_tilt")
    np.testing.assert_allclose(fit["values"], expected, rtol=1e-7)
    np.testing.assert_allclose(space.to_physical(solution.x), expected, rtol=1e-7)
    column_scale = np.max(np.abs(design), axis=0)
    np.testing.assert_allclose(
        fit["jacobian"] / column_scale, -design / column_scale, rtol=2e-6, atol=2e-8
    )
    np.testing.assert_allclose(fit["covariance"], covariance, rtol=3e-6, atol=0)
    assert fit["diagnostics"]["covariance_reason"] is None
    assert fit["diagnostics"]["jacobian_coordinates"] == "physical"
    assert fit["diagnostics"]["covariance_coordinates"] == "physical"
    np.testing.assert_allclose(fit["residuals"].data[valid], residual, atol=1e-8)
    np.testing.assert_array_equal(fit["fitted"].mask, ~valid)
    np.testing.assert_array_equal(fit["fitted"].x.data, data.x.data)
    np.testing.assert_array_equal(fit["fitted"].y.data, data.y.data)
    assert fit["fitted"].units == data.units
    assert fit["units"] == (scp.ur.Unit(unit), scp.ur.Unit(unit))
    np.testing.assert_array_equal(data.data, original.data)
    np.testing.assert_array_equal(data.mask, original.mask)
    assert specs == before
    config = fit["config"]["parameter_coordinates"]["z_gain"]
    assert config["log_reference"] == space.references[0]
    assert config["initial_value"] == space.layout.free_values[0]
    assert config["normalization_scale"] == normalization
    assert config["coordinate_transform"] == "log"
    assert config["initial_value"] != fit["values"][0]
    assert options == {"ftol": 1e-12, "xtol": 1e-12, "gtol": 1e-12}


def test_result_contract_and_configuration_isolation_through_existing_estimator(
    monkeypatch,
):
    data, model, specs, space, *_ = problem()
    module = import_module("spectrochempy.analysis.curvefitting.optimize")
    monkeypatch.setattr(
        module,
        "fit_callable_model",
        partial(fit_callable_model, parameter_coordinates=space),
    )
    first = scp.Optimize(model=model, parameters=specs).fit(data)
    second = scp.Optimize(model=model, parameters=specs).fit(data)
    result = first.result
    assert result.parameter_names == ("z_gain", "a_tilt")
    assert result.fitted_parameters["floor"].to("V").magnitude == 0.2
    np.testing.assert_allclose(result.stderr, np.sqrt(np.diag(result.covariance)))
    assert result.confidence_intervals.shape == (2, 2)
    assert result.parameters["parameter_coordinates"]["z_gain"]["initial_value"] == 0.3
    assert not first.jacobian.flags.writeable
    result.parameters["parameter_coordinates"]["z_gain"]["log_reference"] = 999
    assert (
        first.result.parameters["parameter_coordinates"]["z_gain"]["log_reference"]
        == 0.5
    )
    assert (
        second.result.parameters["parameter_coordinates"]["z_gain"]["log_reference"]
        == 0.5
    )
    assert (
        first.result.fitted.history_entries[-1]["parameters"]["parameter_coordinates"][
            "z_gain"
        ]["log_reference"]
        == 0.5
    )
    first.fit(data)
    np.testing.assert_array_equal(first.result.covariance, second.result.covariance)


@pytest.mark.parametrize(
    "field", ["value", "bounds", "fixed", "unit", "name", "identifier"]
)
def test_stale_coordinate_records_rejected_before_callback(field):
    data, _, specs, space, *_ = problem()
    records = list(space.layout.parameters)
    index = 1 if field in ("value", "bounds") else 0
    changes = {
        "value": {"value": 0.4},
        "bounds": {"bounds": (-1, 1)},
        "fixed": {"fixed": True},
        "unit": {"unit": scp.ur.mV},
        "name": {"name": "other"},
        "identifier": {"identifier": ("other", "z_gain")},
    }
    records[index] = replace(records[index], **changes[field])
    layout = _ParameterLayout(tuple(records))
    candidate = _ParameterCoordinates(
        layout,
        tuple("log" if p.name == "z_gain" else "lin" for p in layout.free_parameters),
        tuple(0.5 if p.name == "z_gain" else None for p in layout.free_parameters),
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Model must not be evaluated for a stale coordinate layout")

    with pytest.raises(ValueError, match="match the current model layout"):
        fit_callable_model(
            data, forbidden, specs, 300, False, {}, parameter_coordinates=candidate
        )


@pytest.mark.parametrize("case", ["type", "x_scale"])
@pytest.mark.parametrize("projected", [False, True])
def test_incompatible_coordinate_configuration_fails_before_callback(case, projected):
    data, _, specs, space, *_ = problem()

    def forbidden(*args, **kwargs):
        pytest.fail("Model must not be evaluated for incompatible coordinates")

    with pytest.raises(
        (TypeError, ValueError), match="coordinates|coordinate|amplitudes"
    ):
        fit_callable_model(
            data,
            forbidden,
            specs,
            300,
            projected,
            {"x_scale": 1} if case == "x_scale" else {},
            parameter_coordinates=object() if case == "type" else space,
        )


def test_coordinate_record_order_must_match_current_declarations():
    data, _, specs, space, *_ = problem()
    reordered = _ParameterCoordinates(
        _ParameterLayout(tuple(reversed(space.layout.parameters))),
        ("lin", "log"),
        (None, 0.5),
    )
    with pytest.raises(ValueError, match="match the current model layout"):
        fit_callable_model(
            data,
            lambda *a, **k: pytest.fail(),
            specs,
            300,
            False,
            {},
            parameter_coordinates=reordered,
        )


def small_grid(values):
    return scp.NDDataset(
        values, coordset=[scp.Coord([0, 1, 2]), scp.Coord([0.1, 0.4, 0.7, 1.0])]
    )


def test_all_fixed_does_not_call_a_solver(monkeypatch):
    data = small_grid(np.arange(12.0).reshape(3, 4))
    specs = {"gain": {"value": 2, "fixed": True}}
    space = _ParameterCoordinates(_parameter_layout(specs))
    module = import_module("spectrochempy.analysis.curvefitting._optimize_model")

    def forbidden(*args, **kwargs):
        pytest.fail("All-fixed evaluation must not call a solver")

    monkeypatch.setattr(module, "_least_squares", forbidden)
    fit, solution = fit_callable_model(
        data,
        lambda template, gain: np.full(template.shape, gain),
        specs,
        20,
        False,
        {},
        parameter_coordinates=space,
    )
    assert solution is None
    assert fit["jacobian"] is None and fit["covariance"] is None
    assert fit["diagnostics"]["covariance_reason"] == "no_free_parameters"
    assert fit["config"]["parameter_coordinates"] == {}


def test_log_offset_units_are_rejected_before_model_evaluation():
    data = small_grid(np.ones((3, 4)))
    specs = {"temperature": {"value": 20, "bounds": (1, 50), "unit": "degC"}}
    space = _ParameterCoordinates(_parameter_layout(specs), ("log",), (10,))
    with pytest.raises(ValueError, match="multiplicative"):
        fit_callable_model(
            data,
            lambda *a, **k: pytest.fail(),
            specs,
            20,
            False,
            {},
            parameter_coordinates=space,
        )


def test_explicit_linear_coordinates_and_default_none_parity():
    data, model, specs, _, *_ = problem()
    ordinary, raw = fit_callable_model(data, model, specs, 300, False, {})
    explicit_none, same = fit_callable_model(
        data, model, specs, 300, False, {}, parameter_coordinates=None
    )
    for key in ("values", "jacobian", "covariance"):
        np.testing.assert_array_equal(ordinary[key], explicit_none[key])
    assert raw.nfev == same.nfev
    assert ordinary["diagnostics"] == explicit_none["diagnostics"]
    assert "config" not in ordinary
    linear = _ParameterCoordinates(_parameter_layout(specs))
    fitted, _ = fit_callable_model(
        data, model, specs, 300, False, {}, parameter_coordinates=linear
    )
    np.testing.assert_allclose(fitted["values"], ordinary["values"], rtol=1e-6)
    np.testing.assert_allclose(fitted["covariance"], ordinary["covariance"], rtol=1e-5)


def test_named_components_are_summed_without_projection():
    data, model, specs, space, *_ = problem()

    def components(template, z_gain, floor, a_tilt):
        x = template.x.data[None, :]
        signal = np.broadcast_to(z_gain * x, template.shape)
        return {
            "signal": signal,
            "background": model(template, z_gain, floor, a_tilt) - signal,
        }

    fit, _ = fit_callable_model(
        data, components, specs, 300, False, {}, parameter_coordinates=space
    )
    assert fit["component_names"] == ("signal", "background")
    assert fit["amplitudes"] == {}
    assert fit["diagnostics"]["n_linear_parameters"] == 0
    np.testing.assert_allclose(
        sum(part.data for part in fit["components"]), fit["fitted"].data
    )


def test_jacobian_restoration_avoids_false_intermediate_overflow_and_underflow():
    matrix = np.array([[1e300, 1e-300], [-1e300, -1e-300]])
    factors = np.array([1e300, 1e-300])
    np.testing.assert_allclose(
        _rescale_jacobian(matrix, factors, factors), matrix, rtol=2e-15, atol=0
    )
    for numerator, derivative in ((1e300, [1e-300]), (1e-300, [1e300])):
        with pytest.raises(ValueError, match="outside float64 range"):
            _rescale_jacobian(np.ones((2, 1)), numerator, derivative)


@pytest.mark.parametrize("observed_gain", [0.2, 2])
def test_active_physical_bounds_exclude_covariance(observed_gain):
    x = np.array([0.1, 0.4, 0.7, 1.0])[None, :]
    data = small_grid(np.broadcast_to(observed_gain * x, (3, 4)))
    specs = {"gain": {"value": 0.75, "bounds": (0.5, 1), "scale": 1}}
    space = _ParameterCoordinates(_parameter_layout(specs), ("log",), (0.8,))
    fit, raw = fit_callable_model(
        data,
        lambda template, gain: np.broadcast_to(gain * x, template.shape),
        specs,
        300,
        False,
        {"ftol": 1e-12, "xtol": 1e-12, "gtol": 1e-12},
        parameter_coordinates=space,
    )
    assert raw.success
    np.testing.assert_allclose(fit["values"], [0.5 if observed_gain < 0.5 else 1])
    assert fit["covariance"] is None
    assert fit["diagnostics"]["covariance_reason"] == "active_bounds"
    assert fit["diagnostics"]["active_parameter_bounds"] == ("gain",)


def test_rank_deficient_physical_problem_has_no_covariance():
    x = np.array([0.1, 0.4, 0.7, 1.0])[None, :]
    data = small_grid(np.broadcast_to(2 * x + 0.01 * np.cos(3 * x), (3, 4)))
    specs = {
        "gain": {"value": 0.5, "bounds": (0.1, 4)},
        "other": {"value": 0.5, "bounds": (-4, 4)},
    }
    space = _ParameterCoordinates(_parameter_layout(specs), ("log", "lin"), (1, None))
    fit, raw = fit_callable_model(
        data,
        lambda template, gain, other: np.broadcast_to(
            (gain + other) * x, template.shape
        ),
        specs,
        300,
        False,
        {},
        parameter_coordinates=space,
    )
    assert raw.success
    assert fit["covariance"] is None
    assert fit["diagnostics"]["covariance_reason"] == "rank_deficient"


def test_inference_normalization_overflow_cannot_hide_an_active_bound():
    x = np.array([0.1, 0.4, 0.7, 1.0])[None, :]
    data = small_grid(np.broadcast_to(2 * x, (3, 4)))
    specs = {"gain": {"value": 0.75, "bounds": (0.5, 1), "scale": 1e-309}}
    space = _ParameterCoordinates(
        _ParameterLayout(_parameter_layout(specs).parameters, (1,)), ("log",), (0.8,)
    )
    with np.errstate(all="raise"):
        fit, raw = fit_callable_model(
            data,
            lambda template, gain: np.broadcast_to(gain * x, template.shape),
            specs,
            300,
            False,
            {},
            parameter_coordinates=space,
        )
    assert raw.success
    np.testing.assert_allclose(fit["values"], [1], rtol=1e-6)
    assert np.isfinite(fit["jacobian"]).all()
    assert fit["diagnostics"]["active_parameter_bounds"] == ("gain",)
    assert fit["covariance"] is None
    assert fit["diagnostics"]["covariance_reason"] == "numeric_range"


def test_exhausted_budget_preserves_solver_failure_and_excludes_covariance():
    data, model, specs, space, *_ = problem()
    fit, raw = fit_callable_model(
        data, model, specs, 1, False, {}, parameter_coordinates=space
    )
    assert not raw.success and raw.nfev == 1
    assert fit["covariance"] is None
    assert fit["diagnostics"]["covariance_reason"] == "not_converged"


def test_degrees_of_freedom_exclusion_is_preserved():
    data = small_grid(np.ones((3, 4)))
    data.mask = np.ones(data.shape, dtype=bool)
    data.mask[0, 0] = False
    specs = {"gain": {"value": 0.5, "bounds": (0.1, 4)}}
    space = _ParameterCoordinates(_parameter_layout(specs), ("log",), (1,))
    fit, _ = fit_callable_model(
        data,
        lambda template, gain: np.full(template.shape, gain),
        specs,
        300,
        False,
        {},
        parameter_coordinates=space,
    )
    assert fit["diagnostics"]["degrees_of_freedom"] == 0
    assert fit["covariance"] is None
    assert fit["diagnostics"]["covariance_reason"] == "insufficient_degrees_of_freedom"


@pytest.mark.parametrize("proposal", [-1000, 1000])
def test_unrepresentable_solver_proposal_stops_before_calling_model(
    monkeypatch, proposal
):
    data = small_grid(np.ones((3, 4)))
    specs = {"gain": {"value": 1, "bounds": (1e-300, None)}}
    space = _ParameterCoordinates(_parameter_layout(specs), ("log",), (1,))
    calls = []

    def model(template, gain):
        calls.append(gain)
        return np.full(template.shape, gain)

    def solver(objective, initial, **options):
        objective([proposal])
        pytest.fail("The inverse proposal must be rejected")

    module = import_module("spectrochempy.analysis.curvefitting._optimize_model")
    monkeypatch.setattr(module, "_least_squares", solver)
    with pytest.raises(ValueError, match="underflow|overflow"):
        fit_callable_model(
            data, model, specs, 300, False, {}, parameter_coordinates=space
        )
    assert calls == [1]


def test_stopping_options_remain_explicit_and_do_not_guarantee_stationarity():
    x = np.array([0.1, 0.4, 0.7, 1.0])[None, :]
    data = small_grid(np.broadcast_to(2 * x, (3, 4)))
    specs = {"gain": {"value": 1, "bounds": (1, 4)}}
    space = _ParameterCoordinates(_parameter_layout(specs), ("log",), (1,))

    def model(template, gain):
        return np.broadcast_to(gain * x, template.shape)

    stalled, raw = fit_callable_model(
        data, model, specs, 500, False, {}, parameter_coordinates=space
    )
    assert raw.success
    assert stalled["values"][0] < 1.01
    fitted, raw = fit_callable_model(
        data,
        model,
        specs,
        500,
        False,
        {"ftol": None, "gtol": None, "xtol": 1e-12},
        parameter_coordinates=space,
    )
    assert raw.success
    np.testing.assert_allclose(fitted["values"], [2], rtol=1e-7)
