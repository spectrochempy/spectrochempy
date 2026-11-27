"""Qualify private log coordinates for a nonnegative variable projection."""

from copy import deepcopy
from functools import partial
from importlib import import_module

import numpy as np
import pytest
from scipy.optimize import least_squares

import spectrochempy as scp
from spectrochempy.analysis.curvefitting._optimize_model import _parameter_layout
from spectrochempy.analysis.curvefitting._optimize_model import fit_callable_model
from spectrochempy.analysis.curvefitting._parameter_coordinates import (
    _ParameterCoordinates,
)
from spectrochempy.analysis.curvefitting._prepared_parameter import _ParameterLayout


def projected_problem(unit="s", normalization=1, intensity=1, second=0.6):
    factor = (1 * scp.ur.s).to(unit).magnitude
    x = scp.Coord(
        [3.1, 2.5, 1.8, 1.3, 0.9, 0.65, 0.4, 0.25, 0.13, 0.07, 0.03, 0], units="s"
    )
    y = scp.Coord([-0.9, -0.4, 0.05, 0.7, 1.1])
    specs = {
        "tilt": {"value": 0.2, "fixed": True},
        "tau": {
            "value": 0.35 * scp.ur.s,
            "unit": unit,
            "bounds": (0.08 * scp.ur.s, 3 * scp.ur.s),
            "scale": 0.8 * scp.ur.s,
        },
    }
    space = _ParameterCoordinates(
        _ParameterLayout(_parameter_layout(specs).parameters, (normalization,)),
        ("log",),
        (0.5 * factor,),
    )
    xx, yy = np.meshgrid(x.data, y.data)

    def basis(tau, tilt=0.2):
        time = tau / factor
        first = np.exp(-xx * (1 + tilt * yy) / time)
        second = np.exp(-0.3 * xx) * np.cos(yy + 0.5)
        return np.column_stack((first.ravel(), second.ravel()))

    data = scp.NDDataset(
        intensity
        * (
            (basis(0.8 * factor) @ [1.8, second]).reshape(xx.shape)
            + 0.001 * np.sin(7 * xx + 3 * yy)
        ),
        coordset=[y, x],
        units="V",
    )
    data[1:3, 3:6] = scp.MASKED
    data.data[0, 0] = np.nan
    valid = ~data.mask & np.isfinite(data.data)

    def model(template, tilt, tau):
        matrix = basis(tau, tilt)
        return {
            "decay": matrix[:, 0].reshape(template.shape),
            "background": matrix[:, 1].reshape(template.shape),
        }

    def derivative(tau):
        matrix = basis(tau)
        dtau = (
            matrix[:, 0] * (xx * (1 + 0.2 * yy)).ravel() / (tau / factor) ** 2 / factor
        )
        return np.column_stack((dtau, np.zeros_like(dtau)))

    return data, model, specs, space, valid, basis, derivative, factor


def joint_reference(data, valid, basis, factor):
    support = valid.ravel()
    scale = np.max(np.abs(data.data[valid]))
    return least_squares(
        lambda p: data.data[valid] / scale - basis(p[0] * factor)[support] @ p[1:],
        [0.35, 1, 0.5],
        bounds=([0.08, 0, 0], [3, np.inf, np.inf]),
        jac="3-point",
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    ), scale


def profiled_derivative(matrix, derivative, observed, amplitudes):
    active = amplitudes > 0
    design = matrix[:, active]
    change = derivative[:, active]
    coefficients = amplitudes[active]
    residual = observed - design @ coefficients
    dc = np.linalg.solve(
        design.T @ design, change.T @ residual - design.T @ change @ coefficients
    )
    return -change @ coefficients - design @ dc


@pytest.mark.parametrize("unit", ["s", "ms"])
@pytest.mark.parametrize("normalization", [0.25, 2])
@pytest.mark.parametrize("intensity", [1e-6, 1, 1e6])
def test_log_projection_matches_joint_fit_and_physical_profile_derivative(
    unit, normalization, intensity
):
    data, model, specs, space, valid, basis, derivative, factor = projected_problem(
        unit, normalization, intensity
    )
    original, initial = data.copy(), deepcopy(specs)
    options = {"ftol": 1e-12, "xtol": 1e-12, "gtol": 1e-12}
    reference, scale = joint_reference(data, valid, basis, factor)
    fit, raw = fit_callable_model(
        data, model, specs, 300, True, options, parameter_coordinates=space
    )
    assert reference.success and raw.success
    tau = float(fit["values"][0])
    coefficients = np.array(list(fit["amplitudes"].values()))
    np.testing.assert_allclose(tau / factor, reference.x[0], rtol=2e-6)
    np.testing.assert_allclose(coefficients / scale, reference.x[1:], rtol=2e-6)
    expected = profiled_derivative(
        basis(tau)[valid.ravel()],
        derivative(tau)[valid.ravel()],
        data.data[valid],
        coefficients,
    )
    np.testing.assert_allclose(
        fit["jacobian"][:, 0] / (intensity / factor),
        expected / (intensity / factor),
        rtol=3e-5,
        atol=3e-8,
    )
    np.testing.assert_allclose(space.to_physical(raw.x), fit["values"], rtol=2e-15)
    assert fit["names"] == ("tau",)
    assert fit["units"] == (scp.ur.Unit(unit),)
    assert fit["component_names"] == ("decay", "background")
    assert fit["covariance"] is None
    assert fit["diagnostics"]["covariance_reason"] == "profiled_amplitudes"
    assert fit["diagnostics"]["n_nonlinear_parameters"] == 1
    assert fit["diagnostics"]["n_linear_parameters"] == 2
    assert fit["diagnostics"]["degrees_of_freedom"] == valid.sum() - 3
    assert fit["diagnostics"]["active_amplitude_bounds"] == ()
    np.testing.assert_allclose(
        sum(p.data for p in fit["components"]), fit["fitted"].data
    )
    np.testing.assert_array_equal(fit["fitted"].mask, ~valid)
    np.testing.assert_array_equal(fit["fitted"].x.data, data.x.data)
    np.testing.assert_array_equal(data.data, original.data)
    np.testing.assert_array_equal(data.mask, original.mask)
    assert specs == initial
    assert (
        fit["config"]["parameter_coordinates"]["tau"]["initial_value"] == 0.35 * factor
    )


def test_stable_active_amplitude_bound_matches_joint_reference():
    data, model, specs, space, valid, basis, derivative, factor = projected_problem(
        second=-0.2
    )
    fit, raw = fit_callable_model(
        data, model, specs, 300, True, {}, parameter_coordinates=space
    )
    reference, scale = joint_reference(data, valid, basis, factor)
    assert raw.success and reference.success
    tau = float(fit["values"][0])
    amplitudes = np.array(list(fit["amplitudes"].values()))
    assert amplitudes[0] > 0 and amplitudes[1] == 0
    np.testing.assert_allclose(tau, reference.x[0], rtol=2e-5)
    np.testing.assert_allclose(amplitudes[0] / scale, reference.x[1], rtol=2e-5)
    expected = profiled_derivative(
        basis(tau)[valid.ravel()],
        derivative(tau)[valid.ravel()],
        data.data[valid],
        amplitudes,
    )
    np.testing.assert_allclose(fit["jacobian"][:, 0], expected, rtol=3e-5, atol=1e-7)
    assert fit["diagnostics"]["active_amplitude_bounds"] == ("background",)
    assert fit["diagnostics"]["covariance_reason"] == "profiled_amplitudes"
    assert fit["covariance"] is None


def test_projected_result_configuration_and_two_estimators_are_isolated(monkeypatch):
    data, model, specs, space, *_ = projected_problem()
    module = import_module("spectrochempy.analysis.curvefitting.optimize")
    monkeypatch.setattr(
        module,
        "fit_callable_model",
        partial(fit_callable_model, parameter_coordinates=space),
    )
    first = scp.Optimize(model=model, parameters=specs, linear_amplitudes=True).fit(
        data
    )
    second = scp.Optimize(model=model, parameters=specs, linear_amplitudes=True).fit(
        data
    )
    result = first.result
    assert result.parameter_names == ("tau",)
    assert result.stderr is None and result.confidence_intervals is None
    assert result.covariance is None
    assert not first.jacobian.flags.writeable
    result.amplitudes["decay"] = 999
    result.parameters["parameter_coordinates"]["tau"]["log_reference"] = 999
    np.testing.assert_equal(first.result.amplitudes, second.result.amplitudes)
    assert (
        first.result.parameters["parameter_coordinates"]["tau"]["log_reference"] == 0.5
    )
    first.fit(data)
    np.testing.assert_array_equal(first.jacobian, second.jacobian)


@pytest.mark.parametrize("q", [0.8, 1, 1.2])
def test_projection_active_set_change_keeps_conservative_inference(q):
    """Qualify both smooth sides without assigning a derivative to the kink."""
    shape = (3, 4)
    f = np.tile([-1, -1, 1, 1], 3).astype(float)
    g = np.tile([-1, 1, -1, 1], 3).astype(float)
    data = scp.NDDataset(
        (2 + g).reshape(shape), coordset=[scp.Coord([0, 1, 2]), scp.Coord([0, 1, 2, 3])]
    )
    specs = {"q": {"value": q, "bounds": (0.5, 1.5)}}
    space = _ParameterCoordinates(_parameter_layout(specs), ("log",), (1,))

    def model(template, q):
        return {
            "constant": np.ones(shape),
            "changing": (f + (q - 1) * g).reshape(shape),
        }

    fit, raw = fit_callable_model(
        data, model, specs, 1, True, {}, parameter_coordinates=space
    )
    delta = q - 1
    expected = max(delta / (1 + delta**2), 0)
    np.testing.assert_allclose(
        list(fit["amplitudes"].values()), [2, expected], atol=2e-15
    )
    np.testing.assert_allclose(fit["values"], [q], rtol=1e-15)
    assert raw.nfev == 1
    if q != 1:
        assert fit["diagnostics"]["active_amplitude_bounds"] == (
            ("changing",) if q < 1 else ()
        )
    assert fit["covariance"] is None
    assert fit["diagnostics"]["covariance_reason"] == "profiled_amplitudes"
    if q != 1:
        dc = (1 - delta**2) / (1 + delta**2) ** 2 if delta > 0 else 0
        expected_jac = -dc * (f + delta * g) - expected * g
        np.testing.assert_allclose(
            fit["jacobian"][:, 0], expected_jac, rtol=2e-6, atol=1e-9
        )


def test_projected_linear_only_fit_never_calls_the_nonlinear_solver(monkeypatch):
    data, model, specs, _, valid, *_ = projected_problem()
    specs["tau"]["fixed"] = True
    space = _ParameterCoordinates(_parameter_layout(specs))
    module = import_module("spectrochempy.analysis.curvefitting._optimize_model")
    monkeypatch.setattr(
        module,
        "_least_squares",
        lambda *a, **k: pytest.fail("No nonlinear solver expected"),
    )
    fit, raw = fit_callable_model(
        data, model, specs, 300, True, {}, parameter_coordinates=space
    )
    assert raw is None and fit["jacobian"] is None
    assert fit["values"].size == 0 and fit["names"] == ()
    assert fit["diagnostics"]["n_linear_parameters"] == 2
    assert fit["diagnostics"]["n_nonlinear_parameters"] == 0
    assert fit["diagnostics"]["degrees_of_freedom"] == valid.sum() - 2
    assert fit["covariance"] is None
    assert fit["diagnostics"]["covariance_reason"] == "profiled_amplitudes"
    assert fit["config"]["parameter_coordinates"] == {}


def test_duplicate_projected_columns_do_not_invent_joint_uncertainties():
    data, original, specs, space, *_ = projected_problem()

    def model(template, tilt, tau):
        part = original(template, tilt, tau)["decay"]
        return {"first": part, "duplicate": part.copy()}

    data.data = 2.3 * model(data, 0.2, 0.8)["first"]
    fit, raw = fit_callable_model(
        data, model, specs, 300, True, {}, parameter_coordinates=space
    )
    assert raw.success
    np.testing.assert_allclose(fit["values"], [0.8], rtol=1e-6)
    np.testing.assert_allclose(sum(fit["amplitudes"].values()), 2.3, rtol=1e-6)
    assert fit["covariance"] is None
    assert fit["diagnostics"]["covariance_reason"] == "profiled_amplitudes"


def test_zero_projected_support_has_no_invented_uncertainty():
    data, _, specs, space, *_ = projected_problem()
    fit, raw = fit_callable_model(
        data,
        lambda template, **p: {"unsupported": np.zeros(template.shape)},
        specs,
        300,
        True,
        {},
        parameter_coordinates=space,
    )
    assert raw.success
    assert fit["amplitudes"] == {"unsupported": 0}
    np.testing.assert_array_equal(fit["jacobian"], 0)
    assert fit["covariance"] is None
    assert fit["diagnostics"]["active_amplitude_bounds"] == ("unsupported",)
    assert fit["diagnostics"]["covariance_reason"] == "profiled_amplitudes"


def test_projected_stale_layout_fails_before_model():
    data, _, specs, space, *_ = projected_problem()
    specs["tilt"]["value"] = 0.3
    with pytest.raises(ValueError, match="match the current model layout"):
        fit_callable_model(
            data,
            lambda *a, **k: pytest.fail(),
            specs,
            300,
            True,
            {},
            parameter_coordinates=space,
        )


def test_explicit_none_projection_is_identical_to_existing_route():
    data, model, specs, _, *_ = projected_problem()
    first, raw = fit_callable_model(data, model, specs, 300, True, {})
    second, same = fit_callable_model(
        data, model, specs, 300, True, {}, parameter_coordinates=None
    )
    for name in ("values", "jacobian"):
        np.testing.assert_array_equal(first[name], second[name])
    assert first["amplitudes"] == second["amplitudes"]
    assert first["diagnostics"] == second["diagnostics"]
    assert raw.nfev == same.nfev
    assert "config" not in first and "config" not in second


def test_projected_fit_keeps_physical_parameter_bound_and_budget_status():
    data, model, specs, _, *_ = projected_problem()
    specs["tau"]["bounds"] = (0.08 * scp.ur.s, 0.5 * scp.ur.s)
    space = _ParameterCoordinates(_parameter_layout(specs), ("log",), (0.4,))
    options = {"ftol": 1e-12, "xtol": 1e-12, "gtol": 1e-12}
    fit, raw = fit_callable_model(
        data, model, specs, 300, True, options, parameter_coordinates=space
    )
    assert raw.success
    np.testing.assert_allclose(fit["values"], [0.5], rtol=1e-6)
    assert fit["diagnostics"]["active_parameter_bounds"] == ("tau",)
    assert fit["covariance"] is None
    assert fit["diagnostics"]["covariance_reason"] == "profiled_amplitudes"
    fit, raw = fit_callable_model(
        data, model, specs, 1, True, options, parameter_coordinates=space
    )
    assert not raw.success and raw.nfev == 1
    assert fit["covariance"] is None
    assert fit["diagnostics"]["covariance_reason"] == "profiled_amplitudes"
