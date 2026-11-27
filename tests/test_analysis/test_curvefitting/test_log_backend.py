"""Qualify explicit private coordinates through the existing 1D fitting engine."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.analysis.curvefitting._modelspec import _ComponentSpec
from spectrochempy.analysis.curvefitting._modelspec import _FitModelSpec
from spectrochempy.analysis.curvefitting._modelspec import _ParamSpec
from spectrochempy.analysis.curvefitting._parameter_coordinates import (
    _ParameterCoordinates,
)
from spectrochempy.analysis.curvefitting._parameter_plan import _compile_parameter_plan
from spectrochempy.analysis.curvefitting._parameter_space import _coordinate_vector
from spectrochempy.analysis.curvefitting._prepared_parameter import _ParameterLayout
from spectrochempy.analysis.curvefitting.optimize import _optimize


def install_coordinates(monkeypatch, *, scale=1, reference=1):
    import spectrochempy.analysis.curvefitting.optimize as module

    original = module._optimize
    captured = []

    def run(*args, **kwargs):
        plan = kwargs["parameter_plan"]
        coordinates = _ParameterCoordinates(
            _ParameterLayout(plan.parameters, (2, 0.5)),
            ("log", "lin"),
            (reference / scale, None),
        )
        captured.append(coordinates)
        return original(*args, **kwargs, parameter_coordinates=coordinates)

    monkeypatch.setattr(module, "_optimize", run)
    return captured


def estimator(frontend, scale=1, **options):
    def line(x, a=1.0, b=0.0, c=0.0):
        return scale * a * x + b + c

    options = {"bounds_mode": "native", "autoampl": False, "autobase": False, **options}
    if frontend == "python":
        model = scp.Model(line)
        params = model.make_params(
            a=scp.Parameter(0.3 / scale, min=0.001 / scale, max=10 / scale),
            b=0.1,
            c=scp.Parameter(expr="b"),
        )
        return scp.Optimize(model=model, params=params, **options), params
    script = f"""COMMON:
$ a: {0.3 / scale}, {0.001 / scale}, {10 / scale}
$ b: 0.1, none, none
> c: b
MODEL: line
shape: customline
> a: a
> b: b
> c: c
"""
    fit = scp.Optimize(
        script="", usermodels={"customline": line}, amplitude_mode="area", **options
    )
    fit.script = script
    return fit, None


@pytest.mark.parametrize("frontend", ["python", "common"])
@pytest.mark.parametrize("scale", [1e-6, 1, 1e6])
def test_fitresult_keeps_physical_covariance_and_configuration(
    monkeypatch, frontend, scale
):
    captured = install_coordinates(monkeypatch, scale=scale)
    x = np.linspace(-1, 1, 51)[::-1]
    y = 2 * x + 0.6 + 0.01 * np.cos(7 * x)
    source = scp.NDDataset(y, coordset=[scp.Coord(x, rounding=False)], units="V")
    source[[3, 8, 17]] = scp.MASKED
    first, params = estimator(frontend, scale)
    second, _ = estimator(frontend, scale)
    initial = deepcopy(params)
    first.fit(source)
    result = first.result
    valid = ~source.mask
    design = np.column_stack([scale * x[valid], np.full(valid.sum(), 2.0)])
    expected = np.linalg.lstsq(design, y[valid], rcond=None)[0]
    covariance = np.linalg.inv(design.T @ design) * result.rss / (valid.sum() - 2)
    names = result.parameter_names if frontend == "python" else ("a", "b")
    assert result.diagnostics["success"]
    assert result.diagnostics["physical_stationarity_status"] == "satisfied"
    assert result.diagnostics["jacobian_coordinates"] == "solver"
    np.testing.assert_allclose(result.covariance, covariance, rtol=2e-6, atol=0)
    np.testing.assert_allclose(result.stderr, np.sqrt(np.diag(covariance)), rtol=2e-6)
    np.testing.assert_allclose(
        result.fitted.data.squeeze()[valid],
        expected[0] * scale * x[valid] + 2 * expected[1],
        rtol=1e-7,
        atol=1e-9,
    )
    derivatives = [expected[0] * 2, 0.5]
    np.testing.assert_allclose(
        first.jacobian / derivatives, -design, rtol=2e-6, atol=1e-9
    )
    assert len(names) == 2
    assert captured[0].layout.free_values[0] == 0.3 / scale
    if frontend == "python":
        assert (
            params["a"].value == initial["a"].value == first.initial_params["a"].value
        )
        assert result.fitted_parameters["c"] == result.fitted_parameters["b"]
        assert result.diagnostics["derived_uncertainty_status"] == "not_propagated"
    before = first.result.covariance.copy()
    second.fit(source)
    np.testing.assert_array_equal(first.result.covariance, before)
    np.testing.assert_allclose(second.result.fitted.data, result.fitted.data)
    assert captured[0] is not captured[1]


@pytest.mark.parametrize("frontend", ["python", "common"])
def test_budget_failure_keeps_existing_uncertainty_reason(monkeypatch, frontend):
    install_coordinates(monkeypatch)
    x = np.linspace(0, 1, 31)
    source = scp.NDDataset(2 * x + 0.6, coordset=[scp.Coord(x)])
    fit, _ = estimator(frontend, max_fun_calls=1)
    result = fit.fit(source).result
    assert not result.diagnostics["success"]
    assert result.diagnostics["solver_nfev"] == 1
    assert result.covariance is None
    assert result.diagnostics["covariance_reason"] == "not_converged"


def test_rank_deficiency_keeps_existing_uncertainty_reason(monkeypatch):
    install_coordinates(monkeypatch)
    x = np.ones(31)
    source = scp.NDDataset(2.6 + 0.01 * np.cos(np.arange(31)), coordset=[scp.Coord(x)])
    fit, _ = estimator("python", max_fun_calls=300)
    result = fit.fit(source).result
    assert result.diagnostics["success"]
    assert result.covariance is None
    assert result.diagnostics["covariance_reason"] == "rank_deficient"


@pytest.mark.parametrize("reference", [1e-20, 1, 1e20])
def test_log_reference_shift_keeps_physical_fit(monkeypatch, reference):
    install_coordinates(monkeypatch, reference=reference)
    x = np.linspace(-1, 1, 21)
    y = 2 * x + 0.6 + 0.01 * np.cos(3 * x)
    fit, _ = estimator("python", max_fun_calls=300)
    result = fit.fit(scp.NDDataset(y, coordset=[scp.Coord(x)])).result
    design = np.column_stack([x, np.full(x.size, 2.0)])
    expected = np.linalg.lstsq(design, y, rcond=None)[0]
    assert result.diagnostics["success"]
    np.testing.assert_allclose(
        [result.fitted_parameters[k] for k in ("a", "b")], expected, rtol=1e-8
    )


def definition():
    return _FitModelSpec(
        common_params={
            "a": _ParamSpec("a", 0.3, bounds=(0.01, 10)),
            "b": _ParamSpec("b", 0.0),
        }
    )


def prepared(spec):
    plan = _compile_parameter_plan(spec)
    return plan, _ParameterCoordinates(
        _ParameterLayout(plan.parameters), ("log", "lin"), (1, None)
    )


@pytest.mark.parametrize("method", ["lm", "trf", "simplex", "basinhopping"])
def test_unqualified_methods_fail_before_objective(method):
    spec = definition()
    plan, coordinates = prepared(spec)

    def forbidden(*args):
        pytest.fail("Invalid configurations must fail before evaluating a model")

    with pytest.raises(ValueError, match="least-squares"):
        _optimize(
            forbidden,
            spec,
            method=method,
            bounds_mode="native",
            parameter_plan=plan,
            parameter_coordinates=coordinates,
        )


@pytest.mark.parametrize("mismatch", ["initial", "bounds", "identity", "order"])
def test_coordinate_mismatches_are_rejected_before_mutation(mismatch):
    spec = definition()
    plan, coordinates = prepared(spec)
    records = list(coordinates.layout.parameters)
    if mismatch == "initial":
        records[0] = replace(records[0], value=0.4)
    elif mismatch == "bounds":
        records[0] = replace(records[0], bounds=(0.01, 20))
    elif mismatch == "identity":
        records[0] = replace(records[0], identifier=("components", "other", "a"))
    else:
        records.reverse()
    coordinates = _ParameterCoordinates(
        _ParameterLayout(records),
        ("log", "lin") if mismatch != "order" else ("lin", "log"),
        (1, None) if mismatch != "order" else (None, 1),
    )
    with pytest.raises(ValueError, match="must match"):
        _coordinate_vector(spec, plan, coordinates)
    assert spec.common_params["a"].value == 0.3


def test_coordinates_require_native_canonical_nonempty_plan():
    spec = definition()
    plan, coordinates = prepared(spec)
    with pytest.raises(ValueError, match="native bounds"):
        _optimize(
            lambda *args: None,
            spec,
            parameter_plan=plan,
            parameter_coordinates=coordinates,
        )
    for source, compiled in [(spec, None), (spec.to_fitparameters(), plan)]:
        with pytest.raises(ValueError, match="canonical scalar plan"):
            _coordinate_vector(source, compiled, coordinates)
    with pytest.raises(TypeError, match="prepared coordinate"):
        _coordinate_vector(spec, plan, {})
    fixed = _FitModelSpec(common_params={"a": _ParamSpec("a", 1, vary=False)})
    fixed_plan = _compile_parameter_plan(fixed)
    with pytest.raises(ValueError, match="nonempty"):
        _coordinate_vector(
            fixed,
            fixed_plan,
            _ParameterCoordinates(_ParameterLayout(fixed_plan.parameters)),
        )
    unqualified = definition()
    unqualified.constraints = {}
    with pytest.raises(ValueError, match="qualified canonical"):
        _coordinate_vector(
            unqualified, _compile_parameter_plan(unqualified), coordinates
        )


@pytest.mark.parametrize(
    "mutation",
    ["component_order", "constraints", "experiment", "expression", "fixed_value"],
)
def test_stale_plan_is_rejected_before_objective_and_restoration(mutation):
    spec = _FitModelSpec(
        components=[
            _ComponentSpec(
                label,
                "gaussian",
                {"ampl": _ParamSpec("ampl", value, bounds=(0.01, 10))},
            )
            for label, value in [("first", 1.0), ("second", 2.0)]
        ],
        common_params={
            "fixed": _ParamSpec("fixed", 3, vary=False),
            "linked": _ParamSpec("linked", vary=False, reference="fixed * 2"),
        },
    )
    plan = _compile_parameter_plan(spec)
    coordinates = _ParameterCoordinates(
        _ParameterLayout(plan.parameters), ("log", "log"), (1, 1)
    )
    if mutation == "component_order":
        spec.components.reverse()
    elif mutation == "constraints":
        spec.constraints = {}
    elif mutation == "experiment":
        spec.expvars = ["ampl"]
        spec.expnumber = 2
    elif mutation == "expression":
        spec.common_params["linked"].reference = "fixed * 3"
    else:
        spec.common_params["fixed"].value = 4
    before = [(c.label, c.params["ampl"].value) for c in spec.components]

    def forbidden(*args):
        pytest.fail("Stale plans must be rejected before objective evaluation")

    with pytest.raises(ValueError, match="current canonical"):
        _optimize(
            forbidden,
            spec,
            bounds_mode="native",
            parameter_plan=plan,
            parameter_coordinates=coordinates,
        )
    assert [(c.label, c.params["ampl"].value) for c in spec.components] == before


@pytest.mark.parametrize(
    "target", ["free_value", "fixed_value", "free_bound", "fixed_bound"]
)
@pytest.mark.parametrize("shape", [(), (1,)])
def test_invalidated_scalar_qualification_rejects_arrays_before_objective(
    target, shape
):
    spec = definition()
    spec.common_params["fixed"] = _ParamSpec("fixed", 3, vary=False)
    plan, coordinates = prepared(spec)
    parameter = spec.common_params["a" if target.startswith("free") else "fixed"]
    if target.endswith("value"):
        parameter.value = np.full(shape, parameter.value)
    else:
        low, high = parameter.bounds
        parameter.bounds = (np.full(shape, 0.01 if low is None else low), high)
    assert not _compile_parameter_plan(spec).vector_compatible

    def forbidden(*args):
        pytest.fail("Non-scalar declarations must fail before model evaluation")

    with pytest.raises(ValueError, match="qualified scalar"):
        _optimize(
            forbidden,
            spec,
            bounds_mode="native",
            parameter_plan=plan,
            parameter_coordinates=coordinates,
        )


def test_explicit_identity_coordinates_preserve_native_evaluation_history():
    initial = definition()
    plan = _compile_parameter_plan(initial)
    coordinates = _ParameterCoordinates(_ParameterLayout(plan.parameters))
    x = np.linspace(-1, 1, 31)
    observed = 2 * x + 0.3 + 0.01 * np.cos(5 * x)
    outcomes, histories = [], []
    for selected in (None, coordinates):
        history = []

        def objective(spec, data, history=history):
            a, b = (spec.common_params[k].value for k in ("a", "b"))
            history.append((a, b))
            return a * x + b - data

        outcomes.append(
            _optimize(
                objective,
                deepcopy(initial),
                args=(observed,),
                bounds_mode="native",
                parameter_plan=plan,
                parameter_coordinates=selected,
                maxfun=300,
            )
        )
        histories.append(history)
    assert histories[0] == histories[1]
    assert outcomes[0][1:3] == outcomes[1][1:3]
    for key, value in outcomes[0][3].items():
        if key != "jacobian_coordinates":
            np.testing.assert_array_equal(value, outcomes[1][3][key])


@pytest.mark.parametrize("target", [-1, 20])
def test_positive_bound_optimum_uses_physical_bound_exclusions(monkeypatch, target):
    install_coordinates(monkeypatch)
    x = np.linspace(-1, 1, 41)
    source = scp.NDDataset(target * x + 0.2, coordset=[scp.Coord(x)])
    fit, _ = estimator("python")
    result = fit.fit(source).result
    assert result.diagnostics["success"]
    assert result.diagnostics["covariance_reason"] == "active_bounds"
    assert result.covariance is None
    assert result.fitted_parameters["a"] == pytest.approx(
        0.001 if target < 0 else 10, abs=1e-7
    )
