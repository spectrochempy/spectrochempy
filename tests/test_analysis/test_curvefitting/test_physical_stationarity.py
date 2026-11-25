"""Qualify physical first-order diagnostics independently of solver success."""

import numpy as np
import pytest
from scipy.optimize import least_squares

import spectrochempy as scp
from spectrochempy.analysis.curvefitting._modelspec import _ComponentSpec
from spectrochempy.analysis.curvefitting._modelspec import _FitModelSpec
from spectrochempy.analysis.curvefitting._modelspec import _ParamSpec
from spectrochempy.analysis.curvefitting._parameter_space import _prepared_parameters
from spectrochempy.analysis.curvefitting._parameter_space import _restore_physical
from spectrochempy.analysis.curvefitting._parameter_transform import (
    _external_transform_state,
)
from spectrochempy.analysis.curvefitting._parameter_transform import _to_external
from spectrochempy.analysis.curvefitting._parameters import FitParameters
from spectrochempy.analysis.curvefitting._stationarity import _physical_stationarity
from spectrochempy.analysis.curvefitting.optimize import _optimize


@pytest.mark.parametrize("native", [False, True])
def test_false_trf_stop_at_lower_bound_is_not_physically_stationary(native):
    x = np.linspace(0, 1, 31)

    def physical(values):
        return (values[0] - 2) * x

    if native:
        solution = least_squares(physical, [0.0], bounds=([0.0], [np.inf]))
        value, derivative = solution.x[0], 1.0
    else:
        solution = least_squares(
            lambda u: physical([_to_external(u[0], 0, None)]), [0.0]
        )
        value = _to_external(solution.x[0], 0, None)
        derivative, _ = _external_transform_state(solution.x[0], 0, None)
    assert solution.success
    assert solution.cost > 20
    diagnostics = _physical_stationarity(
        physical,
        [value],
        [(0, np.inf)],
        solution.fun,
        solution.jac,
        [derivative],
        reference=2 * x,
    )
    assert diagnostics["physical_stationarity_status"] == "not_satisfied"
    assert diagnostics["physical_projected_gradient"] == pytest.approx(1)
    assert diagnostics["stationarity_evaluations"] == 3


@pytest.mark.parametrize(
    "bound,target,point",
    [((0, np.inf), -1, 0), ((-np.inf, 4), 5, 4), ((0, 4), -1, 0), ((0, 4), 5, 4)],
)
def test_true_boundary_optimum_projects_the_outward_gradient(bound, target, point):
    x = np.linspace(0, 1, 31)

    def residual(values):
        return (values[0] - target) * x

    diagnostics = _physical_stationarity(
        residual, [point], [bound], residual([point]), np.zeros((x.size, 1)), [0]
    )
    assert diagnostics["physical_stationarity_status"] == "satisfied"
    assert diagnostics["physical_projected_gradient"] == 0
    assert diagnostics["stationarity_evaluations"] == 3


@pytest.mark.parametrize("parameter_scale", [1e-150, 1, 1e150])
@pytest.mark.parametrize("intensity_scale", [1e-150, 1, 1e150])
def test_physical_probe_metric_is_invariant_to_parameter_and_intensity_units(
    parameter_scale, intensity_scale
):
    x = np.linspace(0, 1, 31)

    def residual(values):
        return (values[0] / parameter_scale - 2) * x * intensity_scale

    diagnostics = _physical_stationarity(
        residual,
        [0.0],
        [(0, 4 * parameter_scale)],
        residual([0]),
        np.zeros((x.size, 1)),
        [0],
    )
    assert diagnostics["physical_stationarity_status"] == "not_satisfied"
    assert diagnostics["physical_projected_gradient"] == pytest.approx(1)


def _line(x, slope=0.0):
    return slope * x


def test_common_false_stop_is_qualified_without_changing_the_solution():
    x = np.linspace(0, 1, 31)
    fit = scp.Optimize(usermodels={"myline": _line}, script="", amplitude_mode="area")
    fit.script = """
COMMON:
$ slope: 0, 0, none
MODEL: line
shape: myline
> slope: slope
"""
    fit.fit(scp.NDDataset(2 * x, coordset=[scp.Coord(x)]))
    diagnostics = fit.result.diagnostics
    assert diagnostics["success"]
    assert diagnostics["physical_stationarity_status"] == "not_satisfied"
    assert diagnostics["stationarity_evaluations"] == 3
    assert fit.fp["slope"] == 0
    assert fit.result.rss == pytest.approx(4 * np.dot(x, x))
    np.testing.assert_array_equal(fit.predict().data, 0)
    assert fit.jacobian is not None
    assert fit.result.covariance is None


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("descending", [False, True])
def test_interior_public_fit_uses_chain_rule_and_preserves_configuration(
    masked, descending
):
    x = np.linspace(0.3, 1.2, 31)
    if descending:
        x = x[::-1]
    y = 2 * x + 0.3 + 0.01 * np.sin(np.arange(x.size))
    data = scp.NDDataset(y, coordset=[scp.Coord(x)])
    if masked:
        data.data[5] = 1e100
        data[5] = scp.MASKED
    model = scp.Linear()
    params = model.make_params(slope=1.0, intercept=0.1)
    params["slope"].set(min=0, max=4)
    fit = scp.Optimize(model=model, params=params).fit(data)
    diagnostics = fit.result.diagnostics
    assert diagnostics["physical_stationarity_status"] == "satisfied"
    assert diagnostics["physical_stationarity_reason"] is None
    assert diagnostics["physical_projected_gradient"] < 1e-6
    assert diagnostics["stationarity_evaluations"] == 0
    assert fit.initial_params["slope"].value == params["slope"].value == 1
    assert fit.result.covariance is not None

    def background(x, slope, intercept):
        return slope * x + intercept

    common = scp.Optimize(
        usermodels={"background": background}, script="", amplitude_mode="area"
    )
    common.script = """
COMMON:
$ intercept: 0.1, none, none
$ slope: 1, 0, 4
MODEL: line
shape: background
> slope: slope
> intercept: intercept
"""
    common.fit(data)
    assert (
        common.result.diagnostics["physical_stationarity_status"]
        == diagnostics["physical_stationarity_status"]
    )
    assert common.result.diagnostics["physical_projected_gradient"] == pytest.approx(
        diagnostics["physical_projected_gradient"], abs=1e-10
    )


@pytest.mark.parametrize("mode", ["dry", "fixed", "simplex"])
def test_no_invented_diagnostic_without_a_least_squares_jacobian(mode):
    model = scp.Linear()
    params = model.make_params(slope=1)
    if mode == "fixed":
        for parameter in params.values():
            parameter.set(vary=False)
    options = {
        "dry": mode == "dry",
        "method": "simplex" if mode == "simplex" else "least_squares",
    }
    x = np.linspace(0, 1, 11)
    fit = scp.Optimize(model=model, params=params, **options).fit(
        scp.NDDataset(x, coordset=[scp.Coord(x)])
    )
    assert fit.result.diagnostics["physical_stationarity_status"] in (
        "not_checked",
        "unavailable",
    )
    assert fit.result.diagnostics["physical_projected_gradient"] is None
    assert fit.result.diagnostics["stationarity_evaluations"] == 0


def test_failed_feasible_probe_is_unavailable_without_changing_the_fit():
    def narrow_domain(x, slope=0.0):
        if slope > 1e-8:
            raise RuntimeError("Outside the user model domain")
        return slope * x

    x = np.linspace(0, 1, 31)
    fit = scp.Optimize(
        usermodels={"narrow": narrow_domain}, script="", amplitude_mode="area"
    )
    fit.script = """
MODEL: line
shape: narrow
$ slope: 0, 0, none
"""
    fit.fit(scp.NDDataset(2 * x, coordset=[scp.Coord(x)]))
    assert fit.result.diagnostics["success"]
    assert fit.result.diagnostics["physical_stationarity_status"] == "unavailable"
    assert (
        "Outside the user model domain"
        in fit.result.diagnostics["physical_stationarity_reason"]
    )
    assert fit.fp["slope_line"] == 0


@pytest.mark.parametrize("legacy", [False, True])
def test_physical_assignment_preserves_named_fields_and_experiments(legacy):
    spec = _FitModelSpec(
        common_params={
            "a": _ParamSpec("a", [2.0, 4.0], bounds=(0, 10)),
            "experiments": _ParamSpec("experiments", 3.0),
        },
        components=[
            _ComponentSpec(
                label="experiments",
                model_name="unused",
                params={"b": _ParamSpec("b", 1.0)},
            )
        ],
        expvars=["a"],
        expnumber=2,
    )
    definition = spec.to_fitparameters() if legacy else spec
    prepared = [p for p in _prepared_parameters(definition) if not p.fixed]
    values = {p.name: p.value for p in prepared}
    values.update(a_exp0=1e-20, a_exp1=3.0, experiments=4.0, b_experiments=5.0)
    _restore_physical(definition, list(values.values()), list(values))
    actual = {p.name: p.value for p in _prepared_parameters(definition) if not p.fixed}
    assert actual == values


@pytest.mark.parametrize("power", [2, 3])
@pytest.mark.parametrize("factor", [1.0, 1e12])
def test_flat_boundary_directions_are_unavailable_when_derivatives_are_unresolved(
    power, factor
):
    def residual(values):
        return np.array([-1 + factor * values[0] ** power])

    diagnostics = _physical_stationarity(
        residual, [0.0], [(0, np.inf)], residual([0]), np.zeros((1, 1)), [0]
    )
    assert diagnostics["physical_stationarity_status"] == "unavailable"
    assert (
        "Unresolved physical derivative" in diagnostics["physical_stationarity_reason"]
    )


def test_exact_fit_roundoff_is_not_mistaken_for_a_nonstationary_solution():
    x = np.arange(11, dtype=float)
    model = scp.Linear()
    fit = scp.Optimize(model=model, params=model.make_params(slope=1)).fit(
        scp.NDDataset(2 * x + 0.3, coordset=[scp.Coord(x)])
    )
    assert fit.result.diagnostics["physical_stationarity_status"] == "satisfied"
    assert (
        fit.result.diagnostics["physical_stationarity_reason"] == "residual_at_roundoff"
    )


def test_signed_chain_rule_matches_an_independent_correlated_matrix_reference():
    design = np.array([[1, 2, -1], [2, -1, 1], [0.5, 0, 1], [-1, 2, 2]])
    derivatives = np.array([-2, 1e100, 1e-100])
    residual = np.array([1, -2, 0.5, 1])
    expected = np.max(
        np.abs(design.T @ residual)
        / np.linalg.norm(design, axis=0)
        / np.linalg.norm(residual)
    )
    diagnostics = _physical_stationarity(
        lambda _: pytest.fail("Regular coordinates require no model probes"),
        [1, 2, 3],
        [(-np.inf, np.inf)] * 3,
        residual,
        design * derivatives,
        derivatives,
    )
    assert diagnostics["physical_projected_gradient"] == pytest.approx(expected)
    assert diagnostics["stationarity_evaluations"] == 0


def test_legacy_single_experiment_probes_preserve_scalar_callback_contract():
    definition = FitParameters()
    definition["a"] = ([0.0], 0, None, False)
    definition.expvars = ["a"]
    definition.expnumber = 1
    x = np.linspace(0, 1, 31)

    def residual(parameters, coordinate):
        assert np.isscalar(parameters["a"])
        return (float(parameters["a"]) - 2) * coordinate

    fitted, _, diagnostics, _ = _optimize(residual, definition, args=(x,))
    assert np.isscalar(fitted["a"]) and fitted["a"] == 0
    assert diagnostics["physical_stationarity_status"] == "not_satisfied"
    assert diagnostics["stationarity_evaluations"] == 3


def test_physical_probes_resolve_parameter_links_on_independent_copies():
    x = np.linspace(0, 1, 31)
    fit = scp.Optimize(usermodels={"myline": _line}, script="", amplitude_mode="area")
    fit.script = """
COMMON:
$ a: 0, 0, none
> twice: 2*a
MODEL: line
shape: myline
> slope: twice
"""
    fit.fit(scp.NDDataset(2 * x, coordset=[scp.Coord(x)]))
    assert fit.result.diagnostics["physical_stationarity_status"] == "not_satisfied"
    assert fit.result.diagnostics["physical_projected_gradient"] == pytest.approx(1)
    assert fit.result.diagnostics["stationarity_evaluations"] == 3
    assert fit.fp["a"] == 0
    np.testing.assert_array_equal(fit.predict().data, 0)


@pytest.mark.parametrize("point,upper", [(0.01, 1e6), (1e-9, np.inf)])
def test_near_boundary_interior_points_retain_feasible_descent(point, upper):
    diagnostics = _physical_stationarity(
        lambda values: np.array([values[0] + 1]),
        [point],
        [(0, upper)],
        [point + 1],
        [[1]],
        [1],
    )
    assert diagnostics["physical_stationarity_status"] == "not_satisfied"
    assert diagnostics["physical_projected_gradient"] == pytest.approx(1)


@pytest.mark.parametrize("false_stationary", [False, True])
def test_compensated_gradient_is_unavailable_when_probe_error_exceeds_score_tolerance(
    false_stationary,
):
    def residual(values):
        p = values[0]
        return np.array(
            [
                1 + (1 - 1e-5) * p - 1e6 * p**3
                if false_stationary
                else 1 + p + 1e6 * p**3,
                -1 + p,
            ]
        )

    diagnostics = _physical_stationarity(
        residual, [0], [(0, np.inf)], residual([0]), np.zeros((2, 1)), [0]
    )
    assert diagnostics["physical_stationarity_status"] == "unavailable"
    assert "stationarity tolerance" in diagnostics["physical_stationarity_reason"]


def test_common_cancellation_does_not_invent_a_stationarity_failure():
    def cancellation(x, p=0.0):
        return -np.where(x == 0, 1 + p + 1e6 * p**3, -1 + p)

    fit = scp.Optimize(
        usermodels={"cancelmodel": cancellation}, script="", amplitude_mode="area"
    )
    fit.script = """
MODEL: line
shape: cancelmodel
$ p: 0, 0, none
"""
    fit.fit(scp.NDDataset([0.0, 0.0], coordset=[scp.Coord([0.0, 1.0])]))
    assert fit.result.diagnostics["success"]
    assert fit.fp["p_line"] == 0
    assert fit.result.diagnostics["physical_stationarity_status"] == "unavailable"


def test_uncertainty_interval_crossing_the_threshold_is_unavailable():
    def residual(values):
        p = values[0]
        return np.array([1 + p, -1 + (1 + 2e-6) * p])

    diagnostics = _physical_stationarity(
        residual, [0], [(0, np.inf)], residual([0]), np.zeros((2, 1)), [0]
    )
    assert diagnostics["physical_stationarity_status"] == "unavailable"
    assert "threshold unresolved" in diagnostics["physical_stationarity_reason"]
