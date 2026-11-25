"""Qualify opt-in native 1D bounds against physical reference problems."""

from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_allclose

import spectrochempy as scp
from spectrochempy.analysis.curvefitting._modelspec import _FitModelSpec
from spectrochempy.analysis.curvefitting._modelspec import _ParamSpec
from spectrochempy.analysis.curvefitting._parameter_space import _physical_vector


def line(x, slope=1.0, intercept=0.0):
    return slope * x + intercept


def fit_line(x, y, initial, bounds, frontend, **options):
    source = scp.NDDataset(y, coordset=[scp.Coord(x, rounding=False)])
    if frontend == "python":
        model = scp.Linear()
        params = model.make_params(
            slope=scp.Parameter(initial, min=bounds[0], max=bounds[1]),
            intercept=scp.Parameter(0, vary=False),
        )
        fit = scp.Optimize(model=model, params=params, **options)
    else:
        lo, hi = ("none" if b is None else b for b in bounds)
        script = f"""COMMON:
$ slope: {initial}, {lo}, {hi}
* intercept: 0
MODEL: peak
shape: customline
> slope: slope
> intercept: intercept
"""
        fit = scp.Optimize(
            script="",
            usermodels={"customline": line},
            amplitude_mode="area",
            **options,
        )
        fit.script = script
    return fit.fit(source)


@pytest.mark.parametrize("frontend", ["python", "common"])
@pytest.mark.parametrize(
    "initial,bounds", [(0, (0, None)), (4, (None, 4)), (0, (0, 4)), (4, (0, 4))]
)
def test_endpoint_start_recovers_interior_optimum(frontend, initial, bounds):
    x = np.linspace(0, 1, 31)
    noise = 0.01 * np.sin(np.arange(x.size))
    noise -= x * (x @ noise) / (x @ x)
    fit = fit_line(x, 2 * x + noise, initial, bounds, frontend, bounds_mode="native")
    result = fit.result
    assert_allclose(result.fitted.data.squeeze(), 2 * x, atol=1e-8)
    assert result.diagnostics["success"]
    assert result.diagnostics["physical_stationarity_status"] == "satisfied"
    assert result.diagnostics["termination_policy"] == "step"
    assert result.parameters["bounds_mode"] == "native"
    assert result.diagnostics["jacobian_coordinates"] == "physical"
    assert_allclose(fit.jacobian[:, 0], -x, atol=1e-9)
    if frontend == "python":
        assert fit.initial_params["slope"].value == initial


@pytest.mark.parametrize("frontend", ["python", "common"])
@pytest.mark.parametrize(
    "bounds,initial,target,expected",
    [
        ((0, None), 0, -1, 0),
        ((None, 4), 4, 5, 4),
        ((0, 4), 2, -1, 0),
        ((0, 4), 2, 5, 4),
    ],
)
def test_true_bound_optimum_excludes_unconstrained_covariance(
    frontend, bounds, initial, target, expected
):
    x = np.linspace(0, 1, 31)
    result = fit_line(
        x, target * x, initial, bounds, frontend, bounds_mode="native"
    ).result
    assert_allclose(result.fitted.data.squeeze(), expected * x, atol=2e-9)
    assert result.diagnostics["success"]
    assert result.diagnostics["covariance_reason"] == "active_bounds"
    assert result.covariance is None
    assert result.stderr is None


@pytest.mark.parametrize("scale", [1e-4, 1, 1e4])
def test_correlated_covariance_links_masks_and_configuration_ownership(scale):
    x = np.linspace(0.1, 1.0, 41)[::-1]
    y = 2 * x + 0.3 + 0.01 * np.sin(np.arange(x.size))
    source = scp.NDDataset(y, coordset=[scp.Coord(x, rounding=False)])
    source[[2, 8, 15]] = scp.MASKED
    model = scp.Model(lambda x, a=0.0, b=0.0, c=0.0: scale * a * x + b + c)
    params = model.make_params(
        a=scp.Parameter(0, min=0), b=0.0, c=scp.Parameter(expr="b")
    )
    initial = deepcopy(params)
    first = scp.Optimize(model=model, params=params, bounds_mode="native")
    second = scp.Optimize(model=model, params=params, bounds_mode="native")
    first.fit(source)
    result = first.result
    valid = ~source.mask
    design = np.column_stack((scale * x[valid], np.full(valid.sum(), 2.0)))
    expected = np.linalg.lstsq(design, y[valid], rcond=None)[0]
    assert_allclose(
        [result.fitted_parameters[k] for k in ("a", "b")], expected, rtol=1e-6
    )
    assert result.fitted_parameters["c"] == result.fitted_parameters["b"]
    covariance = result.rss / (valid.sum() - 2) * np.linalg.inv(design.T @ design)
    assert_allclose(result.covariance, covariance, rtol=1e-5, atol=1e-14)
    assert result.parameter_names == ("a", "b")
    assert params["a"].value == initial["a"].value == 0
    assert second.initial_params["a"].value == 0
    assert_allclose(second.fit(source).result.fitted.data, first.result.fitted.data)


def test_legacy_false_stop_is_preserved_by_default_and_native_budget_is_honest():
    x = np.linspace(0, 1, 31)
    old = fit_line(x, 2 * x, 0, (0, None), "common")
    assert old.result.diagnostics["success"]
    assert old.result.diagnostics["physical_stationarity_status"] == "not_satisfied"
    limited = fit_line(
        x, 2 * x, 0, (0, None), "common", bounds_mode="native", max_fun_calls=1
    )
    assert not limited.result.diagnostics["success"]
    assert limited.result.diagnostics["solver_nfev"] == 1
    assert limited.result.covariance is None


@pytest.mark.parametrize("method", ["simplex", "basinhopping"])
def test_native_mode_rejects_unqualified_methods(method):
    with pytest.raises(ValueError, match="least_squares"):
        scp.Optimize(model=scp.Linear(), bounds_mode="native", method=method)


@pytest.mark.parametrize(
    "options",
    [
        {"peaks": []},
        {"model": lambda template, a: template.data, "parameters": {"a": 1}},
        {"model": "exponential_decay", "parameters": {"amplitude": 1, "tau": 1}},
    ],
)
def test_native_mode_does_not_change_nd_or_named_curve_routes(options):
    with pytest.raises(ValueError, match="only.*1D"):
        scp.Optimize(bounds_mode="native", **options)


def test_policy_is_revalidated_after_estimator_mutation():
    fit = scp.Optimize(model=scp.Linear(), bounds_mode="native")
    fit.method = "simplex"
    with pytest.raises(ValueError, match="least_squares"):
        fit.fit(scp.NDDataset(np.arange(5.0), coordset=[scp.Coord(np.arange(5.0))]))


@pytest.mark.parametrize(
    "value,bounds", [(2, (3, 4)), (2, (0, 1)), (0, (0, 0)), (np.nan, (0, 1))]
)
def test_physical_preparation_rejects_infeasible_or_degenerate_bounds(value, bounds):
    spec = _FitModelSpec(common_params={"a": _ParamSpec("a", value, bounds=bounds)})
    with pytest.raises(ValueError, match="Native bounds"):
        _physical_vector(spec)


def test_all_fixed_and_dry_preserve_initial_configuration():
    x = np.linspace(0, 1, 21)
    source = scp.NDDataset(2 * x, coordset=[scp.Coord(x, rounding=False)])
    model = scp.Linear()
    params = model.make_params(
        slope=scp.Parameter(0, min=0), intercept=scp.Parameter(0, vary=False)
    )
    dry = scp.Optimize(model=model, params=params, bounds_mode="native", dry=True).fit(
        source
    )
    assert dry.result.diagnostics["physical_stationarity_status"] == "not_checked"
    assert_allclose(dry.result.fitted.data, 0)
    params["slope"].set(value=2, vary=False)
    fixed = scp.Optimize(model=model, params=params, bounds_mode="native").fit(source)
    assert not fixed.result.diagnostics["optimization_performed"]
    assert fixed.result.covariance is None


def test_two_linked_gaussian_peaks_and_background_from_zero_amplitudes():
    x = np.linspace(-4, 4, 129)
    source = scp.NDDataset(
        scp.gaussian(x, ampl=30, pos=-1, width=0.6)
        + scp.gaussian(x, ampl=20, pos=1.2, width=0.6)
        + 0.03
        + 0.002 * x
        + 0.001 * np.sin(np.arange(x.size)),
        coordset=[scp.Coord(x, rounding=False)],
    )
    model = (
        scp.Gaussian(prefix="p1_")
        + scp.Gaussian(prefix="p2_")
        + scp.Linear(prefix="bg_")
    )
    params = model.make_params(p1_center=-1, p2_center=1.2, p1_width=0.7)
    params["p1_amplitude"].set(value=0, min=0)
    params["p2_amplitude"].set(value=0, min=0)
    params["p1_center"].set(vary=False)
    params["p2_center"].set(vary=False)
    params["p2_width"].set(expr="p1_width")
    fit = scp.Optimize(model=model, params=params, bounds_mode="native").fit(source)
    result = fit.result
    assert result.diagnostics["success"]
    assert result.fitted_parameters["p1_width"] == pytest.approx(0.6, abs=1e-4)
    assert result.fitted_parameters["p2_width"] == result.fitted_parameters["p1_width"]
    assert result.rss < 1e-4
    assert params["p1_amplitude"].value == 0
    assert result.diagnostics["derived_uncertainty_status"] == "not_propagated"


def test_native_rank_deficiency_does_not_invent_uncertainty():
    x = np.linspace(0, 1, 31)
    ds = scp.NDDataset(
        2 * x + 0.01 * np.sin(np.arange(x.size)),
        coordset=[scp.Coord(x, rounding=False)],
    )
    model = scp.Model(lambda x, a=0.0, b=0.0: (a + b) * x)
    params = model.make_params(a=scp.Parameter(0, min=0), b=scp.Parameter(0, min=0))
    result = (
        scp.Optimize(model=model, params=params, bounds_mode="native").fit(ds).result
    )
    assert result.covariance is None
    assert result.diagnostics["covariance_reason"] == "rank_deficient"


def test_gaussian_domain_stays_strictly_positive_and_constraints_are_rejected():
    model = scp.Gaussian()
    params = model.make_params(width=scp.Parameter(0, min=0))
    with pytest.raises(ValueError, match="strictly positive"):
        scp.Optimize(model=model, params=params, bounds_mode="native")
    with pytest.raises(ValueError, match="legacy constraints"):
        scp.Optimize(bounds_mode="native", constraints={"max_connections": 1})


@pytest.mark.parametrize("scale", [1e-10, 1e-6, 1e-4, 1, 1e4])
@pytest.mark.parametrize("frontend", ["python", "common"])
def test_small_signal_and_bound_interval_do_not_terminate_on_absolute_gradient(
    scale, frontend
):
    x = np.linspace(0, 1, 31)
    result = fit_line(
        x, 2 * scale * x, 0, (0, 4 * scale), frontend, bounds_mode="native"
    ).result
    assert result.diagnostics["success"]
    assert result.diagnostics["solver_nfev"] > 1
    assert result.diagnostics["solver_gtol"] is None
    assert_allclose(result.fitted.data.squeeze() / scale, 2 * x, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("experiments", [1, 2])
def test_native_legacy_sentinels_and_experiment_layout(experiments):
    from spectrochempy.analysis.curvefitting._parameter_space import _restore_native

    spec = _FitModelSpec(
        common_params={"a": _ParamSpec("a", [1.0] * experiments)},
        expvars=["a"],
        expnumber=experiments,
    )
    legacy = spec.to_fitparameters()
    values, keys, bounds = _physical_vector(legacy)
    assert np.isneginf(bounds[:, 0]).all()
    assert np.isposinf(bounds[:, 1]).all()
    restored = _restore_native(legacy, values + 1, keys)
    assert_allclose(restored.data["a"], [2.0] * experiments)
    assert np.isscalar(restored.data["a"]) == (experiments == 1)
