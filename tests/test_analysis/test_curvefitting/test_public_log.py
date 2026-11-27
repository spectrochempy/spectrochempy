"""User workflows for explicit positive-reference log coordinates in Python 1D."""

import pickle

import numpy as np
import pytest

import spectrochempy as scp


def problem(scale=1):
    def line(x, a=1.0, b=0.0, c=0.0):
        return scale * a * x + b + c

    model = scp.Model(line)
    params = model.make_params(
        a=scp.Parameter(
            0.3 / scale,
            min=0.001 / scale,
            max=10 / scale,
            scale="log",
            log_reference=1 / scale,
        ),
        b=0.1,
        c=scp.Parameter(expr="b"),
    )
    x = np.linspace(-1, 1, 51)[::-1]
    y = 2 * x + 0.6 + 0.01 * np.cos(7 * x)
    data = scp.NDDataset(y, coordset=[scp.Coord(x, rounding=False)])
    data[[3, 8, 17]] = scp.MASKED
    return model, params, data, x, y


@pytest.mark.parametrize("scale", [1e-6, 1, 1e6])
def test_public_log_fit_covariance_copies_and_repeated_fit(scale):
    model, params, data, x, y = problem(scale)
    first = scp.Optimize(model=model, params=params, bounds_mode="native")
    second = scp.Optimize(model=model, params=params, bounds_mode="native")
    params["a"].set(value=4 / scale, log_reference=2 / scale)
    first.fit(data)
    result = first.result
    valid = ~data.mask
    design = np.column_stack([scale * x[valid], np.full(valid.sum(), 2.0)])
    expected = np.linalg.lstsq(design, y[valid], rcond=None)[0]
    covariance = np.linalg.inv(design.T @ design) * result.rss / (valid.sum() - 2)
    assert result.diagnostics["success"]
    assert result.diagnostics["physical_stationarity_status"] == "satisfied"
    assert result.diagnostics["jacobian_coordinates"] == "solver"
    assert result.parameter_names == ("a", "b")
    np.testing.assert_allclose(
        [result.fitted_parameters[k] for k in ("a", "b")], expected, rtol=1e-7
    )
    np.testing.assert_allclose(result.covariance, covariance, rtol=2e-6, atol=0)
    np.testing.assert_allclose(result.stderr, np.sqrt(np.diag(covariance)), rtol=2e-6)
    np.testing.assert_allclose(
        first.jacobian / [expected[0], 1], -design, rtol=2e-6, atol=1e-9
    )
    assert result.fitted_parameters["c"] == result.fitted_parameters["b"]
    assert result.diagnostics["derived_uncertainty_status"] == "not_propagated"
    assert result.parameters["parameter_coordinates"]["a"] == {
        "scale": "log",
        "log_reference": 1 / scale,
    }
    assert first.initial_params["a"].value == 0.3 / scale
    assert first.initial_params["a"].log_reference == 1 / scale
    detached = first.initial_params
    detached["a"].set(scale="lin")
    assert first.initial_params["a"].scale == "log"
    second.fit(data)
    np.testing.assert_array_equal(first.result.covariance, result.covariance)
    np.testing.assert_allclose(second.result.fitted.data, result.fitted.data)
    first.fit(data)
    np.testing.assert_allclose(first.result.fitted.data, result.fitted.data)
    assert first.initial_params["a"].log_reference == 1 / scale


@pytest.mark.parametrize(
    "updates",
    [
        {"log_reference": None},
        {"log_reference": 0},
        {"log_reference": -1},
        {"log_reference": np.inf},
        {"log_reference": True},
        {"min": 0},
        {"min": None},
        {"value": 0},
        {"scale": "sqrt"},
    ],
)
def test_invalid_log_updates_are_atomic(updates):
    parameter = scp.Parameter(1, min=0.1, max=10, scale="log", log_reference=1)
    before = parameter.__dict__.copy()
    with pytest.raises(ValueError):
        parameter.set(**updates)
    assert parameter.__dict__ == before


def test_link_and_linear_transitions_clear_independent_coordinates():
    parameter = scp.Parameter(1, min=0.1, scale="log", log_reference=1)
    parameter.set(expr="other")
    assert not parameter.vary and parameter.min is None
    assert parameter.scale == "lin" and parameter.log_reference is None
    parameter.set(expr=None, value=2, min=0.1, scale="log", log_reference=2)
    assert parameter.vary and parameter.scale == "log"
    parameter.set(scale="lin")
    assert parameter.log_reference is None
    with pytest.raises(ValueError, match="independent coordinates"):
        parameter.set(expr="other", scale="log", log_reference=1)
    with pytest.raises(ValueError, match="requires scale"):
        parameter.set(log_reference=1)


@pytest.mark.parametrize(
    "options", [{}, {"method": "simplex", "bounds_mode": "native"}]
)
def test_incompatible_solver_options_fail_on_construction(options):
    model, params, *_ = problem()
    with pytest.raises(ValueError, match="native|least_squares"):
        scp.Optimize(model=model, params=params, **options)


@pytest.mark.parametrize(
    "change", [{"bounds_mode": "transform"}, {"method": "simplex"}]
)
def test_options_are_revalidated_before_model_evaluation(change):
    model, params, data, *_ = problem()
    opt = scp.Optimize(model=model, params=params, bounds_mode="native")
    for name, value in change.items():
        setattr(opt, name, value)
    with pytest.raises(ValueError, match="native|least_squares"):
        opt.fit(data)


def test_fixed_log_and_auxiliary_log_are_copied_and_have_no_extra_coordinates():
    model = scp.Linear()
    params = model.make_params(slope=1, intercept=scp.Parameter(0, vary=False))
    params.add("source", value=1, min=0.1, max=10, scale="log", log_reference=1)
    params["slope"].set(expr="source * 2")
    restored = pickle.loads(pickle.dumps(params))  # noqa: S301
    assert restored["source"].scale == "log"
    x = np.linspace(0, 1, 31)
    data = scp.NDDataset(4 * x, coordset=[scp.Coord(x, rounding=False)])
    opt = scp.Optimize(
        model=model, params=restored, bounds_mode="native", max_fun_calls=200
    ).fit(data)
    assert opt.result.n_varying_parameters == 1
    assert opt.result.fitted_parameters["source"] == pytest.approx(2)
    restored["source"].set(value=2, vary=False)
    fixed = scp.Optimize(model=model, params=restored, bounds_mode="native").fit(data)
    assert not fixed.result.diagnostics["optimization_performed"]
    assert fixed.result.covariance is None


def test_legacy_parameter_state_defaults_to_linear():
    parameter = scp.Parameter(1)
    del parameter.scale
    del parameter.log_reference
    owned = scp.Parameters()
    owned["a"] = parameter
    assert owned["a"].scale == "lin" and owned["a"].log_reference is None
    model = scp.Linear()
    params = model.make_params(slope=parameter)
    opt = scp.Optimize(model=model, params=params)
    for p in opt._python_model_configuration[1].values():
        del p.scale
        del p.log_reference
    x = np.linspace(0, 1, 11)
    result = opt.fit(scp.NDDataset(2 * x + 0.3, coordset=[scp.Coord(x)])).result
    assert result.fitted_parameters["slope"] == pytest.approx(2)
    assert "parameter_coordinates" not in result.parameters


def test_two_linked_peaks_and_background_match_linear_native_fit():
    x = np.linspace(-4, 4, 129)
    y = (
        scp.gaussian(x, ampl=2, pos=-1, width=0.6)
        + scp.gaussian(x, ampl=1.2, pos=1.2, width=0.6)
        + 0.03
        + 0.002 * x
        + 0.0001 * np.cos(5 * x)
    )
    data = scp.NDDataset(y, coordset=[scp.Coord(x, rounding=False)])
    data[[2, 48, 100]] = scp.MASKED
    model = (
        scp.Gaussian(prefix="p1_")
        + scp.Gaussian(prefix="p2_")
        + scp.Linear(prefix="bg_")
    )
    params = model.make_params(
        p1_center=-1,
        p2_center=1.2,
        p1_width=0.7,
        p2_amplitude=0.5,
        bg_slope=0.001,
        bg_intercept=0.02,
    )
    for name in ("p1_center", "p2_center"):
        params[name].set(vary=False)
    for name in ("p1_amplitude", "p2_amplitude"):
        params[name].set(min=0.001, max=5, scale="log", log_reference=1)
    params["p1_width"].set(min=0.1, max=2, scale="log", log_reference=1)
    params["p2_width"].set(expr="p1_width")
    linear = params.copy()
    for parameter in linear.values():
        parameter.set(scale="lin")
    fit = scp.Optimize(model=model, params=params, bounds_mode="native").fit(data)
    reference = scp.Optimize(model=model, params=linear, bounds_mode="native").fit(data)
    assert fit.result.n_varying_parameters == 5
    assert (
        fit.result.fitted_parameters["p1_width"]
        == fit.result.fitted_parameters["p2_width"]
    )
    np.testing.assert_allclose(
        fit.result.fitted.data, reference.result.fitted.data, atol=1e-7
    )
    np.testing.assert_array_equal(fit.result.fitted.mask, reference.result.fitted.mask)
    np.testing.assert_allclose(
        fit.result.covariance, reference.result.covariance, rtol=1e-4, atol=1e-10
    )
    np.testing.assert_array_equal(data.mask, fit.result.fitted.mask.squeeze())
