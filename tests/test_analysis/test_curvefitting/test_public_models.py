"""User workflows for the composable Python facade and canonical fitting parity."""

import numpy as np
import pytest

import spectrochempy as scp


def dataset(x, values, mask=None):
    return scp.NDDataset(values, coordset=[scp.Coord(x)], mask=mask)


def test_personal_function_initial_configuration_and_independent_estimators():
    def line(x, slope_value, offset=0.0):
        return slope_value * x + offset

    model = scp.Model(line, prefix="custom_")
    params = model.make_params(custom_slope_value=1.0)
    params["custom_slope_value"].set(min=0, max=5)
    first = scp.Optimize(model=model, params=params, method="least_squares")
    second = scp.Optimize(model=model, params=params, method="least_squares")
    params["custom_slope_value"].set(value=4)
    detached = first.initial_params
    detached["custom_slope_value"].set(value=3)
    x = np.linspace(-1, 1, 51)
    first.fit(dataset(x, 2 * x + 0.3))
    second.fit(dataset(x, 3 * x - 0.2))
    assert first.result.fitted_parameters["custom_slope_value"] == pytest.approx(2)
    assert second.result.fitted_parameters["custom_slope_value"] == pytest.approx(3)
    assert first.initial_params["custom_slope_value"].value == 1
    assert second.initial_params["custom_slope_value"].value == 1
    assert params["custom_slope_value"].value == 4
    assert callable(first.params)
    assert first.result.parameters["method"] == "least_squares"
    assert "custom_slope_value" in first.report()
    assert first.result.stderr.shape == (2,)
    assert len(first.result.parameter_names) == 2
    assert first.result.diagnostics["derived_uncertainty_status"] == "not_propagated"
    _ = first.plot()
    first.fit(dataset(x, 2.5 * x + 0.1))
    assert first.result.fitted_parameters["custom_slope_value"] == pytest.approx(2.5)
    assert first.initial_params["custom_slope_value"].value == 1


def baseline(x, slope, intercept):
    return slope * x + intercept


SCRIPT = """
COMMON:
 $ a: 0.02, -1, 1
 $ b: 0.001, -1, 1
 $ c: 1, 0, 5
 * d: -1
 $ e: 0.7, 0.1, 2
 $ f: 0.5, 0, 5
 * g: 1.2
MODEL: left
shape: gaussianmodel
 > ampl: c
 > pos: d
 > width: e
MODEL: right
shape: gaussianmodel
 > ampl: f
 > pos: g
 > width: e
MODEL: background
shape: mybaseline
 > slope: b
 > intercept: a
"""


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_two_linked_peaks_and_linear_background_match_common(descending, masked):
    x = np.linspace(-4, 4, 129)
    if descending:
        x = x[::-1]
    y = (
        scp.gaussian(x, ampl=2, pos=-1, width=0.6)
        + scp.gaussian(x, ampl=1.2, pos=1.2, width=0.6)
        + baseline(x, 0.002, 0.03)
    )
    mask = np.zeros(x.size, dtype=bool)
    if masked:
        mask[[2, 48, 100]] = True
    data = dataset(x, y, mask)
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
    params["p1_center"].set(vary=False)
    params["p2_center"].set(vary=False)
    params["p1_amplitude"].set(min=0, max=5)
    params["p2_amplitude"].set(min=0, max=5)
    params["p1_width"].set(min=0.1, max=2)
    params["p2_width"].set(expr="p1_width")
    for name in ("bg_slope", "bg_intercept"):
        params[name].set(min=-1, max=1)
    python = scp.Optimize(model=model, params=params, method="least_squares").fit(data)
    script = scp.Optimize(
        usermodels={"mybaseline": baseline},
        script="",
        amplitude_mode="area",
        method="least_squares",
    )
    script.script = SCRIPT
    script.fit(data)
    for name in ("fitted", "components", "residuals"):
        actual, expected = getattr(python.result, name), getattr(script.result, name)
        np.testing.assert_array_equal(actual.data, expected.data)
        np.testing.assert_array_equal(actual.mask, expected.mask)
    np.testing.assert_array_equal(python.jacobian, script.jacobian)
    np.testing.assert_array_equal(python.result.covariance, script.result.covariance)
    assert python.result.n_varying_parameters == 5
    assert python.result.degrees_of_freedom == int((~mask).sum()) - 5
    values = python.result.fitted_parameters
    assert values["p1_width"] == values["p2_width"]
    assert values["p1_width"] == pytest.approx(0.6, abs=1e-4)
    assert python.result.parameter_names == (
        "bg_intercept",
        "bg_slope",
        "p1_amplitude",
        "p1_width",
        "p2_amplitude",
    )
    assert params["p1_width"].value == 0.7


def test_parameter_identity_never_implies_sharing_and_prefix_collisions_fail():
    model = scp.Gaussian(prefix="one_") + scp.Gaussian(prefix="two_")
    shared = scp.Parameter(value=2)
    params = model.make_params(one_amplitude=shared, two_amplitude=shared)
    params["one_amplitude"].set(value=3)
    assert params["two_amplitude"].value == 2
    assert shared.value == 2
    with pytest.raises(ValueError, match="prefix"):
        _ = scp.Gaussian() + scp.Gaussian()
    with pytest.raises(ValueError, match="prefix"):
        _ = model + scp.Linear(prefix="one_")


def test_forward_expression_chain_auxiliary_parameters_and_negative_power():
    model = scp.Linear()
    params = model.make_params()
    params.add("source", value=-2, vary=False)
    params.add("square", expr="source**2")
    params["slope"].set(expr="square + 1")
    params["intercept"].set(expr="-square / 2")
    x = np.linspace(0, 1, 21)
    opt = scp.Optimize(model=model, params=params, dry=True).fit(dataset(x, 5 * x - 2))
    np.testing.assert_allclose(opt.result.fitted.data.squeeze(), 5 * x - 2)
    assert opt.result.fitted_parameters["square"] == 4
    assert opt.result.n_varying_parameters == 0
    assert opt.result.stderr is None


@pytest.mark.parametrize(
    "expression",
    [
        "missing",
        "slope",
        "intercept+1",
        "np.exp(1)",
        "__import__('os')",
        "slope[0]",
        "True",
        "1/0",
        "1e999",
    ],
)
def test_invalid_expressions_are_rejected_before_fitting(expression):
    model = scp.Linear()
    params = model.make_params()
    params["slope"].set(expr=expression)
    if expression == "intercept+1":
        params["intercept"].set(expr="slope")
    with pytest.raises(ValueError):
        scp.Optimize(model=model, params=params)


@pytest.mark.parametrize(
    "options",
    [
        {"script": ""},
        {"peaks": []},
        {"shared_parameters": {}},
        {"parameters": {}},
        {"warm_start": True},
        {"usermodels": {}},
        {"constraints": {}},
        {"linear_amplitudes": True},
        {"solver_options": {}},
        {"autobase": True},
        {"autoampl": True},
        {"amplitude_mode": "height"},
    ],
)
def test_incompatible_options_are_rejected(options):
    with pytest.raises(ValueError):
        scp.Optimize(model=scp.Linear(), **options)


def test_invalid_parameter_configuration_and_capabilities():
    model = scp.Linear()
    params = model.make_params()
    with pytest.raises(ValueError):
        params["slope"].set(value=np.nan)
    assert params["slope"].value == 0
    for options in (
        {"min": 2, "max": 1},
        {"value": 3, "max": 2},
        {"expr": "intercept", "vary": True},
        {"vary": "yes"},
    ):
        with pytest.raises(ValueError):
            params["slope"].set(**options)
    with pytest.raises(NotImplementedError):
        params["slope"].set(scale="log")
    with pytest.raises(NotImplementedError):
        params.add("aux", scale="log")
    with pytest.raises(ValueError, match="Unused"):
        params.add("aux")
        scp.Optimize(model=model, params=params)
    with pytest.raises(ValueError, match="params requires"):
        scp.Optimize(params=params)
    with pytest.raises(TypeError, match="Parameters"):
        scp.Optimize(model=model, params={})
    with pytest.raises(ValueError, match="initial"):
        scp.Model(lambda x, a: a * x).make_params()
    with pytest.raises(ValueError):
        scp.Optimize(model=scp.Gaussian(), params=scp.Gaussian().make_params(width=-1))


def test_script_replacement_is_explicit_and_callback_output_is_checked():
    opt = scp.Optimize(model=scp.Linear())
    with pytest.raises(ValueError, match="another Optimize"):
        opt.script = SCRIPT
    broken = scp.Model(lambda x, a=1: 1.0)
    with pytest.raises(ValueError, match="vector"):
        scp.Optimize(model=broken).fit(dataset(np.arange(10.0), np.arange(10.0)))


def test_legacy_grid_callback_still_uses_parameters():
    x = np.linspace(0, 1, 31)

    def line(template, slope):
        return slope * np.broadcast_to(template.x.data, template.shape)

    source = scp.NDDataset(
        np.broadcast_to(2 * x, (3, 31)),
        coordset=[scp.Coord(np.arange(3.0)), scp.Coord(x)],
    )
    opt = scp.Optimize(model=line, parameters={"slope": 1.0}).fit(source)
    assert opt.result.fitted_parameters["slope"] == pytest.approx(2)
    assert opt.initial_params is None


@pytest.mark.parametrize("options", [{"min": 0}, {"max": 0}, {"min": 0, "max": 3}])
def test_free_start_on_a_bound_is_rejected_instead_of_false_convergence(options):
    with pytest.raises(ValueError, match="strictly inside"):
        scp.Parameter(0, **options)
    fixed = scp.Parameter(0, vary=False, **options)
    assert fixed.value == 0
    with pytest.raises(ValueError, match="strictly inside"):
        fixed.set(vary=True)
    assert not fixed.vary


def test_bound_configuration_is_revalidated_and_interior_start_recovers_solution():
    model = scp.Linear()
    params = model.make_params(slope=0.1)
    params["slope"].set(min=0)
    params["intercept"].set(vary=False)
    x = np.linspace(0, 1, 21)
    opt = scp.Optimize(model=model, params=params, method="least_squares")
    opt.fit(dataset(x, 2 * x))
    assert opt.result.fitted_parameters["slope"] == pytest.approx(2)
    params["slope"].value = 0
    with pytest.raises(ValueError, match="strictly inside"):
        scp.Optimize(model=model, params=params)
