"""Qualify full-grid fitting against independent numerical problems."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal
from scipy.optimize import least_squares

import spectrochempy as scp
from spectrochempy.analysis._base._analysisbase import NotFittedError


def grid():
    x = scp.Coord.linspace(2.5, -2.5, 31, units="ppm", rounding=False)
    y = scp.Coord.linspace(-2, 2, 27, units="ppm", rounding=False)
    return scp.NDDataset(np.zeros((27, 31)), coordset=[y, x], units="V")


def test_nonseparable_reference_covariance_and_output_isolation():
    source = grid()
    x, y = source.x._data[None, :], source.y._data[:, None]

    def numeric(p):
        a, center, shear, width = p
        return a * np.exp(-0.5 * ((x - center - shear * y) / width) ** 2 - y**2 / 2)

    rng = np.random.default_rng(74)
    source.data = numeric([2.2, 0.17, 0.45, 0.8]) + rng.normal(0, 0.01, source.shape)
    source[4:7, 10:13] = scp.MASKED
    source.data[0, 0] = np.nan
    source.data[-1, -1] = np.inf
    support = ~source.mask & np.isfinite(source.data)
    original = source.copy()
    initial = [2, 0.1, 0.3, 1.0]
    low, high = [0, -1, -0.5, 0.2], [4, 1, 1, 2]

    def model(template, a, center, shear, width):
        assert np.all(template.data == 0)
        xx, yy = template.x.data[None, :], template.y.data[:, None]
        return a * np.exp(-0.5 * ((xx - center - shear * yy) / width) ** 2 - yy**2 / 2)

    specs = {
        name: {"value": value, "bounds": (lo, hi)}
        for name, value, lo, hi in zip(
            ["a", "center", "shear", "width"], initial, low, high, strict=True
        )
    }
    options = {"ftol": 1e-12, "xtol": 1e-12, "gtol": 1e-12}
    reference = least_squares(
        lambda p: (source.data - numeric(p))[support],
        initial,
        bounds=(low, high),
        **options,
    )
    fit = scp.Optimize(model=model, parameters=specs, solver_options=options)
    specs["a"]["value"] = 50
    assert fit.fit(source) is fit
    result = fit.result
    assert result.diagnostics["success"]
    assert_allclose(list(result.fitted_parameters.values()), reference.x, atol=1e-6)
    assert_allclose(result.diagnostics["rss"], reference.fun @ reference.fun, rtol=1e-8)
    assert result.parameter_names == ("a", "center", "shear", "width")
    assert fit.jacobian.shape == (support.sum(), 4)
    assert not fit.jacobian.flags.writeable
    assert_allclose(fit.jacobian, reference.jac, rtol=1e-4, atol=1e-5)
    expected_covariance = (
        reference.fun @ reference.fun / (support.sum() - 4)
    ) * np.linalg.inv(reference.jac.T @ reference.jac)
    assert_allclose(result.covariance, expected_covariance, rtol=1e-4, atol=1e-10)
    assert result.confidence_intervals.shape == (4, 2)
    for output in [result.fitted, result.residuals, *result.components]:
        assert output.units == source.units
        assert_array_equal(output.mask, ~support)
        assert_array_equal(output.x.data, source.x.data)
        assert_array_equal(output.y.data, source.y.data)
        assert output.history_entries[-1]["parameters"]["mode"] == "callable_model"
    assert_allclose(source.data, original.data)
    result.fitted.data[:] = 9
    assert not np.all(fit.predict().data == 9)
    result.fitted_parameters["a"] = 99
    assert fit.result.fitted_parameters["a"] != 99


def test_projected_amplitudes_match_joint_reference_and_count_degrees_of_freedom():
    source = grid()
    x, y = source.x._data[None, :], source.y._data[:, None]

    def basis(q):
        return [
            np.cos(y) * np.exp(-(((x - q * y * y) / 0.9) ** 2)),
            np.sin(y + 0.5) * np.exp(-(((x + 0.8 + q * y) / 0.7) ** 2)),
        ]

    source.data = 1.8 * basis(0.35)[0] + 0.6 * basis(0.35)[1]
    source.data += np.random.default_rng(9).normal(0, 0.003, source.shape)
    source[2:5, 6:10] = scp.MASKED
    support = ~source.mask

    def model(template, q):
        a, b = basis(q)
        return {"first": a, "second": b}

    options = {"ftol": 1e-12, "xtol": 1e-12, "gtol": 1e-12}
    reference = least_squares(
        lambda p: (source.data - p[1] * basis(p[0])[0] - p[2] * basis(p[0])[1])[
            support
        ],
        [0.2, 1, 1],
        bounds=([0, 0, 0], [1, np.inf, np.inf]),
        **options,
    )
    fit = scp.Optimize(
        model=model,
        parameters={"q": {"value": 0.2, "bounds": (0, 1)}},
        linear_amplitudes=True,
        solver_options=options,
    ).fit(source)
    result = fit.result
    assert_allclose(result.fitted_parameters["q"], reference.x[0], atol=1e-6)
    assert_allclose(list(result.amplitudes.values()), reference.x[1:], atol=1e-6)
    assert result.component_names == ("first", "second")
    assert_allclose(result.diagnostics["rss"], reference.fun @ reference.fun, rtol=1e-8)
    assert result.diagnostics["n_varying_parameters"] == 3
    assert result.diagnostics["n_nonlinear_parameters"] == 1
    assert result.diagnostics["n_linear_parameters"] == 2
    assert result.diagnostics["degrees_of_freedom"] == support.sum() - 3
    assert fit.jacobian.shape == (support.sum(), 1)
    assert result.parameter_names == ("q",)
    assert result.covariance is None
    assert result.diagnostics["covariance_reason"] == "profiled_amplitudes"
    assert_allclose(sum(part.data for part in result.components), result.fitted.data)


def test_units_fixed_parameters_and_linear_only_fit():
    source = grid()
    source.data = np.ones(source.shape) * 0.002

    def model(template, rate):
        assert rate == 4
        value = template.copy()
        value.units = "mV"
        value.data = np.ones(value.shape) * 2
        return value

    fit = scp.Optimize(
        model=model,
        parameters={"rate": {"value": 0.004 * scp.ur.kHz, "unit": "Hz", "fixed": True}},
    ).fit(source)
    assert_allclose(fit.predict().data, source.data)
    assert fit.result.fitted_parameters["rate"] == 4 * scp.ur.Hz
    assert fit.result.diagnostics["success"]
    assert fit.result.diagnostics["covariance_reason"] == "no_free_parameters"
    source.data = 2 * np.ones(source.shape)
    projected = scp.Optimize(
        model=lambda g: {"a": np.ones(g.shape), "b": -np.ones(g.shape)},
        linear_amplitudes=True,
    ).fit(source)
    assert_allclose(list(projected.result.amplitudes.values()), [2.0, 0.0], atol=1e-12)
    assert projected.result.diagnostics["active_amplitude_bounds"] == ("b",)
    assert projected.result.diagnostics["degrees_of_freedom"] == source.size - 2


@pytest.mark.parametrize(
    "kind", ["shape", "complex", "nonfinite", "mask", "coordinates"]
)
def test_invalid_callback_and_failed_refit_invalidate_results(kind):
    source = grid()
    state = [False]

    def model(template):
        if not state[0]:
            return np.ones(template.shape)
        if kind == "shape":
            return np.ones(3)
        if kind == "complex":
            return np.ones(template.shape, dtype=complex)
        if kind == "nonfinite":
            return np.full(template.shape, np.nan)
        if kind == "mask":
            template.mask = np.ones(template.shape, dtype=bool)
        else:
            template.x.data = template.x.data + 1
        return template

    fit = scp.Optimize(model=model).fit(source)
    state[0] = True
    with pytest.raises(ValueError):
        fit.fit(source)
    with pytest.raises(NotFittedError):
        _ = fit.result


@pytest.mark.parametrize(
    "spec",
    [
        {"x": {"value": np.nan}},
        {"x": {"value": 1, "bounds": (2, 3)}},
        {"x": {"value": 1, "scale": 0}},
        {"bad name": 1},
    ],
)
def test_invalid_parameters(spec):
    with pytest.raises(ValueError):
        scp.Optimize(model=lambda g, **p: np.zeros(g.shape), parameters=spec).fit(
            grid()
        )


def test_support_and_configuration_validation():
    source = grid()
    source.mask = np.ones(source.shape, dtype=bool)
    with pytest.raises(ValueError, match="observations"):
        scp.Optimize(model=lambda g: np.zeros(g.shape)).fit(source)
    for kwargs in [
        {"model": 3},
        {"model": lambda g: g, "peaks": []},
        {"parameters": {"x": 1}},
        {"model": lambda g: g, "script": ""},
    ]:
        with pytest.raises(ValueError):
            scp.Optimize(**kwargs)
    with pytest.raises(ValueError, match="Unsupported"):
        scp.Optimize(model=lambda g: g, solver_options={"method": "lm"}).fit(grid())
    with pytest.raises(ValueError, match="mapping"):
        scp.Optimize(model=lambda g: np.zeros(g.shape), linear_amplitudes=True).fit(
            grid()
        )


def test_component_topology_cannot_change():
    calls = [0]

    def model(template, x):
        calls[0] += 1
        return {str(calls[0]): np.ones(template.shape) * x}

    with pytest.raises(ValueError, match="remain fixed"):
        scp.Optimize(model=model, parameters={"x": 1}).fit(grid())


def test_component_outputs_are_snapshots_and_masked_model_values_can_be_nonfinite():
    source = grid()
    source[0, :] = scp.MASKED
    component = np.ones(source.shape)
    component[0, :] = np.nan
    fit = scp.Optimize(model=lambda g: component).fit(source)
    component[:] = 8
    assert_allclose(fit.result.components[0].data[1:], 1.0)
    assert_array_equal(fit.result.fitted.mask, source.mask)


def test_solver_limit_and_full_grid_three_dimensions():
    source = scp.NDDataset(
        np.ones((3, 4, 5)) * 4,
        coordset=[scp.Coord.arange(3), scp.Coord.arange(4), scp.Coord.arange(5)],
    )
    fit = scp.Optimize(
        model=lambda g, x: np.ones(g.shape) * x**2,
        parameters={"x": {"value": 0.1, "bounds": (0, 10)}},
        max_fun_calls=1,
    ).fit(source)
    assert not fit.result.diagnostics["success"]
    assert fit.result.diagnostics["ncalls"] == 1
    assert fit.result.diagnostics["covariance_reason"] == "not_converged"
    fit.max_fun_calls = 50
    fit.fit(source)
    assert_allclose(fit.result.fitted_parameters["x"], 2.0, atol=1e-5)
    assert fit.predict().shape == source.shape


def test_fixed_parameter_with_offset_unit():
    source = grid()
    source.data = np.ones(source.shape) * 25
    fit = scp.Optimize(
        model=lambda template, temperature: np.ones(template.shape) * temperature,
        parameters={
            "temperature": {"value": 298.15 * scp.ur.K, "unit": "degC", "fixed": True}
        },
    ).fit(source)
    assert_allclose(
        fit.result.fitted_parameters["temperature"].to("K").magnitude, 298.15
    )


def test_old_optimizer_state_does_not_require_new_model_attribute():
    optimizer = scp.Optimize()
    del optimizer._nd_model
    assert not optimizer._is_grid_fit
    assert optimizer.script


def test_unrepresentable_physical_residual_is_rejected():
    source = grid()
    source.data = np.ones(source.shape) * 1e308
    with pytest.raises(ValueError, match="Physical residuals"):
        scp.Optimize(model=lambda template: np.full(template.shape, -1e308)).fit(source)
