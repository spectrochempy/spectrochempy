"""Analytical qualifications for physical 1D least-squares uncertainty."""

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.analysis.curvefitting._modelspec import _FitModelSpec
from spectrochempy.analysis.curvefitting._modelspec import _ParamSpec
from spectrochempy.analysis.curvefitting._parameter_space import _internal_vector
from spectrochempy.analysis.curvefitting._parameter_space import _transform_artifacts
from spectrochempy.analysis.curvefitting._parameter_transform import (
    _external_transform_state,
)
from spectrochempy.analysis.curvefitting._parameter_transform import _to_external
from spectrochempy.analysis.curvefitting._parameters import FitParameters
from spectrochempy.analysis.curvefitting.optimize import _compute_covariance_matrix
from spectrochempy.analysis.curvefitting.optimize import _optimize


def _line(x, slope=1.0, intercept=0.0):
    return slope * x + intercept


def _fit_line(x, y, bounds, *, frontend="python", mask=False, free_intercept=False):
    data = scp.NDDataset(y, coordset=[scp.Coord(x)])
    if mask:
        data[[2, 8, 14]] = scp.MASKED
    if frontend == "python":
        model = scp.Linear()
        params = model.make_params(slope=1.0, intercept=0.1 if free_intercept else 0)
        params["slope"].set(min=bounds[0], max=bounds[1])
        params["intercept"].set(vary=free_intercept)
        return scp.Optimize(model=model, params=params).fit(data)
    lo, hi = ("none" if value is None else value for value in bounds)
    intercept = "$ intercept: 0.1, none, none" if free_intercept else "* intercept: 0"
    script = f"""
COMMON:
{intercept}
$ slope: 1, {lo}, {hi}
MODEL: line
shape: customline
> intercept: intercept
> slope: slope
"""
    fit = scp.Optimize(
        usermodels={"customline": _line}, script="", amplitude_mode="area"
    )
    fit.script = script
    return fit.fit(data)


@pytest.mark.parametrize("frontend", ["python", "common"])
@pytest.mark.parametrize("bounds", [(None, None), (0, None), (None, 4), (0, 4)])
def test_scalar_covariance_matches_physical_linear_reference(frontend, bounds):
    x = np.linspace(0, 1, 21)
    y = 2 * x + 0.01 * np.sin(np.arange(x.size))
    fit = _fit_line(x, y, bounds, frontend=frontend)
    result = fit.result
    expected = result.rss / (x.size - 1) / np.dot(x, x)
    np.testing.assert_allclose(result.covariance, [[expected]], rtol=2e-6)
    np.testing.assert_allclose(result.stderr, [np.sqrt(expected)], rtol=1e-6)
    assert result.diagnostics["covariance_coordinates"] == "physical"
    assert result.diagnostics["jacobian_coordinates"] == "optimizer"
    assert result.diagnostics["covariance_reason"] is None
    assert not result.covariance.flags.writeable
    raw_covariance = result.rss / (x.size - 1) / (fit.jacobian.T @ fit.jacobian)
    if bounds == (0, 4):
        np.testing.assert_allclose(raw_covariance / result.covariance, 0.25, rtol=1e-5)


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_correlated_covariance_order_masks_axes_and_common_parity(descending, masked):
    x = np.linspace(0.3, 1.2, 31)
    if descending:
        x = x[::-1]
    y = 2 * x + 0.3 + 0.01 * np.sin(np.arange(x.size))
    fits = [
        _fit_line(x, y, (0, 4), frontend=frontend, mask=masked, free_intercept=True)
        for frontend in ("python", "common")
    ]
    valid = np.ones(x.size, dtype=bool)
    if masked:
        valid[[2, 8, 14]] = False
    design = np.column_stack((np.ones(valid.sum()), x[valid]))
    for fit in fits:
        result = fit.result
        expected = result.rss / (valid.sum() - 2) * np.linalg.inv(design.T @ design)
        np.testing.assert_allclose(result.covariance, expected, rtol=3e-6)
        assert result.covariance[0, 1] < 0
        values = np.linalg.lstsq(design, y[valid], rcond=None)[0]
        np.testing.assert_allclose(
            result.confidence_intervals.mean(axis=1), values, atol=1e-8
        )
    assert fits[0].result.parameter_names == ("intercept", "slope")
    np.testing.assert_allclose(
        fits[0].result.covariance, fits[1].result.covariance, rtol=3e-6
    )


@pytest.mark.parametrize(
    "bounds,target", [((0, None), -1), ((None, 4), 5), ((0, 4), -1)]
)
def test_bound_optima_do_not_report_unconstrained_uncertainty(bounds, target):
    x = np.linspace(0, 1, 31)
    fit = _fit_line(x, target * x, bounds)
    result = fit.result
    assert result.diagnostics["covariance_reason"] == "active_bounds"
    for name in (
        "covariance",
        "variance",
        "stderr",
        "correlation",
        "confidence_intervals",
    ):
        assert getattr(result, name) is None
    assert fit.jacobian is not None


def test_legacy_endpoint_false_stop_does_not_invent_zero_uncertainty():
    x = np.linspace(0, 1, 31)
    fit = scp.Optimize(
        usermodels={"customline": _line}, script="", amplitude_mode="area"
    )
    fit.script = """
COMMON:
$ slope: 0, 0, none
MODEL: line
shape: customline
> slope: slope
* intercept: 0
"""
    fit.fit(scp.NDDataset(2 * x, coordset=[scp.Coord(x)]))
    assert fit.result.rss > 1
    assert fit.result.covariance is None
    assert fit.result.diagnostics["covariance_reason"] == "active_bounds"


@pytest.mark.parametrize("u", [-4.0, -0.7, 0.5, 4.0])
@pytest.mark.parametrize("bounds", [(None, None), (0, None), (None, 4), (0, 4)])
def test_signed_transform_derivative_including_folded_branches(u, bounds):
    derivative, near = _external_transform_state(u, *bounds)
    step = 1e-5
    numerical = (_to_external(u + step, *bounds) - _to_external(u - step, *bounds)) / (
        2 * step
    )
    assert derivative == pytest.approx(numerical, rel=1e-8)
    assert not near


@pytest.mark.parametrize(
    "reason", ["rank_deficient", "not_converged", "numeric_range", "singular_transform"]
)
def test_covariance_exclusions_are_explicit(reason):
    observed = scp.NDDataset([1.0, 2.0, 3.0, 4.0])
    fitted = scp.NDDataset([1.1, 2.1, 3.1, 4.1])
    jacobian = np.column_stack((np.ones(4), np.arange(4)))
    derivatives = [1.0, 1.0]
    diagnostics = {
        "degrees_of_freedom": 2,
        "reduced_chi_square": 0.02,
        "n_varying_parameters": 2,
        "success": True,
    }
    if reason == "rank_deficient":
        jacobian[:, 1] = 2 * jacobian[:, 0]
    elif reason == "not_converged":
        diagnostics["success"] = False
    elif reason == "numeric_range":
        jacobian[0, 0] = np.nan
    else:
        derivatives[0] = 0
    covariance = _compute_covariance_matrix(
        observed, fitted, jacobian, diagnostics, transform_derivatives=derivatives
    )
    assert covariance is None
    assert diagnostics["covariance_reason"] == reason


@pytest.mark.parametrize("derivatives", [(-2.0, 3.0), (1e170, 1e-170)])
def test_covariance_handles_signed_and_extreme_coordinate_scales(derivatives):
    observed = scp.NDDataset([1.0, 2.0, 3.0, 4.0])
    fitted = observed + 0.1
    design = np.column_stack((np.ones(4), np.arange(4)))
    diagnostics = {
        "degrees_of_freedom": 2,
        "reduced_chi_square": 0.02,
        "n_varying_parameters": 2,
        "success": True,
    }
    covariance = _compute_covariance_matrix(
        observed,
        fitted,
        design * derivatives,
        diagnostics,
        transform_derivatives=derivatives,
    )
    expected = 0.02 * np.linalg.inv(design.T @ design)
    np.testing.assert_allclose(covariance, expected, rtol=1e-14)


def test_nonidentifiable_python_model_has_no_parameter_uncertainty():
    def redundant(x, first=1.0, second=1.0):
        return (first + second) * x

    x = np.linspace(0, 1, 31)
    data = scp.NDDataset(
        2 * x + 0.01 * np.sin(np.arange(x.size)), coordset=[scp.Coord(x)]
    )
    fit = scp.Optimize(model=scp.Model(redundant)).fit(data)
    assert fit.result.covariance is None
    assert fit.result.diagnostics["covariance_reason"] == "rank_deficient"


@pytest.mark.parametrize("legacy", [False, True])
def test_transform_snapshot_follows_experiment_order_and_excludes_fixed(legacy):
    spec = _FitModelSpec(
        common_params={
            "a": _ParamSpec("a", [2.0, 4.0], bounds=(0, 10)),
            "z": _ParamSpec("z", 5.0),
            "fixed": _ParamSpec("fixed", 0.0, bounds=(0, 10), vary=False),
        },
        expvars=["a"],
        expnumber=2,
    )
    definition = spec.to_fitparameters() if legacy else spec
    vector, names = _internal_vector(definition)
    assert names == ["a_exp0", "a_exp1", "z"]
    snapshot = _transform_artifacts(definition, vector, names)
    np.testing.assert_allclose(
        snapshot["parameter_transform_derivatives"], [4.0, np.sqrt(24), 1.0]
    )
    assert snapshot["near_parameter_bounds"] is False


@pytest.mark.parametrize("n_experiments", [1, 2])
def test_legacy_experiment_fit_retains_transform_after_scalar_restoration(
    n_experiments,
):
    values = [1.0] * n_experiments
    target = np.array([2.0, 2.5][:n_experiments])
    definition = FitParameters()
    definition["a"] = (values, 0, 4, False)
    definition.expvars = ["a"]
    definition.expnumber = n_experiments
    x = np.linspace(0, 1, 11)

    def residual(parameters, coordinate):
        physical = np.atleast_1d(parameters["a"])
        return ((physical - target)[:, None] * coordinate).ravel()

    fitted, _, diagnostics, artifacts = _optimize(residual, definition, args=(x,))
    assert diagnostics["success"]
    np.testing.assert_allclose(fitted["a"], target if n_experiments > 1 else target[0])
    assert np.isscalar(fitted["a"]) is (n_experiments == 1)
    np.testing.assert_allclose(
        artifacts["parameter_transform_derivatives"], np.sqrt(target * (4 - target))
    )
    assert artifacts["jacobian"].shape == (x.size * n_experiments, n_experiments)
