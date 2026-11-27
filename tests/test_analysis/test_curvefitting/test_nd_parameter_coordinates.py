"""Qualify ND preparation for coordinates without enabling ND log fitting."""

from copy import deepcopy

import numpy as np
import pytest
from scipy.optimize import least_squares

import spectrochempy as scp
from spectrochempy.analysis.curvefitting._optimize_model import _parameter_layout
from spectrochempy.analysis.curvefitting._optimize_nd import _prepare_peak_layout
from spectrochempy.analysis.curvefitting._optimize_profiles import _prepare_profiles
from spectrochempy.analysis.curvefitting._parameter_coordinates import (
    _ParameterCoordinates,
)
from spectrochempy.analysis.curvefitting._prepared_parameter import _ParameterLayout


@pytest.mark.parametrize("unit", ["V", "mV"])
@pytest.mark.parametrize("gain_scale", [1e-6, 1, 1e6])
@pytest.mark.parametrize("log_normalization", [0.25, 2.0])
def test_masked_grid_reference_and_existing_linear_route(
    unit, gain_scale, log_normalization
):
    """Use an analytical correlated problem, not the ND covariance machinery."""
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
    before = deepcopy(specs)
    prepared = _parameter_layout(specs)
    conditioning = (log_normalization, 0.5 * factor)
    coordinate_layout = _ParameterLayout(prepared.parameters, conditioning)
    reference = (0.5 / gain_scale * scp.ur.V).to(unit).magnitude
    space = _ParameterCoordinates(coordinate_layout, ("log", "lin"), (reference, None))
    assert prepared.free_names == ("z_gain", "a_tilt")
    assert prepared.free_scales == (factor / gain_scale, 0.5 * factor)
    assert prepared.parameters[1].fixed
    np.testing.assert_allclose(space.initial, [np.log(0.6) / log_normalization, -0.2])

    x = scp.Coord.linspace(1, -1, 17, rounding=False)
    y = scp.Coord.linspace(-0.6, 1.2, 13, rounding=False)
    xx, yy = np.meshgrid(x.data, y.data)
    source = scp.NDDataset(
        2 * xx - 0.3 * (yy + 1.2 * xx) + 0.2 + 0.01 * np.cos(3 * xx + 7 * yy),
        coordset=[y, x],
        units="V",
    )
    source[2:5, 1:8] = scp.MASKED
    source.data[0, 0] = np.nan
    original = source.copy()
    valid = ~source.mask & np.isfinite(source.data)
    design = np.column_stack((gain_scale * xx[valid], (yy + 1.2 * xx)[valid]))
    observed = source.to(unit).data[valid] - prepared.parameters[1].value
    intensity = np.max(np.abs(observed))

    def residual(vector):
        return (observed - design @ space.to_physical(vector)) / intensity

    limits = np.asarray(space.bounds)
    solution = least_squares(
        residual,
        space.initial,
        bounds=(limits[:, 0], limits[:, 1]),
        jac="3-point",
        ftol=1e-13,
        gtol=1e-13,
        xtol=1e-13,
    )
    assert solution.success
    physical = space.to_physical(solution.x)
    expected = np.linalg.lstsq(design, observed, rcond=None)[0]
    np.testing.assert_allclose(physical, expected, rtol=1e-7, atol=0)
    np.testing.assert_allclose(
        space.physical_jacobian(solution.jac * intensity, solution.x),
        -design,
        rtol=2e-6,
        atol=1e-11,
    )
    dof = len(observed) - len(physical)
    covariance = (
        np.linalg.inv(design.T @ design)
        * np.sum((observed - design @ expected) ** 2)
        / dof
    )
    assert abs(covariance[0, 1]) > 0.1 * np.sqrt(np.prod(np.diag(covariance)))
    solver_covariance = np.linalg.inv(solution.jac.T @ solution.jac) * (
        solution.fun @ solution.fun / dof
    )
    np.testing.assert_allclose(
        space.physical_covariance(solver_covariance, solution.x),
        covariance,
        rtol=3e-6,
        atol=0,
    )

    def model(template, z_gain, floor, a_tilt):
        tx, ty = template.x.data[None, :], template.y.data[:, None]
        return (gain_scale * z_gain * tx + a_tilt * (ty + 1.2 * tx) + floor) / factor

    fit = scp.Optimize(
        model=model,
        parameters=specs,
        solver_options={"ftol": 1e-12, "xtol": 1e-12, "gtol": 1e-12},
    ).fit(source)
    assert fit.result.parameter_names == prepared.free_names
    np.testing.assert_allclose(
        [
            fit.result.fitted_parameters[name].to(unit).magnitude
            for name in prepared.free_names
        ],
        expected,
        rtol=1e-7,
        atol=0,
    )
    np.testing.assert_allclose(fit.result.covariance, covariance, rtol=2e-5, atol=0)
    assert specs == before
    np.testing.assert_array_equal(source.data, original.data)
    np.testing.assert_array_equal(source.mask, original.mask)
    np.testing.assert_array_equal(fit.result.fitted.mask, ~valid)
    np.testing.assert_array_equal(fit.result.fitted.x.data, source.x.data)
    np.testing.assert_array_equal(fit.result.fitted.y.data, source.y.data)
    assert fit.result.fitted.units == scp.ur.V


@pytest.mark.parametrize("unit", ["s", "ms"])
def test_shared_profile_reference_is_preconverted_once_for_multiple_axis_units(unit):
    def decay(t, tau):
        return np.exp(-t / tau)

    peak = {
        "height": {"value": 1, "fixed": True},
        "axes": {
            dim: {
                "model": decay,
                "parameters": {"tau": {"shared": "tau", "unit": unit}},
            }
            for dim, unit in (("x", "ms"), ("y", "s"))
        },
    }
    peaks, groups, _ = _prepare_profiles(
        [peak],
        {
            "tau": {
                "value": 2 * scp.ur.s,
                "unit": unit,
                "bounds": (0.1 * scp.ur.s, 10 * scp.ur.s),
            }
        },
        None,
        {"x": scp.ur.ms, "y": scp.ur.s},
    )
    layout = _ParameterLayout(tuple(group["parameter"] for group in groups))
    reference = (500 * scp.ur.ms).to(layout.free_units[0]).magnitude
    space = _ParameterCoordinates(layout, ("log",), (reference,))
    assert len(space.initial) == 1
    assert layout.parameters[0].identifier == ("shared_parameters", "tau")
    assert len(groups[0]["bindings"]) == 2
    assert peaks[0]["axes"]["x"]["parameters"]["tau"] == 2000
    assert peaks[0]["axes"]["y"]["parameters"]["tau"] == 2
    np.testing.assert_allclose(space.initial, [np.log(4)])
    expected = (3 * scp.ur.s).to(unit).magnitude
    np.testing.assert_allclose(space.to_physical([np.log(6)]), [expected])
    assert peaks[0]["axes"]["x"]["parameters"]["tau"] == 2000


def test_historical_peak_layout_keeps_fixed_precision_and_axis_identities():
    center = np.longdouble("1000000.0000000001")
    spec = {
        "height": {"value": 1 * scp.ur.V, "bounds": (0.1, 4)},
        "axes": {
            "y": {
                "model": "gaussian",
                "center": {"value": center, "fixed": True},
                "width": {"value": 2, "fixed": True},
            },
            "x": {
                "model": "gaussian",
                "center": {"value": 0, "fixed": True},
                "width": {"value": 0.5, "bounds": (0.1, 2)},
            },
        },
    }
    peaks, bindings, layout = _prepare_peak_layout(
        [spec], scp.ur.V, {"y": scp.ur.s, "x": scp.ur.Hz}
    )
    space = _ParameterCoordinates(layout, ("log", "log"), (1, 0.25))
    assert layout.free_names == ("peaks[0].height", "peaks[0].axes['x'].width")
    assert len(bindings) == len(space.initial) == 2
    fixed = next(p for p in layout.parameters if p.identifier[-2:] == ("y", "center"))
    assert fixed.value == center and type(fixed.value) is type(center)
    np.testing.assert_array_equal(space.to_physical(space.initial), [1, 0.5])
    assert peaks[0]["axes"]["y"]["center"] == center


@pytest.mark.parametrize("lower", [None, 0, -1])
def test_nd_preparation_does_not_make_an_unqualified_log_domain_positive(lower):
    layout = _parameter_layout({"gain": {"value": 1, "bounds": (lower, 10)}})
    with pytest.raises(ValueError, match="positive finite lower bound"):
        _ParameterCoordinates(layout, ("log",), (1,))
