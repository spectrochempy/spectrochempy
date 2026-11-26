"""Qualify the shared layout on existing native ND profile workflows."""

from copy import deepcopy

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.analysis.curvefitting._prepared_parameter import _ParameterLayout


def flat(axis):
    return np.ones_like(axis)


def gaussian(axis, center, width):
    return np.exp(-4 * np.log(2) * ((axis - center) / width) ** 2)


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("sharing", [False, True])
def test_profile_layout_matches_physical_linear_reference(
    transpose, sharing, monkeypatch
):
    layouts = []
    original = _ParameterLayout.__post_init__

    def record(layout):
        original(layout)
        layouts.append(layout)

    monkeypatch.setattr(_ParameterLayout, "__post_init__", record)
    x = scp.Coord.linspace(-3, 3, 17, units="Hz", rounding=False)[::-1]
    y = scp.Coord.linspace(-1, 1, 9, units="s", rounding=False)
    basis = [
        np.broadcast_to(gaussian(x.data[None, :], center, width), (9, 17)).copy()
        for center, width in [(-1.0, 0.7), (0.2, 0.8), (1.3, 0.5)]
    ]
    columns = [basis[0], (basis[1] + basis[2]) * 0.001] if sharing else basis
    truth = [1.8, 1200.0] if sharing else [1.8, 1.2, 0.9]
    cube = np.stack(columns, axis=-1)
    data = cube @ truth + 0.001 * np.cos(3 * x.data[None, :] + 4 * y.data[:, None])
    mask = np.zeros(data.shape, dtype=bool)
    mask[2:4, 4:7] = True
    data[0, 0] = np.nan
    observed = scp.NDDataset(data, coordset=[y, x], mask=mask, units="V")
    repeated = {"value": 500 * scp.ur.mV, "bounds": (0, 2000 * scp.ur.mV)}
    peaks = []
    for i, (center, width) in enumerate([(-1.0, 0.7), (0.2, 0.8), (1.3, 0.5)]):
        height = (
            {"value": 700 * scp.ur.mV, "bounds": (0, 3)}
            if i == 0
            else {"shared": "level"}
            if sharing
            else repeated
        )
        peaks.append(
            {
                "height": height,
                "axes": {
                    "y": {"model": flat, "parameters": {}},
                    "x": {
                        "model": "gaussian",
                        "center": {"value": center, "fixed": True},
                        "width": {"value": width, "fixed": True},
                    },
                },
            }
        )
    shared = (
        {"level": {"value": 500 * scp.ur.mV, "unit": "mV", "bounds": (0, 2000)}}
        if sharing
        else {}
    )
    initial = deepcopy(peaks)
    shared_initial = deepcopy(shared)
    if transpose:
        observed = observed.T
        cube = cube.transpose(1, 0, 2)
    valid = ~observed.mask & np.isfinite(observed.data)
    design = cube[valid]
    target = observed.data[valid]
    expected = np.linalg.lstsq(design, target, rcond=None)[0]
    residual = target - design @ expected
    covariance = (
        np.linalg.inv(design.T @ design)
        * (residual @ residual)
        / (target.size - len(expected))
    )
    opt = scp.Optimize(peaks=peaks, shared_parameters=shared).fit(observed)
    result = opt.result
    layout = layouts[-1]
    names = (
        ("peaks[0].height", "shared_parameters['level']")
        if sharing
        else tuple(f"peaks[{i}].height" for i in range(3))
    )
    assert layout.free_names == result.parameter_names == names
    assert tuple(p.identifier for p in layout.parameters) == (
        (("peaks", 0, "height"), ("shared_parameters", "level"))
        if sharing
        else tuple(("peaks", i, "height") for i in range(3))
    )
    assert layout.free_units == result.parameter_units
    assert (
        layout.free_units == (scp.ur.V, scp.ur.mV)
        if sharing
        else layout.free_units == (scp.ur.V,) * 3
    )
    np.testing.assert_allclose(
        layout.free_values, [0.7, 500] if sharing else [0.7, 0.5, 0.5], rtol=1e-15
    )
    np.testing.assert_allclose(layout.free_scales, layout.free_values, rtol=1e-15)
    assert all(isinstance(value, np.longdouble) for value in layout.free_scales)
    np.testing.assert_allclose(
        layout.free_bounds, [(0, 3), (0, 2000)] if sharing else [(0, 3), (0, 2), (0, 2)]
    )
    np.testing.assert_allclose(opt.jacobian, -design, rtol=3e-6, atol=1e-9)
    np.testing.assert_allclose(result.covariance, covariance, rtol=3e-5, atol=1e-10)
    np.testing.assert_allclose(
        result.fitted.data[valid], design @ expected, rtol=1e-7, atol=1e-8
    )
    np.testing.assert_array_equal(result.fitted.mask, ~valid)
    assert peaks == initial and shared == shared_initial
    if not sharing:
        assert peaks[1]["height"] is peaks[2]["height"]
        assert layout.parameters[1].identifier != layout.parameters[2].identifier
    first = result.fitted.data.copy()
    other = scp.Optimize(peaks=peaks, shared_parameters=shared).fit(observed)
    np.testing.assert_array_equal(other.result.fitted.data, first)
    assert peaks == initial and shared == shared_initial


@pytest.mark.skipif(
    np.finfo(np.longdouble).nmant <= np.finfo(float).nmant,
    reason="longdouble has no additional mantissa precision",
)
def test_profile_layout_keeps_longdouble_center_and_width_conditioning(monkeypatch):
    layouts = []
    original = _ParameterLayout.__post_init__

    def record(layout):
        original(layout)
        layouts.append(layout)

    monkeypatch.setattr(_ParameterLayout, "__post_init__", record)
    center = np.longdouble("1000000000000.0001")
    width = np.longdouble(".001")
    x = scp.Coord(
        center + np.linspace(-2, 2, 31, dtype=np.longdouble) * width,
        units="Hz",
        rounding=False,
    )
    y = scp.Coord([-1.0, 0.0, 1.0], units="s")
    data = np.broadcast_to(gaussian(x.data[None, :], center, width), (3, 31)).copy()
    observed = scp.NDDataset(data, coordset=[y, x])
    peaks = [
        {
            "height": {"value": 1.0, "fixed": True},
            "axes": {
                "x": {
                    "model": "gaussian",
                    "center": {
                        "value": center,
                        "bounds": (center - width, center + width),
                    },
                    "width": {"value": width, "fixed": True},
                },
                "y": {"model": flat, "parameters": {}},
            },
        }
    ]
    scp.Optimize(peaks=peaks).fit(observed)
    layout = layouts[-1]
    assert layout.free_values == (center,)
    assert isinstance(layout.free_values[0], np.longdouble)
    assert layout.free_values[0] != np.longdouble(float(center))
    assert layout.free_scales == (width,)
    assert layout.free_bounds == ((center - width, center + width),)


def test_all_fixed_profiles_keep_an_empty_free_layout(monkeypatch):
    layouts = []
    original = _ParameterLayout.__post_init__

    def record(layout):
        original(layout)
        layouts.append(layout)

    monkeypatch.setattr(_ParameterLayout, "__post_init__", record)
    x = scp.Coord.linspace(-2, 2, 17, rounding=False)
    y = scp.Coord([-1.0, 0.0, 1.0])
    observed = scp.NDDataset(np.ones((3, 17)), coordset=[y, x])
    peaks = [
        {
            "height": {"value": 1.0, "fixed": True},
            "axes": {
                "x": {"model": flat, "parameters": {}},
                "y": {"model": flat, "parameters": {}},
            },
        }
    ]
    opt = scp.Optimize(peaks=peaks).fit(observed)
    assert (
        layouts[-1].free_names
        == layouts[-1].free_values
        == layouts[-1].free_bounds
        == ()
    )
    assert layouts[-1].free_units == layouts[-1].free_scales == ()
    assert opt.result.parameter_names == ()
    assert opt.result.covariance is None and opt.jacobian is None
    np.testing.assert_array_equal(opt.result.fitted.data, observed.data)
