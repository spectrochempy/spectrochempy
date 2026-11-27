"""Shared metadata for the historical separable-peak adapter."""

import pickle
from copy import deepcopy

import numpy as np
import pytest

import spectrochempy as scp
from spectrochempy.analysis.curvefitting import _optimize_nd
from spectrochempy.analysis.curvefitting._prepared_parameter import _ParameterLayout


def fixed(value):
    return {"value": value, "fixed": True}


def test_parser_layout_preserves_integer_precision_and_legacy_protocol():
    exact = 2**53 + 1
    precise = np.longdouble("0.1234567890123456789")
    specs = [
        {
            "height": fixed(1),
            "axes": {
                "y": {"model": "gaussian", "center": fixed(precise), "width": fixed(1)},
                "x": {
                    "model": "gaussian",
                    "center": {"value": exact, "bounds": (exact - 3, exact + 5)},
                    "width": fixed(1),
                },
            },
        }
    ]
    initial = deepcopy(specs)
    peaks, free, layout = _optimize_nd._prepare_peak_layout(
        specs, None, {"y": None, "x": None}
    )
    assert layout.free_names == ("peaks[0].axes['x'].center",)
    assert [p.identifier for p in layout.parameters] == [
        ("peaks", 0, "height"),
        ("peaks", 0, "axes", "x", "center"),
        ("peaks", 0, "axes", "x", "width"),
        ("peaks", 0, "axes", "y", "center"),
        ("peaks", 0, "axes", "y", "width"),
    ]
    assert layout.free_values == (exact,)
    assert isinstance(layout.free_values[0], int)
    assert layout.parameters[3].value == precise
    assert isinstance(layout.parameters[3].value, np.longdouble)
    assert layout.free_bounds == ((exact - 3, exact + 5),)
    assert [
        _optimize_nd._relative_value(bound, exact) for bound in layout.free_bounds[0]
    ] == [-3, 5]
    assert free[0][0] is peaks[0]["axes"]["x"]
    free[0][0][free[0][1]] = exact + 10
    assert layout.free_values == (exact,)
    assert specs == initial
    for restored in (deepcopy(layout), pickle.loads(pickle.dumps(layout))):  # noqa: S301
        assert restored == layout
    legacy_peaks, legacy_free, limits, names, units = _optimize_nd._prepare_peaks(
        specs, None, {"x": None, "y": None}
    )
    assert isinstance(limits, list) and limits == list(layout.free_bounds)
    assert names == layout.free_names and units == layout.free_units
    assert legacy_free[0][0] is legacy_peaks[0]["axes"]["x"]
    assert legacy_free[0][0]["center"] == exact


@pytest.mark.parametrize("transpose", [False, True])
def test_native_height_fit_matches_analytical_reference_and_independent_runs(
    transpose, monkeypatch
):
    layouts = []
    original = _ParameterLayout.__post_init__

    def record(layout):
        original(layout)
        layouts.append(layout)

    monkeypatch.setattr(_ParameterLayout, "__post_init__", record)

    def forbidden(*args, **kwargs):
        pytest.fail(
            "The core fitter must use its prepared layout rather than unpacking legacy metadata"
        )

    monkeypatch.setattr(_optimize_nd, "_prepare_peaks", forbidden)
    x = scp.Coord.linspace(-3, 3, 17, units="Hz", rounding=False)[::-1]
    y = scp.Coord.linspace(-2, 2, 13, units="s", rounding=False)
    centers = [(-0.8, -0.3), (0.9, 0.4)]
    widths = [(0.7, 0.9), (1.1, 0.8)]

    def profile(center, width):
        return np.exp(
            -4
            * np.log(2)
            * (
                ((x._data[None, :] - center[0]) / width[0]) ** 2
                + ((y._data[:, None] - center[1]) / width[1]) ** 2
            )
        )

    cube = np.stack(
        [profile(c, w) for c, w in zip(centers, widths, strict=True)], axis=-1
    )
    data = cube @ np.array([1.6, 0.9]) + 0.001 * np.cos(
        3 * x._data[None, :] + 4 * y._data[:, None]
    )
    mask = np.zeros(data.shape, dtype=bool)
    mask[2:4, 4:7] = True
    data[0, 0] = np.nan
    observed = scp.NDDataset(data, coordset=[y, x], mask=mask, units="V")
    height = {"value": 700 * scp.ur.mV, "bounds": (0, 3)}
    specs = [
        {
            "height": height,
            "axes": {
                "y": {"model": "gaussian", "center": fixed(c[1]), "width": fixed(w[1])},
                "x": {"model": "gaussian", "center": fixed(c[0]), "width": fixed(w[0])},
            },
        }
        for c, w in zip(centers, widths, strict=True)
    ]
    initial = deepcopy(specs)
    if transpose:
        observed = observed.T
        cube = cube.transpose(1, 0, 2)
    valid = ~observed.mask & np.isfinite(observed.data)
    design = cube[valid]
    target = observed.data[valid]
    coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    residual = target - design @ coefficients
    covariance = (
        np.linalg.inv(design.T @ design) * (residual @ residual) / (target.size - 2)
    )
    opt = scp.Optimize(peaks=specs).fit(observed)
    result = opt.result
    layout = layouts[-1]
    assert (
        layout.free_names
        == result.parameter_names
        == ("peaks[0].height", "peaks[1].height")
    )
    assert len(layout.parameters) == 10
    assert tuple(p.identifier for p in layout.free_parameters) == (
        ("peaks", 0, "height"),
        ("peaks", 1, "height"),
    )
    assert layout.free_units == result.parameter_units == (scp.ur.V, scp.ur.V)
    np.testing.assert_allclose(layout.free_values, [0.7, 0.7], rtol=1e-15)
    np.testing.assert_allclose(layout.free_scales, [0.7, 0.7], rtol=1e-15)
    assert all(isinstance(v, np.longdouble) for v in layout.free_scales)
    np.testing.assert_allclose(opt.jacobian, -design, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(result.covariance, covariance, rtol=3e-6, atol=1e-12)
    np.testing.assert_allclose(
        result.fitted.data[valid], design @ coefficients, rtol=1e-8, atol=1e-9
    )
    assert specs[0]["height"] is specs[1]["height"]
    assert specs == initial
    first = result.fitted.data.copy()
    second_data = observed.copy()
    second_data.data = observed.data * 0.8
    second = scp.Optimize(peaks=specs).fit(second_data)
    np.testing.assert_array_equal(result.fitted.data, first)
    np.testing.assert_allclose(
        second.result.fitted.data[valid], 0.8 * first[valid], rtol=1e-7, atol=1e-8
    )
    assert specs == initial


def test_all_fixed_layout_and_extended_dispatch_are_preserved(monkeypatch):
    x = scp.Coord.linspace(-2, 2, 9, rounding=False)
    y = scp.Coord.linspace(-1, 1, 7, rounding=False)
    grid = scp.NDDataset(np.zeros((7, 9)), coordset=[y, x])
    specs = [
        {
            "height": fixed(1),
            "axes": {
                dim: {"model": "gaussian", "center": fixed(0), "width": fixed(1)}
                for dim in grid.dims
            },
        }
    ]
    observed = scp.simulate_peaks(
        grid,
        [
            {
                "height": 1,
                "axes": {
                    dim: {"model": "gaussian", "center": 0, "width": 1}
                    for dim in grid.dims
                },
            }
        ],
    )
    _, free, layout = _optimize_nd._prepare_peak_layout(
        specs, None, {"x": None, "y": None}
    )
    assert not free
    assert len(layout.parameters) == 5
    assert (
        layout.free_values
        == layout.free_bounds
        == layout.free_names
        == layout.free_units
        == ()
    )
    opt = scp.Optimize(peaks=specs).fit(observed)
    assert opt.result.parameter_names == () and opt.jacobian is None
    np.testing.assert_array_equal(opt.result.fitted.data, observed.data)

    def forbidden(*args, **kwargs):
        pytest.fail("Explicit shared_parameters must retain the profile adapter route")

    monkeypatch.setattr(_optimize_nd, "_prepare_peak_layout", forbidden)
    extended = scp.Optimize(peaks=specs, shared_parameters={}).fit(observed)
    np.testing.assert_array_equal(extended.result.fitted.data, observed.data)
